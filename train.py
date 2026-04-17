
import time
from util.config_loader import ConfigLoader
ConfigLoader.apply("train")   # ← 必须在框架 import 之前执行
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import init_ddp, cleanup_ddp


if __name__ == "__main__":
    start = time.perf_counter()
    opt = TrainOptions().parse()  # get training options
    final_epoch = opt.n_epochs + opt.n_epochs_decay
    if opt.epoch_count > final_epoch:
        print(
            "Warning: no training epochs will run because "
            f"epoch_count ({opt.epoch_count}) is greater than the training end epoch ({final_epoch})."
        )
        print(
            f"Current resume target is epoch='{opt.epoch}'. "
            "Please increase n_epochs/n_epochs_decay or lower epoch_count before continuing."
        )
        raise SystemExit(0)
    opt.device = init_ddp() # initialize distributed training
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print(f"The number of training images = {dataset_size}")
    #print(f"Training begins! Use the gpus_ids: {opt.gpu_ids}")
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)

    # 定义热启动参数 (针对从单图过拟合切换到全量训练)
    initial_lr = opt.lr  # 论文建议 2e-04 [cite: 233]
    warmup_epochs = 10  # 设置 5 个 epoch 进行平滑过渡



    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()

        model.set_epoch(epoch)
        # --- 新增：Warm-up 学习率调节逻辑 ---
        # 防止模型在接触大量新数据的第一轮时崩溃
        if opt.continue_train and epoch <= opt.epoch_count + warmup_epochs:
            # 学习率从极小值线性提升到初始设定值
            current_lr = initial_lr * (epoch - opt.epoch_count + 1) / (warmup_epochs + 1)
            for optimizer in model.optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            print(f'>>> Warm-up 阶段: 当前学习率提升至 {current_lr:.6f}, epoch: {epoch}, epoch_count: {opt.epoch_count}')
        # pass
        # Set epoch for DistributedSampler
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            current_a_paths = data.get("A_paths")
            current_b_paths = data.get("B_paths")
            current_aug_params = data.get("aug_params")
            if isinstance(current_a_paths, (list, tuple)) and len(current_a_paths) == 1:
                current_a_paths = current_a_paths[0]
            if isinstance(current_b_paths, (list, tuple)) and len(current_b_paths) == 1:
                current_b_paths = current_b_paths[0]
            if isinstance(current_aug_params, (list, tuple)) and len(current_aug_params) == 1:
                current_aug_params = current_aug_params[0]
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, total_iters, save_result)

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(
                    epoch,
                    epoch_iter,
                    losses,
                    t_comp,
                    t_data,
                    current_a_paths,
                    current_b_paths,
                    current_aug_params,
                )
                visualizer.plot_current_losses(total_iters, losses)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print(f"saving the latest model (epoch {epoch}, total_iters {total_iters})")
                save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        model.update_learning_rate()  # update learning rates at the end of every epoch

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print(f"saving the model at the end of epoch {epoch}, iters {total_iters}")
            model.save_networks("latest")
            model.save_networks(epoch)

        print(f"End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time:.02f} sec")

    cleanup_ddp()
    #计时
    end = time.perf_counter()
    interval = end - start
    print(f"耗时：{interval:.2f} 秒")
