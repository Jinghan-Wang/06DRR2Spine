import time
import os
from util.config_loader import ConfigLoader

ConfigLoader.apply("train")

from data import create_dataset
from models import create_model
from options.train_options import TrainOptions
from util.util import cleanup_ddp, init_ddp
from util.visualizer import Visualizer
import setproctitle
setproctitle.setproctitle("WJH_Student")

def _unwrap_singleton(value):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


if __name__ == "__main__":
    start_time = time.perf_counter()
    opt = TrainOptions().parse()
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

    opt.device = init_ddp()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print(f"The number of training images = {dataset_size}")

    model = create_model(opt)
    model.setup(opt)

    initial_lr = opt.lr
    warmup_epochs = 10
    visualizer = Visualizer(opt)
    total_iters = 0

    for epoch in range(opt.epoch_count, final_epoch + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        current_a_paths = None
        current_b_paths = None
        current_aug_params = None
        visualizer.reset()

        model.set_epoch(epoch)
        if opt.continue_train and epoch <= opt.epoch_count + warmup_epochs:
            current_lr = initial_lr * (epoch - opt.epoch_count + 1) / (warmup_epochs + 1)
            for optimizer in model.optimizers:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr
            print(
                f">>> Warm-up stage: lr={current_lr:.6f}, "
                f"epoch={epoch}, epoch_count={opt.epoch_count}"
            )

        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        for _, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            current_a_paths = _unwrap_singleton(data.get("A_paths"))
            current_b_paths = _unwrap_singleton(data.get("B_paths"))
            current_aug_params = _unwrap_singleton(data.get("aug_params"))

            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(
                    model.get_current_visuals(),
                    epoch,
                    total_iters,
                    save_result,
                )

            if total_iters % opt.print_freq == 0:
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

            if total_iters % opt.save_latest_freq == 0:
                print(f"saving the latest model (epoch {epoch}, total_iters {total_iters})")
                save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch_iter > 0:
            model.compute_visuals()
            current_visuals = model.get_current_visuals()
            visualizer.display_current_results(current_visuals, epoch, total_iters, save_result=True)
            visualizer.save_epoch_medical_results(
                current_visuals,
                epoch,
                current_a_paths,
                current_b_paths,
            )

        model.update_learning_rate()

        if epoch % opt.save_epoch_freq == 0:
            print(f"saving the model at the end of epoch {epoch}, iters {total_iters}")
            model.save_networks("latest")
            model.save_networks(epoch)

        print(
            f"End of epoch {epoch} / {final_epoch} "
            f"\t Time Taken: {time.time() - epoch_start_time:.02f} sec"
        )

    cleanup_ddp()
    elapsed = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed:.2f} s")
