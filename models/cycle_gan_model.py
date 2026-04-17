import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

def get_gradients(image):
    # 论文：计算图像在 H 和 W 维度上的梯度,用于保护骨骼边缘纹理
    # image shape: [B, C, H, W]
    dy = image[:, :, 1:, :] - image[:, :, :-1, :]
    dx = image[:, :, :, 1:] - image[:, :, :, :-1]
    return dy, dx

class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument("--lambda_A", type=float, default=10.0, help="weight for cycle loss (A -> B -> A)")
            parser.add_argument("--lambda_B", type=float, default=10.0, help="weight for cycle loss (B -> A -> B)")
            parser.add_argument("--lambda_gan", type=float, default=0.5, help="weight for adversarial loss on G_A")
            parser.add_argument("--lambda_L1_supervised", type=float, default=50.0, help="weight for supervised L1 loss on fake_B")
            parser.add_argument("--lambda_grad", type=float, default=5.0, help="weight for gradient consistency loss")
            parser.add_argument("--lambda_mean", type=float, default=0.0, help="weight for mean intensity loss")
            parser.add_argument("--gan_start_epoch", type=int, default=5, help="start adversarial loss after this epoch")
            parser.add_argument("--d_update_ratio", type=int, default=2, help="update D once every N generator steps")
            parser.add_argument("--lr_D_scale", type=float, default=0.5, help="scale factor applied to discriminator learning rate")
            parser.add_argument(
                "--lambda_identity",
                type=float,
                default=0.5,
                help="use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1",
            )

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        #self.loss_names = ["D_A", "G_A", "cycle_A", "idt_A", "D_B", "G_B", "cycle_B", "idt_B"]
        # 论文增加：新损失名
        #self.loss_names = ['G_A', 'G_B', 'cycle_A', 'cycle_B', 'G_L1_Supervised', 'G_gradient', 'D_A', 'D_B']
        # 修改 loss 名字，只保留 A 域相关的（即 A -> B 方向）
        self.loss_names = ['G_A', 'D_A', 'G_L1_Supervised', 'G_gradient', 'G_mean']
        self.current_epoch = opt.epoch_count - 1 if self.isTrain else 0
        self.global_step = 0
        if self.isTrain:
            self.visual_names = ['real_A', 'fake_B', 'real_B']
        else:
            # 测试时只需要看输入(real_A)和结果(fake_B) 如果测试集没有标签，要把 'real_B' 去掉，否则会报同样的错
            self.visual_names = ['real_A', 'fake_B']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        #visual_names_A = ["real_A", "fake_B", "rec_A"]
        #visual_names_B = ["real_B", "fake_A", "rec_B"]
        #if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_B(A)
        #    visual_names_A.append("idt_B")
        #    visual_names_B.append("idt_A")
#
        #self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            #self.model_names = ["G_A", "G_B", "D_A", "D_B"]
            self.model_names = ['G_A', 'D_A']
        else:  # during test time, only load Gs
            #self.model_names = ["G_A", "G_B"]
            self.model_names = ['G_A']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)
        #self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)

        #if self.isTrain:
        #    if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
        #        assert opt.input_nc == opt.output_nc
        #    self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
        #    self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
        #    # define loss functions
        #    self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
        #    self.criterionCycle = torch.nn.L1Loss()
        #    self.criterionIdt = torch.nn.L1Loss()
        #    # 【论文:新增这一行】定义有监督 L1 损失器
        #    self.criterionL1 = torch.nn.L1Loss()
        #    # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        #    self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        #    self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        #    self.optimizers.append(self.optimizer_G)
        #    self.optimizers.append(self.optimizer_D)
        #    #
        #    self.lambda_L1 = 100.0   # 论文通常给 L1 很大权重
        #    self.lambda_grad = 10.0  # 梯度损失权重

        if self.isTrain:
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)

            # 损失器定义
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # 优化器：只优化 G_A 和 D_A
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr * opt.lr_D_scale, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.lambda_gan = opt.lambda_gan
            self.lambda_L1 = 100.0  # 论文通常给 L1 很大权重
            self.lambda_grad = 10.0  # 梯度损失权重
            self.lambda_mean = 1.0  # 灰度平均损失

            self.lambda_L1 = opt.lambda_L1_supervised
            self.lambda_grad = opt.lambda_grad
            self.lambda_mean = opt.lambda_mean
            self.gan_start_epoch = opt.gan_start_epoch
            self.d_update_ratio = max(1, opt.d_update_ratio)

    def set_epoch(self, epoch):
        """供 train.py 调用，每轮更新当前 epoch"""
        self.current_epoch = epoch

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        #self.real_B = input["B" if AtoB else "A"].to(self.device)
        # 增加容错判断
        if 'B' in input or 'A' in input:  # 确保目标键存在
            target_key = 'B' if AtoB else 'A'
            if target_key in input:
                self.real_B = input[target_key].to(self.device)
            else:
                # 如果是测试阶段且没有 B，可以设为 None 或跳过
                self.real_B = None
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        #self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
        #self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        #self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        #"""Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

        # """计算判别器 D_A 的损失，判断 fake_B 是否真实"""
        # # Real
        # pred_real = self.netD_A(self.real_B)
        # loss_D_real = self.criterionGAN(pred_real, True)
        # # Fake
        # pred_fake = self.netD_A(self.fake_B.detach())
        # loss_D_fake = self.criterionGAN(pred_fake, False)
        # # Combined loss
        # self.loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        # self.loss_D_A.backward()

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        # lambda_idt = self.opt.lambda_identity
        # lambda_A = self.opt.lambda_A
        # lambda_B = self.opt.lambda_B
        # # Identity loss
        # if lambda_idt > 0:
        #     # G_A should be identity if real_B is fed: ||G_A(B) - B||
        #     self.idt_A = self.netG_A(self.real_B)
        #     self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
        #     # G_B should be identity if real_A is fed: ||G_B(A) - A||
        #     self.idt_B = self.netG_B(self.real_A)
        #     self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        # else:
        #     self.loss_idt_A = 0
        #     self.loss_idt_B = 0
#
        # # GAN ss D_A(G_A(A))
        # self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # # GAN loss D_B(G_B(B))
        # self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # # Forward cycle loss || G_B(G_A(A)) - A||
        # self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # # Backward cycle loss || G_A(G_B(B)) - B||
        # self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # # combined loss and calculate gradients
        # self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        # self.loss_G.backward()

        # 论文:将原本纯粹的“风格转换”损失，改为“物理分解”损失
        #lambda_idt = self.opt.lambda_identity
        #lambda_A = self.opt.lambda_A
        #lambda_B = self.opt.lambda_B
#
        ## 1. GAN Loss (保持原有)
        #self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        #self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
#
        ## 2. Cycle Loss (保持原有)
        #self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        #self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
#
        ## 2. 【核心修改】引入论文的有监督 L1 损失 (Supervised Loss)
        ## 假设 A 域是 kV 投影，B 域是对应的 DRR 脊柱图
        ## 我们强制让生成的 fake_B 必须在像素级靠近真实的 real_B (DRR)
        #self.loss_G_L1_Supervised = self.criterionL1(self.fake_B, self.real_B) * 100.0
#
        ## 3. 【核心修改】引入梯度一致性损失 (Gradient Consistency Loss)
        ## 论文通过这个损失确保提取出的脊柱具有锐利的边缘，而不是模糊的一团
        #fake_B_gx, fake_B_gy = get_image_gradients(self.fake_B)
        #real_B_gx, real_B_gy = get_image_gradients(self.real_B)
        #self.loss_G_gradient = (self.criterionL1(fake_B_gx, real_B_gx) +
        #                        self.criterionL1(fake_B_gy, real_B_gy)) * 10.0
#
        ## 4. 合并所有损失
        ## 建议在调试单图验证时，给 L1_Supervised 最大的权重
        #self.loss_G = (self.loss_G_A + self.loss_G_B +
        #               self.loss_cycle_A + self.loss_cycle_B +
        #               self.loss_G_L1_Supervised + self.loss_G_gradient)
#
        #self.loss_G.backward()

        """只进行 GAN + 有监督损失，不进行 Cycle 损失"""
        # 1. GAN loss: 让生成图 fake_B 骗过判别器 D_A
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)

        # 2. 有监督 L1 loss: 强制 fake_B 在像素上等于真实 DRR (real_B)
        self.loss_G_L1_Supervised = self.criterionL1(self.fake_B, self.real_B) * self.lambda_L1

        # 2. 自动判定是否进入“衰减阶段”
        # 总轮次 = 当前 epoch + 计数起点
        total_current_epoch = self.current_epoch + self.opt.epoch_count

        # 判定：如果总轮次超过了设定恒定学习率的轮次
        dy_lambda_grad = self.lambda_grad
        self.loss_G_mean = 0.0
        #if total_current_epoch > self.opt.n_epochs:
        #     # 进入衰减阶段，将梯度权重从倍增以保护弱边缘
        #     dy_lambda_grad = dy_lambda_grad * 1.5
        #     # 到了学习率下降阶段，再开启灰度校正
        #     # 灰度均值损失：校正整体亮度偏差, 计算 fake_B 和 real_B 的全局平均值
        #     mask = (self.real_B > -0.9).float()
        #     # fake_mean = torch.mean(self.fake_B)
        #     # real_mean = torch.mean(self.real_B)
        #     # 使用 MSE 计算均值差异，并给予一定的权重 (如 10.0)
        #     # 1. 提取真实图中非背景的区域 (假设背景归一化后接近 -1)
        #     # 2. 只计算掩码区域内的均值
        #     fake_mean = torch.sum(self.fake_B * mask) / (torch.sum(mask) + 1e-6)
        #     real_mean = torch.sum(self.real_B * mask) / (torch.sum(mask) + 1e-6)
        #     self.loss_G_mean = self.criterionL1(fake_mean, real_mean) * self.lambda_mean

        # 3. 梯度一致性损失: 保护骨骼边缘

        fake_B_gx, fake_B_gy = get_gradients(self.fake_B)
        real_B_gx, real_B_gy = get_gradients(self.real_B)
        self.loss_G_gradient = (self.criterionL1(fake_B_gx, real_B_gx) +
                                self.criterionL1(fake_B_gy, real_B_gy)) * dy_lambda_grad
        # 合并损失
        gan_weight = 0.0 if self.current_epoch < self.gan_start_epoch else self.lambda_gan
        self.loss_G = gan_weight * self.loss_G_A + self.loss_G_L1_Supervised + self.loss_G_gradient \
                      + self.loss_G_mean
        self.loss_G.backward()


    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        # self.forward()  # compute fake images and reconstruction images.
        # # G_A and G_B
        # self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        # self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        # self.backward_G()  # calculate gradients for G_A and G_B
        # self.optimizer_G.step()  # update G_A and G_B's weights
        # # D_A and D_B
        # self.set_requires_grad([self.netD_A, self.netD_B], True)
        # self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        # self.backward_D_A()  # calculate gradients for D_A
        # self.backward_D_B()  # calculate graidents for D_B
        # self.optimizer_D.step()  # update D_A and D_B's weights

        self.global_step += 1
        self.forward()  # compute fake images: self.fake_B
        # update G
        self.set_requires_grad(self.netD_A, False)  # D_A requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate gradients for G
        self.step_optimizer(self.optimizer_G)  # update G's weights
        # update D
        if self.global_step % self.d_update_ratio == 0:
            self.set_requires_grad(self.netD_A, True)
            self.optimizer_D.zero_grad()  # set D's gradients to zero
            self.backward_D_A()  # calculate gradients for D
            self.step_optimizer(self.optimizer_D)  # update D's weights
        else:
            self.loss_D_A = torch.tensor(0.0, device=self.device)
