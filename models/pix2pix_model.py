from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from .base_model import BaseModel
from . import networks


def get_gradients(image):
    """Compute first-order gradients along height and width."""
    dy = image[:, :, 1:, :] - image[:, :, :-1, :]
    dx = image[:, :, :, 1:] - image[:, :, :, :-1]
    return dy, dx


def ssim_loss(image1, image2, window_size=11, data_range=2.0):
    """A lightweight SSIM loss for paired image translation."""
    padding = window_size // 2
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu1 = F.avg_pool2d(image1, window_size, stride=1, padding=padding)
    mu2 = F.avg_pool2d(image2, window_size, stride=1, padding=padding)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(image1 * image1, window_size, stride=1, padding=padding) - mu1_sq
    sigma2_sq = F.avg_pool2d(image2 * image2, window_size, stride=1, padding=padding) - mu2_sq
    sigma12 = F.avg_pool2d(image1 * image2, window_size, stride=1, padding=padding) - mu1_mu2

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return 1.0 - ssim_map.mean()


def to_unit_range(image):
    """Map normalized tensors from [-1, 1] to [0, 1]."""
    return (image + 1.0) * 0.5


def masked_l1_loss(prediction, target, mask):
    """Compute the mean L1 error over the active region of a mask."""
    if mask.shape[1] == 1 and prediction.shape[1] != 1:
        mask = mask.expand(-1, prediction.shape[1], -1, -1)
    masked_error = torch.abs(prediction - target) * mask
    return masked_error.sum() / mask.sum().clamp_min(1.0)


def compose_masked_prediction(image_prediction, mask_prob):
    """Combine an image prediction and a soft foreground mask in normalized [-1, 1] space."""
    return mask_prob * (image_prediction + 1.0) - 1.0


def dice_loss_from_probabilities(prediction_prob, target_mask, eps=1e-6):
    """Compute Dice loss directly from predicted probabilities and a binary target mask."""
    reduce_dims = (1, 2, 3)
    intersection = (prediction_prob * target_mask).sum(dim=reduce_dims)
    union = prediction_prob.sum(dim=reduce_dims) + target_mask.sum(dim=reduce_dims)
    dice_score = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice_score.mean()


def soft_mask_dice_loss(prediction, target, fg_threshold, temperature=None, eps=1e-6):
    """Compute a soft Dice loss on foreground masks derived from the target image."""
    pred_gray = to_unit_range(prediction).mean(dim=1, keepdim=True)
    target_gray = to_unit_range(target).mean(dim=1, keepdim=True)
    target_fg = (target_gray > fg_threshold).float()
    if temperature is None or temperature <= 0:
        temperature = max(float(fg_threshold) * 0.25, 1e-3)
    pred_fg_soft = torch.sigmoid((pred_gray - fg_threshold) / temperature)
    reduce_dims = (1, 2, 3)
    intersection = (pred_fg_soft * target_fg).sum(dim=reduce_dims)
    union = pred_fg_soft.sum(dim=reduce_dims) + target_fg.sum(dim=reduce_dims)
    dice_score = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice_score.mean()


def soft_mask_bce_loss(prediction, target, fg_threshold, temperature=None, pos_weight=1.0):
    """Compute a BCE loss on foreground occupancy derived from the target image."""
    pred_gray = to_unit_range(prediction).mean(dim=1, keepdim=True)
    target_gray = to_unit_range(target).mean(dim=1, keepdim=True)
    target_fg = (target_gray > fg_threshold).float()
    if temperature is None or temperature <= 0:
        temperature = max(float(fg_threshold) * 0.25, 1e-3)
    pred_logits = (pred_gray - fg_threshold) / temperature
    pos_weight_tensor = pred_logits.new_tensor([max(float(pos_weight), 1.0)])
    return F.binary_cross_entropy_with_logits(pred_logits, target_fg, pos_weight=pos_weight_tensor)


def save_region_mask_debug_images(target_gray, fg_mask, bg_mask, edge_mask, debug_dir, debug_prefix):
    """Save region-mask debugging images for quick visual inspection."""
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    tensor_map = {
        "target_gray": target_gray,
        "fg_mask": fg_mask,
        "bg_mask": bg_mask,
        "edge_mask": edge_mask,
    }
    for name, tensor in tensor_map.items():
        image = tensor[0, 0].detach().cpu().clamp(0.0, 1.0).mul(255).to(torch.uint8).numpy()
        Image.fromarray(image, mode="L").save(debug_dir / f"{debug_prefix}_{name}.png")


def build_region_masks(target_image, fg_threshold=0.05, edge_kernel_size=9, debug_dir=None, debug_prefix=None):
    """Build foreground/background/edge masks directly from the ground-truth target."""
    target_gray = to_unit_range(target_image).mean(dim=1, keepdim=True)
    fg_mask = (target_gray > fg_threshold).float()
    bg_mask = 1.0 - fg_mask

    kernel_size = max(1, int(edge_kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    if kernel_size == 1:
        edge_mask = fg_mask.clone()
    else:
        padding = kernel_size // 2
        dilated = F.max_pool2d(fg_mask, kernel_size=kernel_size, stride=1, padding=padding)
        eroded = 1.0 - F.max_pool2d(1.0 - fg_mask, kernel_size=kernel_size, stride=1, padding=padding)
        edge_mask = (dilated - eroded).clamp(0.0, 1.0)

    if debug_dir and debug_prefix:
        save_region_mask_debug_images(target_gray, fg_mask, bg_mask, edge_mask, debug_dir, debug_prefix)

    return fg_mask, bg_mask, edge_mask


class Pix2PixModel(BaseModel):
    """This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm="batch", netG="unet_256", dataset_mode="aligned")
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode="vanilla")
            parser.add_argument("--lambda_L1", type=float, default=100.0, help="weight for L1 loss")
            parser.add_argument("--lambda_L1_supervised", type=float, default=None, help="optional alias of lambda_L1 for supervised reconstruction")
            parser.add_argument("--lambda_gan", type=float, default=1.0, help="weight for adversarial loss")
            parser.add_argument("--lambda_ssim", type=float, default=None, help="weight for SSIM loss")
            parser.add_argument("--lambda_grad", type=float, default=None, help="weight for gradient consistency loss")
            parser.add_argument("--lambda_fg", type=float, default=0.0, help="weight for foreground-only reconstruction loss in medical_s1")
            parser.add_argument("--lambda_bg", type=float, default=0.0, help="weight for background-only reconstruction loss in medical_s1")
            parser.add_argument("--lambda_edge", type=float, default=0.0, help="weight for edge-only reconstruction loss in medical_s1")
            parser.add_argument("--lambda_mask_dice", type=float, default=0.0, help="weight for soft Dice overlap loss on medical_s1 foreground masks")
            parser.add_argument("--lambda_mask_bce", type=float, default=0.0, help="weight for BCE occupancy loss on medical_s1 foreground masks")
            parser.add_argument("--mask_temperature", type=float, default=None, help="soft threshold temperature shared by medical_s1 mask Dice/BCE losses; <=0 uses the legacy auto rule")
            parser.add_argument("--mask_pos_weight", type=float, default=1.0, help="positive-class weight for medical_s1 mask BCE loss")
            parser.add_argument("--fg_threshold", type=float, default=0.05, help="foreground threshold applied to target images in medical_s1")
            parser.add_argument("--edge_kernel_size", type=int, default=9, help="odd kernel size used to derive edge bands from the foreground mask in medical_s1")
            parser.add_argument("--save_region_mask_debug", action="store_true", help="save target_gray/fg/bg/edge mask debug images during medical_s1 training")
            parser.add_argument("--region_mask_debug_dir", type=str, default="", help="directory used to save medical_s1 region-mask debug images; defaults to checkpoints/<name>/mask_debug when enabled")
            parser.add_argument("--region_mask_debug_limit", type=int, default=20, help="maximum number of medical_s1 batches to save for region-mask debugging")
            parser.add_argument("--gan_start_epoch", type=int, default=0, help="start adversarial loss after this epoch")
            parser.add_argument("--d_update_ratio", type=int, default=1, help="update D once every N generator steps")
            parser.add_argument("--lr_D_scale", type=float, default=1.0, help="scale factor applied to discriminator learning rate")

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.pix2pix_variant = opt.pix2pix_variant
        self.is_medical_s1_variant = self.pix2pix_variant == "medical_s1"
        self.use_medical_s1 = self.isTrain and self.is_medical_s1_variant
        self.use_dual_head_generator = self.is_medical_s1_variant and opt.netG in {"unet_128_dualhead", "unet_256_dualhead"}
        self.use_adversarial_training = self.isTrain and (not self.use_medical_s1 or opt.lambda_gan > 0)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        if self.use_medical_s1:
            recon_loss_names = ["G_L1", "G_fg", "G_bg", "G_edge", "G_mask_dice", "G_mask_bce", "G_SSIM", "G_gradient"]
            if self.use_adversarial_training:
                self.loss_names = ["G_GAN"] + recon_loss_names + ["D_real", "D_fake"]
            else:
                self.loss_names = recon_loss_names
        else:
            self.loss_names = ["G_GAN", "G_L1", "D_real", "D_fake"]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        if self.use_dual_head_generator:
            self.visual_names = ["real_A", "fake_B", "real_B", "fake_B_img", "fake_mask_vis"]
        else:
            self.visual_names = ["real_A", "fake_B", "real_B"]
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            if self.use_medical_s1 and not self.use_adversarial_training:
                self.model_names = ["G"]
            else:
                self.model_names = ["G", "D"]
        else:  # during test time, only load G
            self.model_names = ["G"]
        self.current_epoch = opt.epoch_count - 1 if self.use_medical_s1 else 0
        self.global_step = 0
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)

        if self.isTrain and (not self.use_medical_s1 or self.use_adversarial_training):  # define a discriminator only when adversarial training is enabled
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)

        if self.isTrain:
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if self.use_medical_s1:
                self.lambda_gan = opt.lambda_gan
                self.lambda_L1 = opt.lambda_L1_supervised if opt.lambda_L1_supervised is not None else opt.lambda_L1
                self.lambda_ssim = 0.2 * self.lambda_L1 if opt.lambda_ssim is None else opt.lambda_ssim
                self.lambda_grad = 0.1 * self.lambda_L1 if opt.lambda_grad is None else opt.lambda_grad
                self.lambda_fg = opt.lambda_fg
                self.lambda_bg = opt.lambda_bg
                self.lambda_edge = opt.lambda_edge
                self.lambda_mask_dice = opt.lambda_mask_dice
                self.lambda_mask_bce = opt.lambda_mask_bce
                self.mask_temperature = opt.mask_temperature
                self.mask_pos_weight = opt.mask_pos_weight
                self.fg_threshold = opt.fg_threshold
                self.edge_kernel_size = opt.edge_kernel_size
                self.mask_debug_enabled = opt.save_region_mask_debug
                if self.mask_debug_enabled:
                    self.mask_debug_dir = opt.region_mask_debug_dir or str(self.save_dir / "mask_debug")
                    self.mask_debug_limit = max(0, opt.region_mask_debug_limit)
                else:
                    self.mask_debug_dir = ""
                    self.mask_debug_limit = 0
                self.mask_debug_counter = 0
                self.gan_start_epoch = opt.gan_start_epoch
                self.d_update_ratio = max(1, opt.d_update_ratio)
                if self.use_adversarial_training:
                    self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # move to the device for custom loss
                    discriminator_lr = opt.lr * opt.lr_D_scale
                    self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=discriminator_lr, betas=(opt.beta1, 0.999))
                    self.optimizers.append(self.optimizer_D)
                    if opt.continue_train:
                        self.optional_model_names_for_loading.add("D")
            else:
                self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # move to the device for custom loss
                self.lambda_L1 = opt.lambda_L1
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)

    def _region_mask_debug_prefix(self):
        """Build a readable prefix for saved region-mask debugging images."""
        image_path = self.image_paths
        if isinstance(image_path, (list, tuple)):
            image_path = image_path[0]
        stem = Path(str(image_path)).stem
        return f"epoch{self.current_epoch:03d}_step{self.global_step:06d}_{stem}"

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        generator_output = self.forward_generator(self.netG, self.real_A)
        if self.use_dual_head_generator:
            self.fake_B_img, self.fake_mask_logits = generator_output
            self.fake_mask_prob = torch.sigmoid(self.fake_mask_logits)
            self.fake_mask_vis = self.fake_mask_prob * 2.0 - 1.0
            self.fake_B = compose_masked_prediction(self.fake_B_img, self.fake_mask_prob)
        else:
            self.fake_B = generator_output  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def _compute_medical_reconstruction_losses(self):
        """Compute the medical_s1 reconstruction losses, including region-focused terms."""
        debug_dir = None
        debug_prefix = None
        if self.mask_debug_enabled and self.mask_debug_counter < self.mask_debug_limit:
            debug_dir = self.mask_debug_dir
            debug_prefix = self._region_mask_debug_prefix()
            self.mask_debug_counter += 1

        fg_mask, bg_mask, edge_mask = build_region_masks(
            self.real_B,
            fg_threshold=self.fg_threshold,
            edge_kernel_size=self.edge_kernel_size,
            debug_dir=debug_dir,
            debug_prefix=debug_prefix,
        )

        if self.use_dual_head_generator:
            self.fake_B_teacher = compose_masked_prediction(self.fake_B_img, fg_mask)
            reconstruction_prediction = self.fake_B_teacher
        else:
            self.fake_B_teacher = self.fake_B
            reconstruction_prediction = self.fake_B

        self.loss_G_L1 = self.criterionL1(reconstruction_prediction, self.real_B) * self.lambda_L1

        if self.lambda_fg > 0:
            self.loss_G_fg = masked_l1_loss(reconstruction_prediction, self.real_B, fg_mask) * self.lambda_fg
        else:
            self.loss_G_fg = torch.tensor(0.0, device=self.device)

        if self.lambda_bg > 0:
            self.loss_G_bg = masked_l1_loss(self.fake_B, self.real_B, bg_mask) * self.lambda_bg
        else:
            self.loss_G_bg = torch.tensor(0.0, device=self.device)

        if self.lambda_edge > 0:
            self.loss_G_edge = masked_l1_loss(reconstruction_prediction, self.real_B, edge_mask) * self.lambda_edge
        else:
            self.loss_G_edge = torch.tensor(0.0, device=self.device)

        if self.lambda_mask_dice > 0:
            if self.use_dual_head_generator:
                self.loss_G_mask_dice = dice_loss_from_probabilities(self.fake_mask_prob, fg_mask) * self.lambda_mask_dice
            else:
                self.loss_G_mask_dice = soft_mask_dice_loss(
                    self.fake_B,
                    self.real_B,
                    fg_threshold=self.fg_threshold,
                    temperature=self.mask_temperature,
                ) * self.lambda_mask_dice
        else:
            self.loss_G_mask_dice = torch.tensor(0.0, device=self.device)

        if self.lambda_mask_bce > 0:
            if self.use_dual_head_generator:
                pos_weight_tensor = self.fake_mask_logits.new_tensor([max(float(self.mask_pos_weight), 1.0)])
                self.loss_G_mask_bce = F.binary_cross_entropy_with_logits(
                    self.fake_mask_logits,
                    fg_mask,
                    pos_weight=pos_weight_tensor,
                ) * self.lambda_mask_bce
            else:
                self.loss_G_mask_bce = soft_mask_bce_loss(
                    self.fake_B,
                    self.real_B,
                    fg_threshold=self.fg_threshold,
                    temperature=self.mask_temperature,
                    pos_weight=self.mask_pos_weight,
                ) * self.lambda_mask_bce
        else:
            self.loss_G_mask_bce = torch.tensor(0.0, device=self.device)

        if self.lambda_ssim > 0:
            self.loss_G_SSIM = ssim_loss(reconstruction_prediction, self.real_B) * self.lambda_ssim
        else:
            self.loss_G_SSIM = torch.tensor(0.0, device=self.device)

        if self.lambda_grad > 0:
            fake_B_dy, fake_B_dx = get_gradients(reconstruction_prediction)
            real_B_dy, real_B_dx = get_gradients(self.real_B)
            self.loss_G_gradient = (
                self.criterionL1(fake_B_dy, real_B_dy) + self.criterionL1(fake_B_dx, real_B_dx)
            ) * self.lambda_grad
        else:
            self.loss_G_gradient = torch.tensor(0.0, device=self.device)

        self.loss_G = (
            self.loss_G_L1
            + self.loss_G_fg
            + self.loss_G_bg
            + self.loss_G_edge
            + self.loss_G_mask_dice
            + self.loss_G_mask_bce
            + self.loss_G_SSIM
            + self.loss_G_gradient
        )

    def _gan_objective_active(self):
        """Whether GAN losses should participate at the current epoch."""
        if not self.use_adversarial_training:
            return False
        if not self.use_medical_s1:
            return True
        return self.current_epoch >= self.gan_start_epoch

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        if self.use_medical_s1:
            self._compute_medical_reconstruction_losses()
            if self.use_adversarial_training:
                if self._gan_objective_active():
                    fake_AB = torch.cat((self.real_A, self.fake_B), 1)
                    pred_fake = self.netD(fake_AB)
                    self.loss_G_GAN = self.criterionGAN(pred_fake, True)
                    self.loss_G = self.loss_G + self.lambda_gan * self.loss_G_GAN
                else:
                    self.loss_G_GAN = torch.tensor(0.0, device=self.device)
        else:
            # First, G(A) should fake the discriminator
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            # Second, G(A) = B
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.lambda_L1
            self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        if self.use_medical_s1:
            self.global_step += 1
            self.forward()  # compute fake images: G(A)
            # update G
            if self.use_adversarial_training:
                self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()  # set G's gradients to zero
            self.backward_G()  # calculate gradients for G
            self.step_optimizer(self.optimizer_G)  # update G's weights
            # update D
            if self.use_adversarial_training and self._gan_objective_active() and self.global_step % self.d_update_ratio == 0:
                self.set_requires_grad(self.netD, True)  # enable backprop for D
                self.optimizer_D.zero_grad()  # set D's gradients to zero
                self.backward_D()  # calculate gradients for D
                self.step_optimizer(self.optimizer_D)  # update D's weights
            elif self.use_adversarial_training:
                self.loss_D_real = torch.tensor(0.0, device=self.device)
                self.loss_D_fake = torch.tensor(0.0, device=self.device)
            return

        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.step_optimizer(self.optimizer_D)  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.step_optimizer(self.optimizer_G)  # update G's weights
