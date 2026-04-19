import torch
import torch.nn.functional as F

from . import networks
from .base_model import BaseModel


def get_gradients(image):
    """Compute finite-difference gradients along height and width."""
    dy = image[:, :, 1:, :] - image[:, :, :-1, :]
    dx = image[:, :, :, 1:] - image[:, :, :, :-1]
    return dy, dx


def get_laplacian_response(image):
    """Highlight local high-frequency detail using a fixed Laplacian filter."""
    channels = image.shape[1]
    kernel = image.new_tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return F.conv2d(image, kernel, padding=1, groups=channels)


class CycleGANModel(BaseModel):
    """A simplified one-direction generator-only A -> B translation model."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument("--lambda_A", type=float, default=10.0, help="legacy option kept for CLI compatibility")
            parser.add_argument("--lambda_B", type=float, default=10.0, help="legacy option kept for CLI compatibility")
            parser.add_argument("--lambda_gan", type=float, default=0.1, help="legacy option kept for CLI compatibility")
            parser.add_argument("--lambda_L1_supervised", type=float, default=50.0, help="weight for supervised L1 loss on fake_B")
            parser.add_argument("--lambda_grad", type=float, default=10.0, help="weight for gradient consistency loss")
            parser.add_argument("--lambda_laplacian", type=float, default=3.0, help="weight for Laplacian detail loss")
            parser.add_argument("--lambda_mean", type=float, default=0.0, help="weight for mean intensity loss")
            parser.add_argument("--gan_start_epoch", type=int, default=20, help="legacy option kept for CLI compatibility")
            parser.add_argument("--d_update_ratio", type=int, default=1, help="legacy option kept for CLI compatibility")
            parser.add_argument("--lr_D_scale", type=float, default=1.0, help="legacy option kept for CLI compatibility")
            parser.add_argument(
                "--lambda_identity",
                type=float,
                default=0.5,
                help="legacy option kept for CLI compatibility",
            )

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ["G_L1_Supervised", "G_gradient", "G_laplacian", "G_mean"]
        self.current_epoch = opt.epoch_count - 1 if self.isTrain else 0
        self.visual_names = ["real_A", "fake_B", "real_B"] if self.isTrain else ["real_A", "fake_B"]
        self.model_names = ["G_A"]

        self.netG_A = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
        )

        if self.isTrain:
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.lambda_L1 = opt.lambda_L1_supervised
            self.lambda_grad = opt.lambda_grad
            self.lambda_laplacian = opt.lambda_laplacian
            self.lambda_mean = opt.lambda_mean

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def set_input(self, input):
        a_to_b = self.opt.direction == "AtoB"
        source_key = "A" if a_to_b else "B"
        target_key = "B" if a_to_b else "A"
        path_key = "A_paths" if a_to_b else "B_paths"

        self.real_A = input[source_key].to(self.device)
        self.real_B = input[target_key].to(self.device) if target_key in input else None
        self.image_paths = input[path_key]

    def forward(self):
        self.fake_B = self.forward_generator(self.netG_A, self.real_A)

    def backward_G(self):
        self.loss_G_L1_Supervised = self.criterionL1(self.fake_B, self.real_B) * self.lambda_L1

        fake_B_gx, fake_B_gy = get_gradients(self.fake_B)
        real_B_gx, real_B_gy = get_gradients(self.real_B)
        self.loss_G_gradient = (
            self.criterionL1(fake_B_gx, real_B_gx) + self.criterionL1(fake_B_gy, real_B_gy)
        ) * self.lambda_grad

        fake_B_lap = get_laplacian_response(self.fake_B)
        real_B_lap = get_laplacian_response(self.real_B)
        self.loss_G_laplacian = self.criterionL1(fake_B_lap, real_B_lap) * self.lambda_laplacian

        self.loss_G_mean = torch.tensor(0.0, device=self.device)
        self.loss_G = (
            self.loss_G_L1_Supervised
            + self.loss_G_gradient
            + self.loss_G_laplacian
            + self.loss_G_mean
        )
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.step_optimizer(self.optimizer_G)
