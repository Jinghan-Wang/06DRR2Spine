import os
import torch
import torch.distributed as dist
from pathlib import Path
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.isTrain = opt.isTrain
        self.experiment_dir = Path(opt.checkpoints_dir) / opt.name
        self.save_dir = self.experiment_dir / "Models"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = opt.device
        # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
        if opt.preprocess != "scale_width":
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'
        self._optimizer_step_counts = {}
        self.optional_model_names_for_loading = set()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def set_epoch(self, epoch):
        """Training hook for models that need the current epoch."""
        self.current_epoch = epoch

    def forward_generator(self, net, input_tensor):
        """Run a generator, padding/cropping medical UNet inputs when needed."""
        multiple = self._generator_input_multiple()
        if multiple is None:
            return net(input_tensor)

        _, _, height, width = input_tensor.shape
        target_height = max(multiple, ((height + multiple - 1) // multiple) * multiple)
        target_width = max(multiple, ((width + multiple - 1) // multiple) * multiple)
        if target_height == height and target_width == width:
            return net(input_tensor)

        pad_bottom = target_height - height
        pad_right = target_width - width
        padded_input = torch.nn.functional.pad(input_tensor, (0, pad_right, 0, pad_bottom), mode="constant", value=-1.0)

        if not hasattr(self, "_has_printed_unet_padding_warning"):
            print(
                f"Auto-padding generator input from ({height}, {width}) to "
                f"({target_height}, {target_width}) for netG={self.opt.netG}"
            )
            self._has_printed_unet_padding_warning = True

        output = net(padded_input)
        return self._crop_generator_output(output, height, width)

    def _crop_generator_output(self, output, height, width):
        """Crop generator outputs back to the original spatial size."""
        if isinstance(output, tuple):
            return tuple(self._crop_generator_output(item, height, width) for item in output)
        if isinstance(output, list):
            return [self._crop_generator_output(item, height, width) for item in output]
        return output[:, :, :height, :width]

    def _generator_input_multiple(self):
        """Return required spatial multiple for medical UNet inference/training."""
        if self.opt.netG in {"unet_128", "unet_128_lite", "unet_128_dualhead", "resunet_128"}:
            return 128
        if self.opt.netG in {"unet_256", "unet_256_lite", "unet_256_dualhead", "resunet_256"}:
            return 256
        return None

    def step_optimizer(self, optimizer):
        """Step an optimizer and record that it has been used at least once."""
        optimizer.step()
        key = id(optimizer)
        self._optimizer_step_counts[key] = self._optimizer_step_counts.get(key, 0) + 1

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # Initialize all networks and load if needed
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                net = networks.init_net(net, opt.init_type, opt.init_gain)

                # Load networks if needed
                if not self.isTrain or opt.continue_train:
                    load_suffix = f"iter_{opt.load_iter}" if opt.load_iter > 0 else opt.epoch
                    load_filename = f"{load_suffix}_net_{name}.pth"
                    load_path = self.save_dir / load_filename

                    if isinstance(net, torch.nn.parallel.DistributedDataParallel):
                        net = net.module

                    if not load_path.exists():
                        if name in self.optional_model_names_for_loading:
                            print(f"Warning: optional checkpoint not found for net{name}: {load_path}. Using a fresh initialization instead.")
                        else:
                            raise FileNotFoundError(f"Checkpoint not found for net{name}: {load_path}")
                        net.to(self.device)
                        if dist.is_initialized():
                            if self.opt.norm == "syncbatch":
                                raise ValueError(f"For distributed training, opt.norm must be 'syncbatch' or 'inst', but got '{self.opt.norm}'. " "Please set --norm syncbatch for multi-GPU training.")
                            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[self.device.index])
                            dist.barrier()
                        setattr(self, "net" + name, net)
                        continue

                    print(f"loading the model from {load_path}")

                    state_dict = self._safe_load_state_dict(load_path)

                    if hasattr(state_dict, "_metadata"):
                        del state_dict._metadata

                    # patch InstanceNorm checkpoints
                    for key in list(state_dict.keys()):
                        self.__patch_instance_norm_state_dict(state_dict, net, key.split("."))
                    net.load_state_dict(state_dict)

                # Move network to device
                net.to(self.device)

                # Wrap networks with DDP after loading
                if dist.is_initialized():
                    # Check if using syncbatch normalization for DDP
                    if self.opt.norm == "syncbatch":
                        raise ValueError(f"For distributed training, opt.norm must be 'syncbatch' or 'inst', but got '{self.opt.norm}'. " "Please set --norm syncbatch for multi-GPU training.")

                    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[self.device.index])
                    # Sync all processes after DDP wrapping
                    dist.barrier()

                setattr(self, "net" + name, net)

        self.print_networks(opt.verbose)

        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]["lr"]
        for scheduler, optimizer in zip(self.schedulers, self.optimizers):
            if self._optimizer_step_counts.get(id(optimizer), 0) == 0:
                continue
            if self.opt.lr_policy == "plateau":
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]["lr"]
        print(f"learning rate {old_lr:.7f} -> {lr:.7f}")

    def _safe_load_state_dict(self, load_path):
        """Load a checkpoint state_dict with compatibility across PyTorch versions."""
        try:
            return torch.load(load_path, map_location=str(self.device), weights_only=True)
        except TypeError:
            return torch.load(load_path, map_location=str(self.device))

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, "loss_" + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk, unwrapping them first."""

        # Only allow the main process (rank 0) to save the checkpoint
        if not dist.is_initialized() or dist.get_rank() == 0:
            for name in self.model_names:
                if isinstance(name, str):
                    save_filename = f"{epoch}_net_{name}.pth"
                    save_path = self.save_dir / save_filename
                    net = getattr(self, "net" + name)

                    # 1. First, unwrap from DDP if it exists
                    if hasattr(net, "module"):
                        model_to_save = net.module
                    else:
                        model_to_save = net

                    # 2. Second, unwrap from torch.compile if it exists
                    if hasattr(model_to_save, "_orig_mod"):
                        model_to_save = model_to_save._orig_mod

                    # 3. Save the final, clean state_dict
                    torch.save(model_to_save.state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith("InstanceNorm") and (key == "running_mean" or key == "running_var"):
                if getattr(module, key) is None:
                    state_dict.pop(".".join(keys))
            if module.__class__.__name__.startswith("InstanceNorm") and (key == "num_batches_tracked"):
                state_dict.pop(".".join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all networks from the disk for DDP."""

        for name in self.model_names:
            if isinstance(name, str):
                load_filename = f"{epoch}_net_{name}.pth"
                load_path = self.save_dir / load_filename
                net = getattr(self, "net" + name)

                if isinstance(net, torch.nn.parallel.DistributedDataParallel):
                    net = net.module

                if not load_path.exists():
                    if name in self.optional_model_names_for_loading:
                        print(f"Warning: optional checkpoint not found for net{name}: {load_path}. Using a fresh initialization instead.")
                        continue
                    raise FileNotFoundError(f"Checkpoint not found for net{name}: {load_path}")
                print(f"loading the model from {load_path}")

                state_dict = self._safe_load_state_dict(load_path)

                if hasattr(state_dict, "_metadata"):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints
                for key in list(state_dict.keys()):
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split("."))
                net.load_state_dict(state_dict)

        # Add a barrier to sync all processes before continuing
        if dist.is_initialized():
            dist.barrier()

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print("---------- Networks initialized -------------")
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print(f"[Network {name}] Total number of parameters : {num_params / 1e6:.3f} M")
        print("-----------------------------------------------")

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def init_networks(self, init_type="normal", init_gain=0.02):
        """Initialize all networks: 1. move to device; 2. initialize weights

        Parameters:
            init_type (str) -- initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float) -- scaling factor for normal, xavier and orthogonal
        """
        import os

        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)

                # Move to device
                if torch.cuda.is_available():
                    if "LOCAL_RANK" in os.environ:
                        local_rank = int(os.environ["LOCAL_RANK"])
                        net.to(local_rank)
                        print(f"Initialized network {name} with device cuda:{local_rank}")
                    else:
                        net.to(0)
                        print(f"Initialized network {name} with device cuda:0")
                else:
                    net.to("cpu")
                    print(f"Initialized network {name} with device cpu")

                # Initialize weights using networks function
                networks.init_weights(net, init_type, init_gain)
