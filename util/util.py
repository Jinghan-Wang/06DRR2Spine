"""Common utility helpers used across training and inference."""

from __future__ import print_function

import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist

from util.medical_image_io import is_nifti_path, save_nifti_image


def tensor2im(input_image, imtype=np.uint8):
    """Convert a tensor array into a uint8 image array."""
    if isinstance(input_image, np.ndarray):
        image_numpy = input_image
    elif isinstance(input_image, torch.Tensor):
        image_numpy = input_image.data[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        return input_image
    return image_numpy.astype(imtype)


def tensor2medical(input_image, imtype=np.float32):
    """Convert a tensor in [-1, 1] into a single-channel float image in [0, 1]."""
    if isinstance(input_image, torch.Tensor):
        image_numpy = input_image.detach().cpu().float().numpy()
        if image_numpy.ndim == 4:
            image_numpy = image_numpy[0]
        if image_numpy.ndim == 3:
            if image_numpy.shape[0] == 1:
                image_numpy = image_numpy[0]
            else:
                image_numpy = np.transpose(image_numpy, (1, 2, 0))
        image_numpy = np.clip((image_numpy + 1.0) / 2.0, 0.0, 1.0)
        if image_numpy.ndim == 2:
            image_numpy = image_numpy[:, :, np.newaxis]
        return image_numpy.astype(imtype)
    if isinstance(input_image, np.ndarray):
        return input_image.astype(imtype)
    return input_image


def diagnose_network(net, name="network"):
    """Calculate and print the mean absolute gradient magnitude."""
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def init_ddp():
    """Initialize distributed training when requested by the environment."""
    print(f"--- Environment Check ---")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")

    is_ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if is_ddp:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")

    print(f"Initialized with device {device}")
    return device


def cleanup_ddp():
    """Shut down the distributed process group when it is active."""
    if dist.is_initialized():
        dist.destroy_process_group()


def save_image(image_numpy, image_path, aspect_ratio=1.0, reference_path=None):
    """Save a numpy image to disk."""
    image_path_str = str(image_path)
    if is_nifti_path(image_path_str):
        save_nifti_image(image_numpy, image_path_str, reference_path=reference_path)
        return

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape
    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print summary statistics for a numpy array."""
    x = x.astype(np.float64)
    if shp:
        print("shape,", x.shape)
    if val:
        x = x.flatten()
        print(
            "mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f"
            % (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x))
        )


def mkdirs(paths):
    """Create empty directories if they do not exist."""
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """Create a single directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
