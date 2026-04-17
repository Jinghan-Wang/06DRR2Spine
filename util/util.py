"""This module contains simple helper functions"""

from __future__ import print_function
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import torch.distributed as dist
import os
from data.prj_parser import PrjImage
from util.medical_image_io import is_nifti_path, save_nifti_image

def tensor2im(input_image, imtype=np.uint8):
    """ "Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def tensor2prj(input_image, imtype=PrjImage.dtype):
    """ 将 Tensor 转换回医学影像 NumPy 数组 (保持单通道精度)

    Parameters:
        input_image (tensor) -- 输入的图像 Tensor (通常范围在 -1 到 1)
        imtype (type)        -- 目标 numpy 类型 (如 np.float32 或 np.uint16)
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            # 去掉 batch 维度并转到 CPU
            # input_image shape: [C, H, W]
            image_numpy = input_image.detach().cpu().float().numpy()
        else:
            return input_image

        # 1. 核心修改：保持单通道，不进行 RGB 复制
        # 如果是 [1, H, W]，转换后变成 [H, W]
        if image_numpy.shape[0] == 1:
            image_numpy = image_numpy[0]
        else:
            # 如果是多通道，转置为 [H, W, C]
            image_numpy = np.transpose(image_numpy, (1, 2, 0))

        # 2. 反归一化逻辑
        # CycleGAN 输出通常在 [-1, 1]。我们需要将其恢复。
        # 如果你的 PRJ 原始数据是 0 到 1 归一化的：
        image_numpy = (image_numpy + 1.0) / 2.0

        # 3. 这里的逻辑取决于你希望保存什么样的结果：
        # 如果要恢复到 0-255 方便查看：
        # image_numpy = image_numpy * 255.0

        # 如果要恢复到原始 PRJ 的物理数值（假设你记得原始的 min/max）:
        # image_numpy = image_numpy * (orig_max - orig_min) + orig_min

    else:
        image_numpy = input_image

    src_dtype = np.dtype(PrjImage.dtype)
    if np.issubdtype(src_dtype, np.integer):
        dtype_max = np.iinfo(src_dtype).max * 1.0
    else:
        dtype_max = 1.0

    # image_numpy_prj = (image_numpy.astype(PrjImage.dtype) / image_numpy_max * dtype_max).astype(PrjImage.dtype)

    image_numpy_prj = image_numpy * dtype_max
    return image_numpy_prj.astype(imtype)


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
        image_numpy = (image_numpy + 1.0) / 2.0
        image_numpy = np.clip(image_numpy, 0.0, 1.0)
        if image_numpy.ndim == 2:
            image_numpy = image_numpy[:, :, np.newaxis]
        return image_numpy.astype(imtype)
    if isinstance(input_image, np.ndarray):
        return input_image.astype(imtype)
    return input_image


def diagnose_network(net, name="network"):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
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


# initialize ddp
def init_ddp():
    print(f"--- Environment Check ---")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    # device = torch.device("cuda:0")
    # torch.cuda.set_device(0)
    # print(f"Initialized with device {device}")
    # return device

    # Initialize DDP if LOCAL_RANK is set
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


# cleanup ddp
def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def save_prj_image(image_numpy, image_path, raw_header_bytes=None):
    """
    拼接原始二进制头和新的像素数据，保存为 .prj
    """
    # dimensionality reduction: squeezing out the extra channel dimensions, turning (h, W, 1) into a purely two-dimensional (h, W)
    if len(image_numpy.shape) == 3 and image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]

    # 量程反归一化与类型转换
    # CycleGAN 传进来的 image_numpy 是 0-255 的 uint8 数据
    # 我们需要将其线性拉伸回 16-bit 的完整量程 (0-65535)，再转为 uint16
    # (如果是浮点数 Float32 协议，你需要将其映射回真实物理值区间)

    # src_dtype = np.dtype(PrjImage.dtype)
    # if np.issubdtype(src_dtype, np.integer):
    #     dtype_max = np.iinfo(src_dtype).max * 1.0
    # else:
    #     dtype_max = 1.0
#
    # image_numpy_max = np.iinfo(src_dtype).max
    # image_numpy_prj = (image_numpy.astype(PrjImage.dtype) / image_numpy_max * dtype_max).astype(PrjImage.dtype)

    with open(image_path, 'wb') as f:
        if raw_header_bytes is not None:
            f.write(raw_header_bytes)
        f.write(image_numpy.tobytes())


def save_image(image_numpy, image_path, aspect_ratio=1.0, raw_header_bytes=None, reference_path=None):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    # 无论传进来的是 str 还是 PosixPath，统一强制转成普通字符串
    image_path_str = str(image_path)
    # ======= add：Intercept. PRJ format =======
    if image_path_str.endswith(('.prj', '.PRJ')):
        save_prj_image(image_numpy, image_path_str, raw_header_bytes)
        return  #go straight back after processing, without the PIL logic
    if is_nifti_path(image_path_str):
        save_nifti_image(image_numpy, image_path_str, reference_path=reference_path)
        return
    # =====================================
    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print("shape,", x.shape)
    if val:
        x = x.flatten()
        print("mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f" % (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)


