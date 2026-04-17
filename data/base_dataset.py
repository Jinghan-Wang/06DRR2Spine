"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""

import math
import random
import numpy as np
import torch.utils.data as data  # PyTorch 的数据加载工具
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod  # 用于定义抽象类和抽象方法
import torch

class BaseDataset(data.Dataset, ABC):
    """
    该模块实现了一个抽象基类 BaseDataset，用于构建数据集的基础框架。
    同时提供了常用的图像变换函数（如 get_transform、__scale_width 等），这些函数可以在子类中复用。
    继承自 PyTorch 的 data.Dataset 和 Python 的 ABC（抽象基类）。
    目标是为所有数据集类提供统一接口。
    """

    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
            opt（Option类）——存储所有实验标志；需要是BaseOptions的子类
        """
        self.opt = opt
        self.root = opt.dataroot
        if self.opt.sample_ratio < 1:
            raise ValueError(f"sample_ratio must be >= 1, but got {self.opt.sample_ratio}")

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.
        允许子类添加特定于数据集的命令行参数, 默认返回原始解析器，子类可重写以扩展功能。
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        """ 这里返回 0 仅为占位，实际由子类覆盖 """
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        # 参数：index--一个用于数据索引的随机整数返回：一个包含其名称的数据字典。它通常包含数据本身及其元数据信息。
        """
        pass

    @staticmethod
    def _is_unlimited_size(max_dataset_size):
        try:
            return math.isinf(float(max_dataset_size))
        except (TypeError, ValueError):
            return False

    def _build_sample_indices(self, total_count):
        if total_count == 0 or self.opt.sample_ratio <= 1:
            return None

        if self.opt.sample_mode == "stride":
            return list(range(0, total_count, self.opt.sample_ratio))

        target_count = max(1, math.ceil(total_count / self.opt.sample_ratio))
        rng = random.Random(self.opt.sample_seed)
        return sorted(rng.sample(range(total_count), target_count))

    def _apply_sample_ratio_to_paths(self, paths, dataset_label):
        indices = self._build_sample_indices(len(paths))
        if indices is None:
            return paths

        sampled_paths = [paths[i] for i in indices]
        extra = f", seed={self.opt.sample_seed}" if self.opt.sample_mode == "random" else ""
        print(
            f"{dataset_label}: sample_ratio={self.opt.sample_ratio}, mode={self.opt.sample_mode}"
            f"{extra}, samples {len(paths)} -> {len(sampled_paths)}"
        )
        return sampled_paths

    def _apply_sample_ratio_to_pairs(self, a_paths, b_paths, dataset_label):
        indices = self._build_sample_indices(len(a_paths))
        if indices is None:
            return a_paths, b_paths

        sampled_a = [a_paths[i] for i in indices]
        sampled_b = [b_paths[i] for i in indices]
        extra = f", seed={self.opt.sample_seed}" if self.opt.sample_mode == "random" else ""
        print(
            f"{dataset_label}: sample_ratio={self.opt.sample_ratio}, mode={self.opt.sample_mode}"
            f"{extra}, paired samples {len(a_paths)} -> {len(sampled_a)}"
        )
        return sampled_a, sampled_b

    def _limit_paths(self, paths, max_dataset_size, dataset_label):
        if self._is_unlimited_size(max_dataset_size) or len(paths) <= max_dataset_size:
            return paths

        limited_paths = paths[:max_dataset_size]
        print(f"{dataset_label}: max_dataset_size enabled, samples {len(paths)} -> {len(limited_paths)}")
        return limited_paths

    def _limit_pairs(self, a_paths, b_paths, max_dataset_size, dataset_label):
        if self._is_unlimited_size(max_dataset_size) or len(a_paths) <= max_dataset_size:
            return a_paths, b_paths

        limited_a = a_paths[:max_dataset_size]
        limited_b = b_paths[:max_dataset_size]
        print(f"{dataset_label}: max_dataset_size enabled, paired samples {len(a_paths)} -> {len(limited_a)}")
        return limited_a, limited_b


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == "resize_and_crop":
        new_h = new_w = opt.load_size
    elif opt.preprocess == "scale_width_and_crop":  # 保持宽高比，将宽度缩放到 load_size，高度按比例缩放（但不低于 crop_size）。
        new_w = opt.load_size
        new_h = opt.load_size * h // w
    # 随机生成裁剪左上角坐标 (x, y)，确保裁剪区域在图像内
    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))
    # 随机决定是否水平翻转（50% 概率）
    flip = random.random() > 0.5

    return {"crop_pos": (x, y), "flip": flip}


def get_transform(opt, params=None, grayscale=False, method=transforms.InterpolationMode.BICUBIC, convert=True, is_medical=False):
    transform_list = []
    if is_medical:
        # 针对医学影像 NumPy 数据的变换序列
        # 将 NumPy (H, W, C) 转换为 Tensor (C, H, W)
        # 注意：此时 img 已经是归一化到 [0, 1] 的 float32 numpy 数组
        transform_list += [transforms.Lambda(lambda img: torch.from_numpy(img).float().permute(2, 0, 1))]

        # 最终统一到 [-1, 1]，这是 CycleGAN 生成器 Tanh 层的要求
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        return transforms.Compose(transform_list)

    if grayscale:  # 是否转为灰度图
        transform_list.append(transforms.Grayscale(1))
    if "resize" in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif "scale_width" in opt.preprocess:
        # transforms.Lambda(...) ,也就是把“任意函数”变成一个可组合的 transform。
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))

    if "crop" in opt.preprocess:
        if params is None: # params：若提供，则使用固定裁剪/翻转（用于测试或配对图像一致性）
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params["crop_pos"], opt.crop_size)))

    if opt.preprocess == "none":  # 若预处理为 "none"，则调整图像尺寸为 4 的倍数（满足某些网络输入要求）
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:  # params：若提供，则使用固定裁剪/翻转（用于测试或配对图像一致性）
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params["flip"]:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params["flip"])))

    if convert:
        transform_list += [transforms.ToTensor()]  # 变成 PyTorch Tensor，并且（对常见的 8-bit 图像）把像素值做缩放归一化。
        if grayscale: # 转为 Tensor 并归一化到 [-1, 1]（因为 (x - 0.5) / 0.5）
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __transforms2pil_resize(method):  # 将 transforms.InterpolationMode 转换为 PIL.Image.InterpolationMode
    mapper = {
        transforms.InterpolationMode.BILINEAR: Image.BILINEAR,
        transforms.InterpolationMode.BICUBIC: Image.BICUBIC,
        transforms.InterpolationMode.NEAREST: Image.NEAREST,
        transforms.InterpolationMode.LANCZOS: Image.LANCZOS,
    }
    return mapper[method]


def __make_power_2(img, base, method=transforms.InterpolationMode.BICUBIC):
    # 将图像尺寸调整为 base（默认 4）的整数倍,若已满足，则直接返回；否则缩放并打印警告（仅一次）
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=transforms.InterpolationMode.BICUBIC):
    """
    Scale the width of the image to target_size and the height proportionally,
    but not lower than cropsize (to avoid out of bounds during cropping).
    """
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    """
    Crop the size × size area starting from (x1, y1)
    If the original image is smaller than the cropped size, return it directly to the original image (to avoid errors)
    """
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if ow > tw or oh > th:
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, "has_printed"):
        print("The image size needs to be a multiple of 4. " "The loaded image size was (%d, %d), so it was adjusted to " "(%d, %d). This adjustment will be done to all images " "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
