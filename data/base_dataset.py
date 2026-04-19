"""Dataset base classes and shared transform helpers."""

import math
import random
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class BaseDataset(data.Dataset, ABC):
    """Abstract dataset base class."""

    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        if self.opt.sample_ratio < 1:
            raise ValueError(f"sample_ratio must be >= 1, but got {self.opt.sample_ratio}")

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of samples in the dataset."""

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata."""

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
    elif opt.preprocess == "scale_width_and_crop":
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))
    flip = random.random() > 0.5
    return {"crop_pos": (x, y), "flip": flip}


def get_transform(
    opt,
    params=None,
    grayscale=False,
    method=transforms.InterpolationMode.BICUBIC,
    convert=True,
    is_medical=False,
):
    transform_list = []
    if is_medical:
        transform_list.append(
            transforms.Lambda(lambda img: torch.from_numpy(img).float().permute(2, 0, 1))
        )
        transform_list.append(transforms.Lambda(lambda tensor: (tensor - 0.5) / 0.5))
        return transforms.Compose(transform_list)

    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if "resize" in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif "scale_width" in opt.preprocess:
        transform_list.append(
            transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method))
        )

    if "crop" in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(
                transforms.Lambda(lambda img: __crop(img, params["crop_pos"], opt.crop_size))
            )

    if opt.preprocess == "none":
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params["flip"]:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params["flip"])))

    if convert:
        transform_list.append(transforms.ToTensor())
        if grayscale:
            transform_list.append(transforms.Normalize((0.5,), (0.5,)))
        else:
            transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return transforms.Compose(transform_list)


def __transforms2pil_resize(method):
    mapper = {
        transforms.InterpolationMode.BILINEAR: Image.BILINEAR,
        transforms.InterpolationMode.BICUBIC: Image.BICUBIC,
        transforms.InterpolationMode.NEAREST: Image.NEAREST,
        transforms.InterpolationMode.LANCZOS: Image.LANCZOS,
    }
    return mapper[method]


def __make_power_2(img, base, method=transforms.InterpolationMode.BICUBIC):
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=transforms.InterpolationMode.BICUBIC):
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
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
    """Print the resize warning once."""
    if not hasattr(__print_size_warning, "has_printed"):
        print(
            "The image size needs to be a multiple of 4. "
            "The loaded image size was (%d, %d), so it was adjusted to (%d, %d). "
            "This adjustment will be done to all images whose sizes are not multiples of 4"
            % (ow, oh, w, h)
        )
        __print_size_warning.has_printed = True
