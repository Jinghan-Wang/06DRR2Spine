import glob
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from util.medical_image_io import collect_image_paths, is_supported_image_path, load_medical_image


class AlignedDataset(BaseDataset):
    """A dataset class for paired image datasets.

    It supports two paired-data layouts:
    1. The original pix2pix layout where ``dataroot/phase`` contains AB images concatenated side by side.
    2. A medical paired layout where A/B images live in mirrored subdirectories such as
       ``case_x/kv/*.nii.gz`` and ``case_x/drr_spine/*.nii.gz``.
    """

    def __init__(self, opt):
        """Initialize this dataset class."""
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.input_nc = self.opt.output_nc if self.opt.direction == "BtoA" else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == "BtoA" else self.opt.output_nc
        self.pix2pix_variant = opt.pix2pix_variant

        if self.pix2pix_variant == "medical_s1":
            self.AB_paths = []
            self.use_medical_pairs = True
            self.A_paths, self.B_paths = self._load_paired_paths(opt)
            self.A_paths, self.B_paths = self._apply_sample_ratio_to_pairs(self.A_paths, self.B_paths, "AlignedDataset")
            self.A_paths, self.B_paths = self._limit_pairs(self.A_paths, self.B_paths, opt.max_dataset_size, "AlignedDataset")
            self.pair_count = len(self.A_paths)
            self.transform_A = get_transform(self.opt, grayscale=(self.input_nc == 1), is_medical=True)
            self.transform_B = get_transform(self.opt, grayscale=(self.output_nc == 1), is_medical=True)
            print(f"AlignedDataset: using paired medical layout with {self.pair_count} pairs")
        else:
            self.use_medical_pairs = False
            self.AB_paths = self._load_legacy_ab_paths(opt)
            self.AB_paths = self._apply_sample_ratio_to_paths(self.AB_paths, "AlignedDataset(legacy)")
            self.AB_paths = self._limit_paths(self.AB_paths, opt.max_dataset_size, "AlignedDataset(legacy)")
            assert self.opt.load_size >= self.opt.crop_size
            print(f"AlignedDataset: using legacy AB layout with {len(self.AB_paths)} samples")

    def _load_legacy_ab_paths(self, opt):
        if not os.path.isdir(self.dir_AB):
            raise RuntimeError(f"Legacy pix2pix expects concatenated AB images under: {self.dir_AB}")
        legacy_paths = sorted(make_dataset(self.dir_AB, float("inf")))
        if not legacy_paths:
            raise RuntimeError(
                f"No legacy AB samples found under: {self.dir_AB}. "
                "If you want paired medical loading, set --pix2pix_variant medical_s1."
            )
        return legacy_paths

    def _load_paired_paths(self, opt):
        if opt.input_a_path or opt.input_b_path:
            if not opt.input_a_path or not opt.input_b_path:
                raise RuntimeError("Aligned medical loading requires both --input_a_path and --input_b_path.")
            a_paths = collect_image_paths(opt.input_a_path)
            b_paths = collect_image_paths(opt.input_b_path)
            return self._pair_paths_by_key(a_paths, b_paths, opt.dataset_a_subdir, opt.dataset_b_subdir)

        a_paths = self._scan_domain_paths(opt.dataset_a_subdir)
        b_paths = self._scan_domain_paths(opt.dataset_b_subdir)
        paired_a, paired_b = self._pair_paths_by_key(a_paths, b_paths, opt.dataset_a_subdir, opt.dataset_b_subdir)
        return paired_a, paired_b

    def _scan_domain_paths(self, subdir_name):
        search_pattern = os.path.join(self.root, "**", subdir_name, "**", "*.*")
        all_paths = glob.glob(search_pattern, recursive=True)
        filtered_paths = [
            path for path in all_paths if os.path.isfile(path) and is_supported_image_path(path)
        ]
        return sorted(filtered_paths)

    def _pair_paths_by_key(self, a_paths, b_paths, a_subdir, b_subdir):
        a_map = self._build_pair_map(a_paths, a_subdir)
        b_map = self._build_pair_map(b_paths, b_subdir)

        common_keys = sorted(set(a_map) & set(b_map))
        missing_a = sorted(set(a_map) - set(b_map))
        missing_b = sorted(set(b_map) - set(a_map))

        if missing_a:
            print(f"Warning: {len(missing_a)} A-side files have no matching B-side pair and will be skipped.")
        if missing_b:
            print(f"Warning: {len(missing_b)} B-side files have no matching A-side pair and will be skipped.")
        if not common_keys:
            raise RuntimeError(
                f"No paired samples found under '{a_subdir}' and '{b_subdir}'. "
                "Please verify dataroot, dataset_a_subdir, and dataset_b_subdir."
            )

        paired_a = [a_map[key] for key in common_keys]
        paired_b = [b_map[key] for key in common_keys]
        return paired_a, paired_b

    def _build_pair_map(self, paths, subdir_name):
        pair_map = {}
        for path in paths:
            key = self._pair_key(path, subdir_name)
            if key in pair_map:
                raise RuntimeError(f"Duplicate paired key '{key}' found for subdir '{subdir_name}'.")
            pair_map[key] = path
        return pair_map

    def _pair_key(self, path, subdir_name):
        rel_path = Path(path).resolve().relative_to(Path(self.root).resolve())
        parts = list(rel_path.parts)
        parts_lower = [part.lower() for part in parts]
        subdir_lower = subdir_name.lower()

        if subdir_lower in parts_lower:
            idx = parts_lower.index(subdir_lower)
            key_parts = parts[:idx] + parts[idx + 1 :]
        else:
            key_parts = parts

        return Path(*key_parts).as_posix().lower()

    def _read_medical_image(self, image_path, channels):
        image = load_medical_image(image_path)
        if channels == 1:
            return image
        return np.repeat(image[:, :, :1], channels, axis=2)

    def _should_apply_medical_augmentation(self):
        return self.opt.isTrain and self.opt.medical_aug_enable

    @staticmethod
    def _tensor_to_numpy_image(image_tensor):
        return image_tensor.detach().cpu().permute(1, 2, 0).clamp(0.0, 1.0).numpy().astype(np.float32)

    def _sample_medical_affine_params(self, height, width):
        angle = random.uniform(-self.opt.medical_aug_rotate_deg, self.opt.medical_aug_rotate_deg)
        translate_ratio = max(0.0, self.opt.medical_aug_translate)
        max_dx = int(round(width * translate_ratio))
        max_dy = int(round(height * translate_ratio))
        translate = (
            random.randint(-max_dx, max_dx) if max_dx > 0 else 0,
            random.randint(-max_dy, max_dy) if max_dy > 0 else 0,
        )
        scale_min = min(self.opt.medical_aug_scale_min, self.opt.medical_aug_scale_max)
        scale_max = max(self.opt.medical_aug_scale_min, self.opt.medical_aug_scale_max)
        scale = random.uniform(scale_min, scale_max) if scale_min != scale_max else scale_min
        hflip_prob = min(max(self.opt.medical_aug_hflip_prob, 0.0), 1.0)
        vflip_prob = min(max(self.opt.medical_aug_vflip_prob, 0.0), 1.0)
        do_hflip = random.random() < hflip_prob if hflip_prob > 0 else False
        do_vflip = random.random() < vflip_prob if vflip_prob > 0 else False
        return angle, translate, scale, do_hflip, do_vflip

    def _apply_affine_to_numpy_image(self, image, angle, translate, scale, do_hflip=False, do_vflip=False):
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        augmented = TF.affine(
            image_tensor,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
        if do_hflip:
            augmented = TF.hflip(augmented)
        if do_vflip:
            augmented = TF.vflip(augmented)
        return self._tensor_to_numpy_image(augmented)

    def _apply_domain_a_appearance_aug(self, image):
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        appearance_params = {}

        brightness = max(0.0, self.opt.medical_aug_a_brightness)
        if brightness > 0:
            factor = random.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness)
            image_tensor = TF.adjust_brightness(image_tensor, factor)
            appearance_params["brightness"] = factor

        contrast = max(0.0, self.opt.medical_aug_a_contrast)
        if contrast > 0:
            factor = random.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast)
            image_tensor = TF.adjust_contrast(image_tensor, factor)
            appearance_params["contrast"] = factor

        noise_std = max(0.0, self.opt.medical_aug_a_noise_std)
        if noise_std > 0:
            image_tensor = image_tensor + torch.randn_like(image_tensor) * noise_std
            appearance_params["noise_std"] = noise_std

        image_tensor = image_tensor.clamp(0.0, 1.0)
        return self._tensor_to_numpy_image(image_tensor), appearance_params

    @staticmethod
    def _format_medical_aug_message(angle, translate, scale, do_hflip, do_vflip, appearance_params):
        parts = [
            f"aug(rot={angle:.2f}",
            f"tx={translate[0]}",
            f"ty={translate[1]}",
            f"scale={scale:.3f}",
        ]
        if do_hflip:
            parts.append("hflip=1")
        if do_vflip:
            parts.append("vflip=1")
        if "brightness" in appearance_params:
            parts.append(f"bright={appearance_params['brightness']:.3f}")
        if "contrast" in appearance_params:
            parts.append(f"contrast={appearance_params['contrast']:.3f}")
        if "noise_std" in appearance_params:
            parts.append(f"noise={appearance_params['noise_std']:.3f}")
        return ", ".join(parts) + ")"

    def _augment_medical_pair(self, image_a, image_b):
        if not self._should_apply_medical_augmentation():
            return image_a, image_b, ""

        height, width = image_a.shape[:2]
        angle, translate, scale, do_hflip, do_vflip = self._sample_medical_affine_params(height, width)
        image_a = self._apply_affine_to_numpy_image(image_a, angle, translate, scale, do_hflip, do_vflip)
        image_b = self._apply_affine_to_numpy_image(image_b, angle, translate, scale, do_hflip, do_vflip)
        image_a, appearance_params = self._apply_domain_a_appearance_aug(image_a)
        aug_message = self._format_medical_aug_message(
            angle, translate, scale, do_hflip, do_vflip, appearance_params
        )
        return image_a, image_b, aug_message

    def __getitem__(self, index):
        """Return a data point and its metadata information."""
        if self.use_medical_pairs:
            A_path = self.A_paths[index]
            B_path = self.B_paths[index]
            A_img = self._read_medical_image(A_path, self.input_nc)
            B_img = self._read_medical_image(B_path, self.output_nc)
            A_img, B_img, aug_message = self._augment_medical_pair(A_img, B_img)
            A = self.transform_A(A_img)
            B = self.transform_B(B_img)
            return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path, "aug_params": aug_message}

        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert("RGB")
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)
        return {"A": A, "B": B, "A_paths": AB_path, "B_paths": AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        if self.use_medical_pairs:
            return self.pair_count
        return len(self.AB_paths)
