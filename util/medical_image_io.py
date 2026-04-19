import os
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import nibabel as nib
except ImportError:
    nib = None


SUPPORTED_IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".tif",
    ".tiff",
    ".nii",
    ".nii.gz",
)


def is_nifti_path(path):
    return str(path).lower().endswith((".nii", ".nii.gz"))


def is_supported_image_path(path):
    return str(path).lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)


def collect_image_paths(source_path):
    if not source_path:
        return []

    path = Path(source_path)
    if path.is_file():
        if not is_supported_image_path(path):
            raise RuntimeError(f"Unsupported medical image file: {path}")
        return [str(path)]

    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    paths = [str(item) for item in sorted(path.rglob("*")) if item.is_file() and is_supported_image_path(item)]
    if not paths:
        raise RuntimeError(f"No supported images found under: {path}")
    return paths


def _require_nibabel():
    if nib is None:
        raise ImportError("nibabel is required for .nii/.nii.gz support. Install it with `pip install nibabel`.")


def _normalize_to_unit_range(array):
    array = np.asarray(array, dtype=np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value > min_value:
        array = (array - min_value) / (max_value - min_value)
    else:
        array = np.zeros_like(array, dtype=np.float32)
    return np.clip(array, 0.0, 1.0)


def _load_nifti_array(path):
    _require_nibabel()
    nifti = nib.load(str(path))
    array = np.asarray(nifti.get_fdata(dtype=np.float32))
    array = np.squeeze(array)
    if array.ndim != 2:
        raise RuntimeError(f"Expected a 2D NIfTI image after squeeze, got shape {array.shape} from {path}")

    array = np.nan_to_num(array, copy=False)
    return array.astype(np.float32, copy=False)


def _build_threshold_channel(array, lower_bound):
    upper_bound = float(array.max())
    if upper_bound <= lower_bound:
        return np.zeros_like(array, dtype=np.float32)
    clipped = np.clip(array, lower_bound, upper_bound)
    return _normalize_to_unit_range(clipped)


def load_nifti_image(path, channels=1):
    array = _load_nifti_array(path)
    if channels == 4:
        channels_list = [_normalize_to_unit_range(array)]
        for lower_bound in (500.0, 1000.0, 1500.0):
            channels_list.append(_build_threshold_channel(array, lower_bound))
        return np.stack(channels_list, axis=2)

    return _normalize_to_unit_range(array)[:, :, np.newaxis]


def load_medical_image(path, channels=1):
    if is_nifti_path(path):
        return load_nifti_image(path, channels=channels)

    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.clip(array, 0.0, 1.0)
    if channels == 1:
        return array[:, :, np.newaxis]
    return np.repeat(array[:, :, np.newaxis], channels, axis=2)


def save_nifti_image(image_numpy, image_path, reference_path=None):
    _require_nibabel()

    array = np.asarray(image_numpy, dtype=np.float32)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    array = np.flip(array, axis=0)

    affine = np.eye(4, dtype=np.float32)
    header = None
    if reference_path and is_nifti_path(reference_path) and os.path.isfile(reference_path):
        reference = nib.load(str(reference_path))
        affine = reference.affine
        header = reference.header.copy()

    nifti = nib.Nifti1Image(array, affine, header=header)
    nib.save(nifti, str(image_path))
