"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data
from pathlib import Path  # For cross-platform path operations (recursive search is supported)
from PIL import Image
from util.medical_image_io import is_supported_image_path

# Defines a list of supported image file extensions (case sensitive) .
IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tif",
    ".TIF",
    ".tiff",
    ".TIFF",
    ".nii",
    ".NII",
    ".nii.gz",
    ".NII.GZ",
    ".prj",
    ".PRJ",
]


def is_image_file(filename):
    return is_supported_image_path(filename)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    dir_path = Path(dir)
    assert dir_path.is_dir(), f"{dir} is not a valid directory"

    for path in sorted(dir_path.rglob("*")):
        """ 
        Use rglob (“*”) to traverse all subdirectories recursively,
        Sorted () ensures that the order is consistent (to avoid randomness affecting reproducibility)
        """
        if path.is_file() and is_image_file(path.name):
            images.append(str(path))
    return images[: min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert("RGB")


class ImageFolder(data.Dataset):
    """
    ImageFolder is a lightweight, recursive image dataset class that is more flexible than Pytorch's original
    Core advantages: automatic traversal of subdirectories, support for custom loaders and transformations, optional return file path
    Called by other data sets in the project (such as AlignedDataset) as the underlying component.
    """

    def __init__(self, root, transform=None, return_paths=False, loader=default_loader):
        imgs = make_dataset(root)  # Gets all the image paths
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + root + "\n" "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):  # Gets the image path from the index and loads it with the loader
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
