import glob
import os

from PIL import Image

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from util.medical_image_io import collect_image_paths, is_supported_image_path, load_medical_image


class SingleDataset(BaseDataset):
    """Load a single input domain for inference."""

    def __init__(self, opt):
        """Initialize this dataset class."""
        BaseDataset.__init__(self, opt)
        self.pix2pix_variant = opt.pix2pix_variant
        input_nc = self.opt.output_nc if self.opt.direction == "BtoA" else self.opt.input_nc

        if self.pix2pix_variant == "medical_s1":
            self.A_paths = self._load_medical_paths(opt)
            self.A_paths = self._apply_sample_ratio_to_paths(self.A_paths, "SingleDataset")
            self.A_paths = self._limit_paths(self.A_paths, opt.max_dataset_size, "SingleDataset")
            self.transform = get_transform(opt, grayscale=(input_nc == 1), is_medical=True)
            print(f"SingleDataset: using medical input layout with {len(self.A_paths)} samples")
        else:
            self.A_paths = sorted(make_dataset(opt.dataroot, float("inf")))
            self.A_paths = self._apply_sample_ratio_to_paths(self.A_paths, "SingleDataset(legacy)")
            self.A_paths = self._limit_paths(self.A_paths, opt.max_dataset_size, "SingleDataset(legacy)")
            self.transform = get_transform(opt, grayscale=(input_nc == 1))
            print(f"SingleDataset: using legacy input layout with {len(self.A_paths)} samples")

    def _load_medical_paths(self, opt):
        if opt.input_a_path:
            return collect_image_paths(opt.input_a_path)

        if not opt.dataroot:
            raise RuntimeError("Set --input_a_path or provide --dataroot for SingleDataset.")
        search_pattern = os.path.join(opt.dataroot, "**", opt.dataset_a_subdir, "**", "*.*")
        all_paths = glob.glob(search_pattern, recursive=True)
        a_paths = [
            path for path in all_paths if os.path.isfile(path) and is_supported_image_path(path)
        ]
        a_paths = sorted(a_paths)
        if not a_paths:
            raise RuntimeError(
                f"No medical inputs found under subdir '{opt.dataset_a_subdir}'. "
                "If you want official pix2pix/test examples, use --pix2pix_variant legacy."
            )
        return a_paths

    def _read_medical_image(self, image_path):
        return load_medical_image(image_path)

    def __getitem__(self, index):
        """Return a data point and its metadata information."""
        A_path = self.A_paths[index]

        if self.pix2pix_variant == "medical_s1":
            A_img = self._read_medical_image(A_path)
        else:
            A_img = Image.open(A_path).convert("RGB")

        A = self.transform(A_img)
        return {"A": A, "A_paths": A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
