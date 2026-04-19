import glob
import os

from data.base_dataset import BaseDataset, get_transform
from util.medical_image_io import collect_image_paths, is_supported_image_path, load_medical_image


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class."""
        BaseDataset.__init__(self, opt)

        if opt.input_a_path or opt.input_b_path:
            if not opt.input_a_path or not opt.input_b_path:
                raise RuntimeError("Unaligned training requires both --input_a_path and --input_b_path.")
            self.A_paths = collect_image_paths(opt.input_a_path)
            self.B_paths = collect_image_paths(opt.input_b_path)
        else:
            if not opt.dataroot:
                raise RuntimeError("Set --input_a_path/--input_b_path or provide --dataroot for UnalignedDataset.")
            search_pattern_A = os.path.join(opt.dataroot, "**", opt.dataset_a_subdir, "*.*")
            search_pattern_B = os.path.join(opt.dataroot, "**", opt.dataset_b_subdir, "*.*")

            all_A_paths = glob.glob(search_pattern_A, recursive=True)
            all_B_paths = glob.glob(search_pattern_B, recursive=True)

            self.A_paths = sorted(p for p in all_A_paths if is_supported_image_path(p))
            self.B_paths = sorted(p for p in all_B_paths if is_supported_image_path(p))
        print(f"UnalignedDataset: discovered A={len(self.A_paths)} B={len(self.B_paths)}")

        self.A_paths, self.B_paths = self._apply_sample_ratio_to_pairs(
            self.A_paths,
            self.B_paths,
            "UnalignedDataset",
        )
        self.A_paths, self.B_paths = self._limit_pairs(
            self.A_paths,
            self.B_paths,
            opt.max_dataset_size,
            "UnalignedDataset",
        )

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        if self.A_size != self.B_size:
            print(
                f"Warning: number of files in domain A ({self.A_size}) "
                f"does not match domain B ({self.B_size})"
            )

        btoA = self.opt.direction == "BtoA"
        self.input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        self.output_nc = self.opt.input_nc if btoA else self.opt.output_nc
        self.transform_A = get_transform(self.opt, grayscale=(self.input_nc == 1), is_medical=True)
        self.transform_B = get_transform(self.opt, grayscale=(self.output_nc == 1), is_medical=True)

    def __getitem__(self, index):
        """Return a data point and its metadata information."""
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]

        A_img = load_medical_image(A_path, channels=self.input_nc)
        B_img = load_medical_image(B_path, channels=self.output_nc)

        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return max(self.A_size, self.B_size)
