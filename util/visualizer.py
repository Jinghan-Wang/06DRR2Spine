import ntpath
import os
import time
from pathlib import Path

import torch.distributed as dist
import wandb

from . import html, util
from util.medical_image_io import is_nifti_path


def _to_path_list(paths):
    if paths is None:
        return []
    if isinstance(paths, (list, tuple)):
        return [str(path) for path in paths]
    return [str(paths)]


def _extract_filenames(paths):
    return [ntpath.basename(path) for path in _to_path_list(paths)]


def _stem_without_medical_suffix(path):
    name = Path(path).name
    lower_name = name.lower()
    if lower_name.endswith(".nii.gz"):
        return name[:-7]
    return Path(path).stem


def _build_path_check_message(a_paths, b_paths):
    a_names = _extract_filenames(a_paths)
    b_names = _extract_filenames(b_paths)
    if not a_names and not b_names:
        return None
    if len(a_names) != len(b_names):
        return f"path_check: batch_size_mismatch A={a_names} B={b_names}"

    mismatches = [f"{a_name}!={b_name}" for a_name, b_name in zip(a_names, b_names) if a_name != b_name]
    if not mismatches:
        return None

    preview = ", ".join(mismatches[:3])
    if len(mismatches) > 3:
        preview += f", ... total={len(mismatches)}"
    return f"path_check: mismatch {preview}"


def _saved_label_name(label):
    label_map = {
        "real_A": "drr",
        "fake_B": "spine",
    }
    return label_map.get(label, label)


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save model outputs to the webpage output directory."""
    image_dir = webpage.get_image_dir()
    source_path = image_path[0]
    name = Path(source_path).stem

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        saved_label = _saved_label_name(label)
        if is_nifti_path(source_path):
            extension = ".nii.gz" if source_path.lower().endswith(".nii.gz") else ".nii"
            image_name = f"{name}_{saved_label}{extension}"
            save_path = image_dir / image_name
            util.save_image(
                util.tensor2medical(im_data),
                save_path,
                aspect_ratio=aspect_ratio,
                reference_path=source_path,
            )
            continue

        image_name = f"{name}_{saved_label}.png"
        save_path = image_dir / image_name
        util.save_image(util.tensor2im(im_data), save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(saved_label)
        links.append(image_name)

    if ims:
        webpage.add_images(ims, txts, links, width=width)


class Visualizer:
    """Display, log, and save intermediate training results."""

    def __init__(self, opt):
        self.opt = opt
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.experiment_dir = Path(opt.checkpoints_dir) / opt.name
        self.saved = False
        self.use_wandb = opt.use_wandb
        self.current_epoch = 0
        self.visuals_ext = getattr(opt, "visuals_ext", "png").lower()

        if self.use_wandb:
            if not dist.is_initialized() or dist.get_rank() == 0:
                self.wandb_project_name = getattr(opt, "wandb_project_name", "CycleGAN-and-pix2pix")
                self.wandb_run = (
                    wandb.init(project=self.wandb_project_name, name=opt.name, config=opt)
                    if not wandb.run
                    else wandb.run
                )
                self.wandb_run._label(repo="CycleGAN-and-pix2pix")
            else:
                self.wandb_run = None

        if self.use_html:
            self.web_dir = self.experiment_dir
            self.img_dir = self.experiment_dir / "Images"
            print(f"create image directory {self.img_dir}...")
            util.mkdirs([self.web_dir, self.img_dir])

        self.log_name = self.experiment_dir / "loss_log.txt"
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write(f"================ Training Loss ({now}) ================\n")

    def reset(self):
        self.saved = False

    def set_dataset_size(self, dataset_size):
        self.dataset_size = dataset_size

    def _calculate_global_step(self, epoch, epoch_iter):
        return (epoch - 1) * self.dataset_size + epoch_iter

    def display_current_results(self, visuals, epoch: int, total_iters: int, save_result=False):
        if "LOCAL_RANK" in os.environ and dist.is_initialized() and dist.get_rank() != 0:
            return

        if self.use_wandb:
            ims_dict = {}
            for label, image in visuals.items():
                ims_dict[f"results/{label}"] = wandb.Image(
                    util.tensor2im(image), caption=f"{label} - Step {total_iters}"
                )
            self.wandb_run.log(ims_dict, step=total_iters)

        if self.use_html and (save_result or not self.saved):
            self.saved = True
            if self.visuals_ext in {"nii", ".nii", "nii.gz", ".nii.gz"}:
                return

            for label, image in visuals.items():
                saved_label = _saved_label_name(label)
                image_numpy = util.tensor2im(image)
                img_path = self.img_dir / f"epoch{epoch:03d}_{saved_label}.png"
                util.save_image(image_numpy, img_path)

            webpage = html.HTML(self.web_dir, f"Experiment name = {self.name}", refresh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header(f"epoch [{n}]")
                ims, txts, links = [], [], []
                for label in visuals.keys():
                    saved_label = _saved_label_name(label)
                    img_path = f"epoch{n:03d}_{saved_label}.png"
                    ims.append(img_path)
                    txts.append(saved_label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def save_epoch_medical_results(self, visuals, epoch, a_paths=None, b_paths=None):
        """Force-save one set of medical outputs for the current epoch."""
        if not self.use_html:
            return
        if "LOCAL_RANK" in os.environ and dist.is_initialized() and dist.get_rank() != 0:
            return
        if self.visuals_ext not in {"nii", ".nii", "nii.gz", ".nii.gz"}:
            return

        a_list = _to_path_list(a_paths)
        b_list = _to_path_list(b_paths)
        reference_a = a_list[0] if a_list else None
        reference_b = b_list[0] if b_list else None
        base_name = _stem_without_medical_suffix(reference_a or reference_b or f"epoch{epoch:03d}")
        suffix = ".nii.gz" if "gz" in self.visuals_ext else ".nii"

        for label, image in visuals.items():
            saved_label = _saved_label_name(label)
            if label == "real_B" and reference_b:
                reference_path = reference_b
            else:
                reference_path = reference_a or reference_b
            save_name = f"epoch{epoch:03d}_{base_name}_{saved_label}{suffix}"
            save_path = self.img_dir / save_name
            util.save_image(util.tensor2medical(image), save_path, reference_path=reference_path)

    def plot_current_losses(self, total_iters, losses):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        if self.use_wandb:
            self.wandb_run.log(losses, step=total_iters)

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data, a_paths=None, b_paths=None, aug_params=None):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        message = f"[Rank {local_rank}] (epoch: {epoch}, iters: {iters}, time: {t_comp:.3f}, data: {t_data:.3f}) "
        for key, value in losses.items():
            message += f", {key}: {value:.3f}"

        path_check_message = _build_path_check_message(a_paths, b_paths)
        if path_check_message is not None:
            message += f", {path_check_message}"
        if isinstance(aug_params, (list, tuple)) and len(aug_params) == 1:
            aug_params = aug_params[0]
        if aug_params:
            message += f", {aug_params}"
        message += "\n"
        print(message)

        if local_rank == 0:
            with open(self.log_name, "a") as log_file:
                log_file.write(f"{message}\n")
