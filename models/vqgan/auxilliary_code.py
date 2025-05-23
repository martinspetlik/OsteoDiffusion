import os
import numpy as np
import torch
import nibabel as nib
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TensorBoardLogger: self._tensorboardlogger,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def _tensorboardlogger(self, pl_module, images, batch_idx, split):
        # Optional: Implement logging to TensorBoard here
        pass

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            img = images[k].squeeze(0)
            if len(img.shape) != 4:
                continue
            img = img.permute(1, 2, 3, 0)  # (C, D, H, W) -> (D, H, W, C)
            img = img.numpy()
            filename = f"{k}_gs-{global_step:06}_e-{current_epoch:06}_b-{batch_idx:06}.nii.gz"
            path = os.path.join(root, filename)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            nifti_img = nib.Nifti1Image(img, np.eye(4))
            nib.save(nifti_img, path)

    def log_img(self, trainer, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx)
                and hasattr(pl_module, "log_images")
                and callable(pl_module.log_images)
                and self.max_images > 0):
            logger = type(trainer.logger)

            was_training = pl_module.training
            if was_training:
                pl_module.eval()

            with torch.no_grad():
                print("batch.shape ", batch.shape)
                images = pl_module.log_images(batch, split=split, pl_module=pl_module)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            save_dir = getattr(trainer.logger, 'log_dir', trainer.default_root_dir)
            self.log_local(save_dir, split, images,
                           trainer.global_step, trainer.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, trainer.global_step, split)

            if was_training:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int):
        self.log_img(trainer, pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int):
        self.log_img(trainer, pl_module, batch, batch_idx, split="val")
