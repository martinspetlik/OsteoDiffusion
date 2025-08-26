import os
import torch
from torch_fidelity import calculate_metrics
from pytorch_lightning.callbacks import Callback
import tempfile
from torchvision.utils import save_image


import torch
from pytorch_lightning.callbacks import Callback
from torchvision.utils import save_image
import os, tempfile
from torch_fidelity import calculate_metrics
import torch.nn.functional as F
import random

class FIDTopNModels3D(Callback):
    def __init__(self, val_loader, top_n=3, dirpath="checkpoints_fid", verbose=True,
                 every_n_epochs=5, slice_step=2, resize_to=None):
        """
        Args:
            val_loader: validation dataloader
            top_n: number of best models to keep
            dirpath: checkpoint directory
            verbose: print info
            every_n_epochs: compute FID every N epochs
            slice_step: take every N-th slice along depth
            resize_to: optional (H,W) to downsample slices
        """
        super().__init__()
        self.val_loader = val_loader
        self.top_n = top_n
        self.dirpath = dirpath
        os.makedirs(dirpath, exist_ok=True)
        self.verbose = verbose
        self.best_models = []
        self.every_n_epochs = every_n_epochs
        self.slice_step = slice_step
        self.resize_to = resize_to

    @torch.no_grad()
    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs != 0:
            return  # skip computation this epoch

        pl_module.eval()
        with tempfile.TemporaryDirectory() as real_dir, tempfile.TemporaryDirectory() as fake_dir:
            idx = 0
            for vol_batch in self.val_loader:
                vol, cond = vol_batch
                vol = vol.to(pl_module.device)
                recon_vol, diff = pl_module(vol)

                vol = vol.squeeze(0)        # [C,D,H,W]
                recon_vol = recon_vol.squeeze(0)

                # Save selected slices to disk
                for i in range(0, vol.shape[1], self.slice_step):
                    r_slice = vol[:, i, :, :]
                    f_slice = recon_vol[:, i, :, :]

                    # Optional: downsample slices
                    if self.resize_to is not None:
                        r_slice = F.interpolate(r_slice.unsqueeze(0), size=self.resize_to, mode='bilinear', align_corners=False).squeeze(0)
                        f_slice = F.interpolate(f_slice.unsqueeze(0), size=self.resize_to, mode='bilinear', align_corners=False).squeeze(0)

                    # Ensure 3 channels
                    if r_slice.shape[0] == 1:
                        r_slice = r_slice.expand(3, -1, -1)
                        f_slice = f_slice.expand(3, -1, -1)

                    save_image(r_slice.cpu(), os.path.join(real_dir, f"{idx}.png"))
                    save_image(f_slice.cpu(), os.path.join(fake_dir, f"{idx}.png"))
                    idx += 1

            # Compute FID on disk
            metrics = calculate_metrics(
                input1=real_dir,
                input2=fake_dir,
                cuda=True,
                isc=False,
                fid=True,
                kid=False
            )

        fid_score = metrics['frechet_inception_distance']
        if self.verbose:
            print(f"Validation FID (3D slices, every {self.every_n_epochs} epochs): {fid_score:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(self.dirpath, f"epoch{trainer.current_epoch}_fid{fid_score:.4f}.ckpt")
        torch.save(pl_module.state_dict(), checkpoint_path)
        self.best_models.append((fid_score, checkpoint_path))

        # Keep top-N
        self.best_models.sort(key=lambda x: x[0])
        if len(self.best_models) > self.top_n:
            worst_score, worst_path = self.best_models.pop(-1)
            if os.path.exists(worst_path):
                os.remove(worst_path)
                if self.verbose:
                    print(f"Removed checkpoint with FID={worst_score:.4f}")

        if self.verbose:
            print(f"Current top-{self.top_n} FID: {[round(m[0], 4) for m in self.best_models]}")
