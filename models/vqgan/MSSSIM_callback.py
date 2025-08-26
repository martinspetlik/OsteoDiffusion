import torch
import pytorch_lightning as pl
from pytorch_msssim import ms_ssim  # pip install pytorch-msssim
import os


class MSSSIMTopNModels3D(pl.callbacks.Callback):
    def __init__(self, val_loader, top_n=3, dirpath="checkpoints_msssim", verbose=True,
                 every_n_epochs=5, slice_step=2, resize_to=None):
        """
        MS-SSIM-based top-N model saver for 3D CT scans.
        Args:
            val_loader: validation DataLoader yielding 3D volumes [C, D, H, W]
            top_n: number of best models to keep
            dirpath: folder to save checkpoints
            verbose: print info
            every_n_epochs: compute MS-SSIM every N epochs
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
        ms_ssim_scores = []

        for vol_batch in self.val_loader:
            vol, cond = vol_batch
            vol = vol.to(pl_module.device)  # [C, D, H, W]
            recon_vol, diff = pl_module(vol)

            vol = vol.squeeze(0)        # [C,D,H,W]
            recon_vol = recon_vol.squeeze(0)

            # slice-by-slice MS-SSIM
            for i in range(0, vol.shape[1], self.slice_step):
                x_slice = vol[:, i, :, :]
                recon_slice = recon_vol[:, i, :, :]

                # Optional downsampling
                if self.resize_to is not None:
                    x_slice = F.interpolate(x_slice.unsqueeze(0), size=self.resize_to, mode='bilinear', align_corners=False).squeeze(0)
                    recon_slice = F.interpolate(recon_slice.unsqueeze(0), size=self.resize_to, mode='bilinear', align_corners=False).squeeze(0)

                x_slice = x_slice.unsqueeze(0)       # [1,C,H,W]
                recon_slice = recon_slice.unsqueeze(0)

                score = ms_ssim(x_slice, recon_slice, data_range=1.0, size_average=True)
                ms_ssim_scores.append(score.item())

        avg_msssim = sum(ms_ssim_scores) / len(ms_ssim_scores)
        if self.verbose:
            print(f"Validation MS-SSIM (3D slices avg, every {self.every_n_epochs} epochs): {avg_msssim:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(self.dirpath, f"epoch{trainer.current_epoch}_msssim{avg_msssim:.4f}.ckpt")
        torch.save(pl_module.state_dict(), checkpoint_path)
        self.best_models.append((avg_msssim, checkpoint_path))

        # Keep top-N (highest MS-SSIM)
        self.best_models.sort(key=lambda x: x[0], reverse=True)
        if len(self.best_models) > self.top_n:
            worst_score, worst_path = self.best_models.pop(-1)
            if os.path.exists(worst_path):
                os.remove(worst_path)
                if self.verbose:
                    print(f"Removed checkpoint with MS-SSIM={worst_score:.4f}")

        if self.verbose:
            print(f"Current top-{self.top_n} MS-SSIM: {[round(m[0],4) for m in self.best_models]}")