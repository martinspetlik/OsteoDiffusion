import torch
import pytorch_lightning as pl
from pytorch_msssim import ms_ssim  # pip install pytorch-msssim
import os


class MSSSIMTopNModels3D(pl.callbacks.Callback):
    def __init__(self, val_loader, top_n=3, dirpath="checkpoints_msssim", verbose=True):
        super().__init__()
        self.val_loader = val_loader
        self.top_n = top_n
        self.dirpath = dirpath
        os.makedirs(dirpath, exist_ok=True)
        self.verbose = verbose
        self.best_models = []  # list of tuples (MS-SSIM, path)

    @torch.no_grad()
    def on_validation_end(self, trainer, pl_module):
        pl_module.eval()
        ms_ssim_scores = []

        for vol in self.val_loader:
            vol, cond = vol
            vol = vol.to(pl_module.device)  # [C, D, H, W]
            recon_vol, diff = pl_module(vol)

            vol = vol.squeeze(0)
            recon_vol = recon_vol.squeeze(0)

            # slice-by-slice MS-SSIM
            for i in range(vol.shape[1]):
                x_slice = vol[:, i, :, :].unsqueeze(0)  # [1, C, H, W]
                recon_slice = recon_vol[:, i, :, :].unsqueeze(0)
                score = ms_ssim(x_slice, recon_slice, data_range=1.0, size_average=True)
                ms_ssim_scores.append(score.item())

        avg_msssim = sum(ms_ssim_scores) / len(ms_ssim_scores)
        if self.verbose:
            print(f"Validation MS-SSIM (3D slices avg): {avg_msssim:.4f}")

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
