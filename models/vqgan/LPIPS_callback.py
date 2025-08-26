import torch
import lpips
from pytorch_lightning.callbacks import Callback
import os


class LPIPSTopNModels3D(Callback):
    def __init__(self, val_loader, top_n=3, net='alex', dirpath="checkpoints_lpips",
                 verbose=True, every_n_epochs=5, slice_step=2, resize_to=None):
        """
        LPIPS-based top-N model saver for 3D CT scans.
        Args:
            val_loader: validation DataLoader yielding 3D volumes [C, D, H, W]
            top_n: number of best models to keep
            net: LPIPS backbone ('alex', 'vgg', etc.)
            dirpath: folder to save checkpoints
            verbose: print info
            every_n_epochs: compute LPIPS every N epochs
            slice_step: take every N-th slice along depth
            resize_to: optional (H,W) to downsample slices
        """
        super().__init__()
        self.val_loader = val_loader
        self.top_n = top_n
        self.dirpath = dirpath
        os.makedirs(dirpath, exist_ok=True)
        self.verbose = verbose
        self.lpips_fn = lpips.LPIPS(net=net).cuda()
        self.best_models = []
        self.every_n_epochs = every_n_epochs
        self.slice_step = slice_step
        self.resize_to = resize_to

    @torch.no_grad()
    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs != 0:
            return  # skip computation this epoch

        pl_module.eval()
        lpips_scores = []

        for vol_batch in self.val_loader:
            vol, cond = vol_batch
            vol = vol.to(pl_module.device)  # [C, D, H, W]
            recon_vol, diff = pl_module(vol)

            vol = vol.squeeze(0)        # [C,D,H,W]
            recon_vol = recon_vol.squeeze(0)

            # Compute LPIPS slice-by-slice
            for i in range(0, vol.shape[1], self.slice_step):
                x_slice = vol[:, i, :, :]           # [C,H,W]
                recon_slice = recon_vol[:, i, :, :] # [C,H,W]

                # Optional downsampling
                if self.resize_to is not None:
                    x_slice = F.interpolate(x_slice.unsqueeze(0), size=self.resize_to, mode='bilinear', align_corners=False).squeeze(0)
                    recon_slice = F.interpolate(recon_slice.unsqueeze(0), size=self.resize_to, mode='bilinear', align_corners=False).squeeze(0)

                # Convert to 3-channel if needed
                if x_slice.shape[0] == 1:
                    x_slice_rgb = x_slice.expand(3, -1, -1).unsqueeze(0)
                    recon_slice_rgb = recon_slice.expand(3, -1, -1).unsqueeze(0)
                else:
                    x_slice_rgb = x_slice.unsqueeze(0)
                    recon_slice_rgb = recon_slice.unsqueeze(0)

                lpips_val = self.lpips_fn(x_slice_rgb, recon_slice_rgb)
                lpips_scores.append(lpips_val.item())

        avg_lpips = sum(lpips_scores) / len(lpips_scores)
        if self.verbose:
            print(f"Validation LPIPS (3D slices avg, every {self.every_n_epochs} epochs): {avg_lpips:.4f}")

        # Save checkpoint if top-N
        checkpoint_path = os.path.join(self.dirpath, f"epoch{trainer.current_epoch}_lpips{avg_lpips:.4f}.ckpt")
        torch.save(pl_module.state_dict(), checkpoint_path)
        self.best_models.append((avg_lpips, checkpoint_path))

        # Keep only top-N (lowest LPIPS)
        self.best_models.sort(key=lambda x: x[0])
        if len(self.best_models) > self.top_n:
            worst_lpips, worst_path = self.best_models.pop(-1)
            if os.path.exists(worst_path):
                os.remove(worst_path)
                if self.verbose:
                    print(f"Removed checkpoint with LPIPS={worst_lpips:.4f}")

        if self.verbose:
            print(f"Current top-{self.top_n} LPIPS: {[round(m[0],4) for m in self.best_models]}")
