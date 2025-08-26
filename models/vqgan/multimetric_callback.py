import torch
import lpips
from pytorch_msssim import ms_ssim
from torch_fidelity import calculate_metrics
from pytorch_lightning.callbacks import Callback
import os


class MultiMetricTopNModels3D(Callback):
    def __init__(self, val_loader, top_n=3, dirpath="checkpoints_multimetric",
                 weights=(0.33, 0.33, 0.34), verbose=True):
        """
        Multi-metric top-N model saver for 3D CT scans.
        Metrics: LPIPS (lower), MS-SSIM (higher), FID (lower)
        val_loader: PyTorch DataLoader yielding 3D volumes [C, D, H, W]
        weights: tuple of weights for (LPIPS, MS-SSIM, FID)
        """
        super().__init__()
        self.val_loader = val_loader
        self.top_n = top_n
        self.dirpath = dirpath
        os.makedirs(dirpath, exist_ok=True)
        self.weights = weights
        self.verbose = verbose

        self.lpips_fn = lpips.LPIPS(net='alex').cuda()
        self.best_models = []  # list of tuples (score, filepath)

    # --------------------------
    # Normalize metrics to [0,1] and make lower=better
    # --------------------------
    def normalize_metrics(self, lpips_val, msssim_val, fid_val):
        # adjust ranges according to your data
        lpips_range = (0, 0.2)
        msssim_range = (0, 1)
        fid_range = (0, 200)

        lpips_n = (lpips_val - lpips_range[0]) / (lpips_range[1] - lpips_range[0])
        msssim_n = 1 - (msssim_val - msssim_range[0]) / (msssim_range[1] - msssim_range[0])  # invert
        fid_n = (fid_val - fid_range[0]) / (fid_range[1] - fid_range[0])
        return lpips_n, msssim_n, fid_n

    # --------------------------
    # Compute combined score
    # --------------------------
    def compute_combined_score(self, lpips_val, msssim_val, fid_val):
        lpips_n, msssim_n, fid_n = self.normalize_metrics(lpips_val, msssim_val, fid_val)
        score = self.weights[0] * lpips_n + self.weights[1] * msssim_n + self.weights[2] * fid_n
        return score

    # --------------------------
    # Validation end callback
    # --------------------------
    @torch.no_grad()
    def on_validation_end(self, trainer, pl_module):
        pl_module.eval()
        lpips_scores, msssim_scores = [], []
        real_slices, fake_slices = [], []

        # Collect metrics slice-by-slice
        for vol in self.val_loader:
            vol, cond = vol
            vol = vol.to(pl_module.device)
            recon_vol, diff = pl_module(vol)

            for i in range(vol.shape[1]):  # depth slices
                x_slice = vol[:, i, :, :].unsqueeze(0)
                recon_slice = recon_vol[:, i, :, :].unsqueeze(0)

                lpips_scores.append(self.lpips_fn(x_slice, recon_slice).item())
                msssim_scores.append(ms_ssim(x_slice, recon_slice, data_range=1.0, size_average=True).item())

                # For FID
                real_slices.append(x_slice.cpu())
                fake_slices.append(recon_slice.cpu())

        avg_lpips = sum(lpips_scores) / len(lpips_scores)
        avg_msssim = sum(msssim_scores) / len(msssim_scores)

        # FID computation
        real_slices = torch.cat(real_slices, dim=0)
        fake_slices = torch.cat(fake_slices, dim=0)
        fid_metrics = calculate_metrics(
            input1=real_slices,
            input2=fake_slices,
            cuda=True,
            isc=False, fid=True, kid=False
        )
        fid_score = fid_metrics['frechet_inception_distance']

        # Combined score
        combined_score = self.compute_combined_score(avg_lpips, avg_msssim, fid_score)

        if self.verbose:
            print(
                f"Validation metrics - LPIPS: {avg_lpips:.4f}, MS-SSIM: {avg_msssim:.4f}, FID: {fid_score:.4f}, Combined: {combined_score:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(self.dirpath, f"epoch{trainer.current_epoch}_score{combined_score:.4f}.ckpt")
        torch.save(pl_module.state_dict(), checkpoint_path)
        self.best_models.append((combined_score, checkpoint_path))

        # Keep only top-N models (lowest combined score)
        self.best_models.sort(key=lambda x: x[0])
        if len(self.best_models) > self.top_n:
            worst_score, worst_path = self.best_models.pop(-1)
            if os.path.exists(worst_path):
                os.remove(worst_path)
                if self.verbose:
                    print(f"Removed checkpoint with combined score={worst_score:.4f}")

        if self.verbose:
            print(f"Current top-{self.top_n} combined scores: {[round(m[0], 4) for m in self.best_models]}")
