import os
import torch
from torch_fidelity import calculate_metrics
from pytorch_lightning.callbacks import Callback
import tempfile
from torchvision.utils import save_image


class FIDTopNModels3D(Callback):
    def __init__(self, val_loader, top_n=3, dirpath="checkpoints_fid", verbose=True):
        super().__init__()
        self.val_loader = val_loader
        self.top_n = top_n
        self.dirpath = dirpath
        os.makedirs(dirpath, exist_ok=True)
        self.verbose = verbose
        self.best_models = []  # list of tuples (FID, path)

    @torch.no_grad()
    def on_validation_end(self, trainer, pl_module):
        pl_module.eval()
        real_slices = []
        fake_slices = []

        for vol in self.val_loader:
            vol, cond = vol
            vol = vol.to(pl_module.device)
            recon_vol, diff = pl_module(vol)

            vol = vol.squeeze(0)
            recon_vol = recon_vol.squeeze(0)

            # Flatten depth slices into 2D images
            for i in range(vol.shape[1]):
                real_slices.append(vol[:, i, :, :].cpu())
                fake_slices.append(recon_vol[:, i, :, :].cpu())

        # Convert to [N,C,H,W] for torch-fidelity
        real_slices = torch.cat(real_slices, dim=0)
        fake_slices = torch.cat(fake_slices, dim=0)

        with tempfile.TemporaryDirectory() as real_dir, tempfile.TemporaryDirectory() as fake_dir:
            for idx, (r, f) in enumerate(zip(real_slices, fake_slices)):
                save_image(r, os.path.join(real_dir, f"{idx}.png"))
                save_image(f, os.path.join(fake_dir, f"{idx}.png"))

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
            print(f"Validation FID (3D slices): {fid_score:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(self.dirpath, f"epoch{trainer.current_epoch}_fid{fid_score:.4f}.ckpt")
        torch.save(pl_module.state_dict(), checkpoint_path)
        self.best_models.append((fid_score, checkpoint_path))

        # Keep top-N (lowest FID)
        self.best_models.sort(key=lambda x: x[0])
        if len(self.best_models) > self.top_n:
            worst_score, worst_path = self.best_models.pop(-1)
            if os.path.exists(worst_path):
                os.remove(worst_path)
                if self.verbose:
                    print(f"Removed checkpoint with FID={worst_score:.4f}")

        if self.verbose:
            print(f"Current top-{self.top_n} FID: {[round(m[0],4) for m in self.best_models]}")
