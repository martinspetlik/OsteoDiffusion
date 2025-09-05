import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np


class LatentsDataset(Dataset):
    """
    Dataset that loads individual latent samples stored as .npz files.
    Each file should contain keys: 'latent' and 'cond'.
    """

    def __init__(self, latents_dir):
        # collect all npz files in directory
        self.files = sorted(glob.glob(os.path.join(latents_dir, "*.npz")))
        if len(self.files) == 0:
            raise ValueError(f"No .npz files found in {latents_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        x = torch.as_tensor(data["latent"], dtype=torch.float32)
        cond = torch.as_tensor(data["cond"], dtype=torch.float32)
        return x, cond
