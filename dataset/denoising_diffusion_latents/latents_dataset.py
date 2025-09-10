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
    def __init__(self, latents_dir, transform=None):
        # collect all npz files in directory
        self.files = sorted(glob.glob(os.path.join(latents_dir, "*.npz")))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        latents = torch.as_tensor(data["latent"], dtype=torch.float32)
        if self.transform is not None:
            latents = self.transform(latents)
        cond = torch.as_tensor(data["cond"], dtype=torch.float32)
        return latents, cond
