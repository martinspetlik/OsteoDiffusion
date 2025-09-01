import torch
from torch.utils.data import Dataset
import numpy as np


class LatentsDataset(Dataset):
    def __init__(self, latents_file):
        data = np.load(latents_file)
        self.latents = data["latents"]
        self.conds = data["conds"]

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        x = torch.tensor(self.latents[idx], dtype=torch.float32)
        cond = torch.tensor(self.conds[idx], dtype=torch.float32)
        return x, cond


