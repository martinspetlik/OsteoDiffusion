import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from models.auxiliary_functions import extract
#from torch_geometric.data import Data


class CategoricalDiffusionModel(nn.Module):
    def __init__(self,
                 cnn_model,
                 noise_scheduler):
        super(CategoricalDiffusionModel, self).__init__()
        self._name = "CategoricalDiffusionModel"

        self.model = cnn_model
        self.noise_scheduler = noise_scheduler

    def q_sample(self, x_start, t):
        # For t > 0: randomly corrupt tokens (e.g., uniform or learned transition matrix)
        # For t = 0: return x_start unchanged

        if t == 0:
            return x_start

        # Example: uniform corruption
        corrupted = x_start.clone()
        mask = torch.rand_like(x_start.float()) < self.noise_scheduler.alphas_cumprod[t]
        random_tokens = torch.randint_like(x_start, low=0, high=self.num_classes)
        corrupted[~mask] = random_tokens[~mask]
        return corrupted

    def forward(self, x, target=None):
        b = x.size(0)
        t = torch.randint(0, self.noise_scheduler.num_timesteps, (b,), device=x.device).long()

        x_t = self.q_sample(x, t)

        logits = self.model(x_t, t)  # Shape: (B, C=K, H, W, ...) for classification

        if target is not None:
            loss = F.cross_entropy(logits.permute(0, 2, 3, 4, 1).reshape(-1, self.num_classes),
                                   target.reshape(-1))
            return loss, logits
        return logits

    def p_sample(self, x_t, t):
        logits = self.model(x_t, t)
        probs = F.softmax(logits, dim=1)  # Shape: (B, K, H, W)

        if t == 0:
            return probs.argmax(dim=1)  # final prediction

        # Sample from predicted distribution
        x_prev = torch.multinomial(probs.permute(0, 2, 3, 4, 1).reshape(-1, self.num_classes), 1)
        x_prev = x_prev.view(x_t.shape)
        return x_prev

    @torch.inference_mode()
    def sample(self, batch_size, inverse_transform=None,  dataset=None, return_all_timesteps=None):
        #print("self.model.adj_matrix.shape ", self.model.adj_matrix.shape)

        shape = (batch_size, 1, self.model.dim,  self.model.dim,  self.model.dim)

        samples = torch.randn(shape)#, device=self.model.adj_matrix.device)

        print("samples ", samples.shape)

        if torch.cuda.is_available():
            samples = samples.cuda()
        # This cause me a RunTimeError on MPS device due to MPS back out of memory
        # No ideas how to resolve it at this point

        print("graphs device ", samples.device)
        print("self.noise_scheduler.num_timesteps ", self.noise_scheduler.num_timesteps)
        print("self.noise_scheduler.num_gen_timesteps ", self.noise_scheduler.num_gen_timesteps)

        #edge_indices = dataset.edge_indices if dataset is not None else None

        # imgs = [img]

        print("samples mean: {}, std: {}".format(torch.mean(samples.cpu().flatten()), torch.std(samples.cpu().flatten())))

        for t in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps)), total=self.noise_scheduler.num_gen_timesteps):
            print("t ", t)
            samples = self.p_samples(samples, t)

            print("samples.cpu().flatten() ", samples.cpu().flatten())

            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            # axes.hist(samples.cpu().flatten(), bins=100, density=True, label="Sampled bone density distr")
            # fig.legend()
            # plt.show()


            # imgs.append(img)

        print("samples ", samples)

        inv_samples = samples
        if inverse_transform is not None:
            inv_samples = inverse_transform(samples)

        print("inverse transform graphs ", samples)

        return inv_samples, samples
