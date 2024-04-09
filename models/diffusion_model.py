import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from auxiliary_functions import extract


class DiffusionModel(nn.Module):
    def __init__(self,
                 gnn_model,
                 noise_scheduler):
        super(DiffusionModel, self).__init__()
        self._name = "DiffusionModel"

        self.model = gnn_model
        self.noise_scheduler = noise_scheduler

    def q_sample(self, x_start, t, noise=None):
        """
        This method is used during the forward diffusion process
        :param x_start:
        :param t:
        :param noise:
        :return:
        """

        print("x_start.shape ", x_start.shape)
        print("t shape ", t.shape)

        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.noise_scheduler.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.noise_scheduler.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


    def p_samples(self, x, timestamp: int):
        """
        This method is used during the backward diffusion process
        :param x:
        :param timestamp:
        :return:
        """
        pass

    def forward(self, x, noise=None):
        """
        :param x: graph data
        :return:
        """
        batch_size = x.shape[0]
        if noise is None:
            noise = torch.randn_like(x)

        print("x device ", x.device)

        t = torch.randint(0, self.noise_scheduler.num_timesteps, (batch_size,), device=x.device).long()


        print("x ", x)


        x_noised = self.q_sample(x, t, noise=noise)

        print("x noised ", x_noised)
        print("t ", t)
        predicted_noise = self.model(x_noised, t)

        return noise, predicted_noise

    @torch.inference_mode()
    def sample(self, batch_size: int, return_all_timesteps: bool =False) -> torch.Tensor:
        shape = (batch_size, self.channels, self.image_size, self.image_size)

        batch, device = shape[0], "mps"

        img = torch.randn(shape, device=device)
        # This cause me a RunTimeError on MPS device due to MPS back out of memory
        # No ideas how to resolve it at this point

        # imgs = [img]

        for t in tqdm(reversed(range(0, self.num_timesteps)), total=self.num_timesteps):
            img = self.p_sample(img, t)
            # imgs.append(img)

        ret = img  # if not return_all_timesteps else torch.stack(imgs, dim=1)

        ret = self.unnormalize(ret)
        return ret
