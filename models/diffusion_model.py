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
        # print("x_start.shape ", x_start.shape)
        print("t ", t)

        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.noise_scheduler.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.noise_scheduler.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        print("sqrt_alphas_cumprod_t ", sqrt_alphas_cumprod_t)
        print("sqrt_one_minus_alphas_cumprod_t ", sqrt_one_minus_alphas_cumprod_t)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, x, noise=None):
        """
        :param x: graph data
        :return:
        """
        batch_size = x.shape[0]
        if noise is None:
            noise = torch.randn_like(x)

        #print("x device ", x.device)

        print("noise mean: {}, std: {}".format(torch.mean(noise), torch.std(noise)))

        # t = torch.randint(0, self.noise_scheduler.num_timesteps, (batch_size,1), device=x.device).long()
        #
        # print("self.noise_scheduler.num_timesteps ", self.noise_scheduler.num_timesteps)

        timestamp = torch.randint(0, self.noise_scheduler.num_timesteps, (1,), device=x.device).long()

        # print("t.shape ", t.shape)
        # print("t ", t)
        # print("timestamp shape ", timestamp.shape)
        # print("timestamp ", timestamp)
        # exit()


        #print("x ", x)

        x_noised = self.q_sample(x, timestamp, noise=noise)

        print("x noised ", x_noised)
        print("timestamp ", timestamp)
        print("mean noise ", torch.mean(noise))

        predicted_noise = self.model(x_noised, timestamp)

        predicted_noise = predicted_noise.permute(2, 0, 1)  # (batch size, num vertices, num attrs)

        return noise, predicted_noise

    def p_samples(self, x, timestamp: int):
        """
        This method is used during the backward diffusion process
        :param x:
        :param timestamp:
        :return:
        """
        b, *_, device = *x.shape, x.device

        batched_timestamps = torch.full(
            (b,), timestamp, device=device, dtype=torch.long
        )

        if torch.cuda.is_available():
            batched_timestamps = batched_timestamps.cuda()
            self.noise_scheduler.betas = self.noise_scheduler.betas.cuda()
            self.noise_scheduler.sqrt_recip_alphas = self.noise_scheduler.sqrt_recip_alphas.cuda()
            self.noise_scheduler.sqrt_one_minus_alphas_cumprod = self.noise_scheduler.sqrt_one_minus_alphas_cumprod.cuda()
            self.noise_scheduler.posterior_variance = self.noise_scheduler.posterior_variance.cuda()

        # print("p sample x shape ", x.shape)
        # print("batched timestamps ", batched_timestamps.device)

        preds = self.model(x, batched_timestamps)

        preds = preds.permute(2, 0, 1) # (batch size, num vertices, num attrs)

        # print("preds shape ", preds.shape)
        #
        # print("self.noise_scheduler.betas shape ", self.noise_scheduler.betas.shape)

        betas_t = extract(self.noise_scheduler.betas, batched_timestamps, x.shape)
        sqrt_recip_alphas_t = extract(
            self.noise_scheduler.sqrt_recip_alphas, batched_timestamps, x.shape
        )
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.noise_scheduler.sqrt_one_minus_alphas_cumprod, batched_timestamps, x.shape
        )

        #print("x.shape ", x.shape)
        #print("betas_t.shape ", betas_t.shape)

        predicted_mean = sqrt_recip_alphas_t * (
                x - betas_t * preds / sqrt_one_minus_alphas_cumprod_t
        )

        if timestamp == 0:
            return predicted_mean
        else:
            posterior_variance = extract(
                self.noise_scheduler.posterior_variance, batched_timestamps, x.shape
            )
            noise = torch.randn_like(x)
            return predicted_mean + torch.sqrt(posterior_variance) * noise

    @torch.inference_mode()
    def sample(self, batch_size, inverse_transform=None, return_all_timesteps=None):
        #print("self.model.adj_matrix.shape ", self.model.adj_matrix.shape)
        shape = (batch_size, self.model.adj_matrix.shape[0], self.model.num_node_attrs)

        graphs = torch.randn(shape)#, device=self.model.adj_matrix.device)

        if torch.cuda.is_available():
            graphs = graphs.cuda()
        # This cause me a RunTimeError on MPS device due to MPS back out of memory
        # No ideas how to resolve it at this point

        print("graphs device ", graphs.device)

        # imgs = [img]

        for t in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps)), total=self.noise_scheduler.num_timesteps):
            graphs = self.p_samples(graphs, t)
            # imgs.append(img)

        print("graphs ", graphs)

        if inverse_transform is not None:
            graphs = inverse_transform(graphs)

        print("inverse transform graphs ", graphs)
        return graphs
