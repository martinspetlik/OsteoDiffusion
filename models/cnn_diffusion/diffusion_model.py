import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from models.auxiliary_functions import extract
#from torch_geometric.data import Data


class DiffusionModel(nn.Module):
    def __init__(self,
                 cnn_model,
                 noise_scheduler):
        super(DiffusionModel, self).__init__()
        self._name = "DiffusionModel"

        self.model = cnn_model
        self.noise_scheduler = noise_scheduler

    def q_sample(self, x_start, t, noise):
        """
        This method is used during the forward diffusion process
        :param x_start:
        :param t:
        :param noise:
        :return:
        """
        # if noise is None:
        #     noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.noise_scheduler.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.noise_scheduler.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, data, noise=None):
        """
        :param data: data
        :return:
        """
        #print("data.shape ", data.shape)
        #data = data.float()
        batch_size = data.shape[0]

        if noise is None:
            noise = torch.randn_like(data)

        #print("noise shape ", noise.shape)
        #print("x device ", x.device)
        #print("noise mean: {}, std: {}".format(torch.mean(noise), torch.std(noise)))

        # t = torch.randint(0, self.noise_scheduler.num_timesteps, (batch_size,1), device=x.device).long()
        #
        # print("self.noise_scheduler.num_timesteps ", self.noise_scheduler.num_timesteps)

        timestamp = torch.randint(0, self.noise_scheduler.num_timesteps, (batch_size,), device=data.device).long()

        # print("t.shape ", t.shape)
        # print("t ", t)
        # print("timestamp shape ", timestamp.shape)
        #print("timestamp ", timestamp)

        # print("x ", x)

        #######
        ## p_losses
        ######
        #b, c, f, h, w, device = *data.shape, data.device

        x_noised = self.q_sample(data, timestamp, noise=noise)

        #print("x_noised.cpu().flatten() ", x_noised.cpu().flatten())

        # print(
        #     "timestamp: {}, samples mean: {}, std: {}".format(timestamp, torch.mean(x_noised.cpu().flatten()), torch.std(x_noised.cpu().flatten())))

        #data = x_noised

        # print("x noised ", x_noised)
        # print("timestamp ", timestamp)
        # print("mean noise ", torch.mean(noise))
        # print("data type ", type(data))
        # print("data ", data.shape)
        predicted_noise = self.model(x_noised, timestamp)

        # print("noise shape ", noise.shape)
        #print("predicted noise shape", predicted_noise.shape)

        predicted_noise = predicted_noise#.permute(2, 0, 1)  # (batch size, num vertices, num attrs)

        #predicted_noise = torch.unsqueeze(predicted_noise, 0)

        #print("predicted noise shape ", predicted_noise.shape)

        return noise, predicted_noise

    def p_samples(self, x, timestamp):
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

        #print("p sample x shape ", x.shape)
        #print("batched timestamps ", batched_timestamps.shape)


        # if edge_indices is not None:
        #     data = Data(x=torch.squeeze(x.float(), dim=0), edge_index=edge_indices)
        # else:

        #data = torch.squeeze(x.float(), dim=0)

        data = x.float()

        #print("data with noise ", data.cpu().flatten())

        #print("data.shape ", data.shape)

        import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        # axes.hist(data.cpu().flatten(), bins=100, density=True, label="Sampled bone density distr")
        # fig.legend()
        # plt.show()


        print("batched_timestamps ", batched_timestamps)

        #@TODO: wrong predictions
        preds = self.model(data, batched_timestamps)

        print("predicted noise ", preds.cpu().flatten())

        #preds = preds.permute(2, 0, 1) # (batch size, num vertices, num attrs)

        #print("preds shape ", preds.shape)
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
                data - betas_t * preds / sqrt_one_minus_alphas_cumprod_t
        )

        print("predicted_mean ", predicted_mean.cpu().flatten())

        # print("predicted mean shape ", predicted_mean.shape)
        # print("time stamp ", timestamp)

        if timestamp == 0:
            #return torch.clamp(predicted_mean, -1, 1)
            return predicted_mean
        else:
            posterior_variance = extract(
                self.noise_scheduler.posterior_variance, batched_timestamps, x.shape
            )
            # print("self.noise_scheduler.posterior_variance ", self.noise_scheduler.posterior_variance)
            # print("posterior variance ", posterior_variance)

            #posterior_variance = posterior_variance.expand_as(predicted_mean)

            #posterior_variance = posterior_variance[timestamp]

            noise = torch.randn_like(x)
            #print("noise shape ", noise.shape)

            scaled_noise = torch.sqrt(posterior_variance) * noise

            # print("scaled noise ",  scaled_noise.cpu().flatten())
            # print("scaled noise shape ", scaled_noise.shape)
            # print("predicted mean shape ", predicted_mean.shape)

            #x_t = predicted_mean + scaled_noise

            # if timestamp in [450, 350, 250, 150, 50, 10]:
            #     fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
            #     axes.hist(x_t.cpu().flatten(), bins=100, density=True, label="timestep: {}".format(timestamp))
            #     fig.legend()
            #     plt.show()



            return predicted_mean + scaled_noise

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
