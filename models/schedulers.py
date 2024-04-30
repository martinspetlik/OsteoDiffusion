import torch
import torch.nn as nn
from torch.functional import F


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def sigmoid_beta_schedule(
    timesteps: int, start: int = 3, end: int = 3, tau: int = 1
) -> torch.Tensor:
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
        v_end - v_start
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class NoiseScheduler(nn.Module):
    SCHEDULER_MAPPING = {
        "linear": linear_beta_schedule,
        "cosine": cosine_beta_schedule,
        "sigmoid": sigmoid_beta_schedule,
    }

    def __init__(self, beta_scheduler_type, num_timesteps, scheduler_kwargs=None, use_cuda=True):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.beta_scheduler_fn = self.SCHEDULER_MAPPING.get(beta_scheduler_type)
        if self.beta_scheduler_fn is None:
            raise ValueError("An unknown beta scheduler type: {}".format(beta_scheduler_type))

        if scheduler_kwargs is None:
            scheduler_kwargs = {}

        self.betas = self.beta_scheduler_fn(num_timesteps, **scheduler_kwargs)
        self.alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = (
            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        self.sqrt_recip_alphas =  torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        if torch.cuda.is_available() and use_cuda:
            self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.cuda()
            self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.cuda()

