import torch
import torch.nn as nn
from tqdm import tqdm
from models.auxiliary_functions import extract


def debug_tensor(name, t, step=None):
    """Helper: print stats for a tensor (min/max/mean)"""
    if isinstance(t, torch.Tensor):
        print(f"[{step}] {name}: min={t.min().item():.4f}, max={t.max().item():.4f}, mean={t.mean().item():.4f}")


class ConditionalDiffusion(nn.Module):
    """
    Conditional Denoising Diffusion model with classifier-free guidance.

    Wraps a denoising UNet (or similar CNN backbone) together with a noise scheduler
    to handle the forward noising process (q_sample) and the reverse denoising process
    (p_sample and sample).

    :param cnn_model: backbone model (e.g., UNet3D) that predicts noise from (x_noised, t, cond).
    :param image_size: tuple of input shape (C, D, H, W) or (C, H, W).
    :param noise_scheduler: diffusion noise scheduler containing betas, alphas, etc.
    :param p_null: probability of dropping condition during training (for classifier-free guidance).
    """

    def __init__(self, cnn_model, image_size, noise_scheduler, p_null=0.1):
        super().__init__()
        self._name = "ConditionalDiffusion"

        self.image_size = image_size
        self.model = cnn_model
        self.noise_scheduler = noise_scheduler

        # Probability to randomly drop conditions during training
        # (important for classifier-free guidance)
        self.p_null = p_null

    # -------------------
    # Forward process (q)
    # -------------------
    def q_sample(self, x_start, t, noise):
        """
        Diffusion forward process: add noise to clean input x_start at timestep t.

        :param x_start: clean input sample.
        :param t: timestep indices (batch,).
        :param noise: Gaussian noise tensor.
        :return: x_t (noised sample).
        """
        sqrt_alphas_cumprod_t = extract(self.noise_scheduler.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.noise_scheduler.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, data, cond, noise=None):
        """
        Training forward pass: generates noisy data and predicts noise.

        :param data: clean input (batch, C, ...).
        :param cond: conditioning vector (batch, cond_dim).
        :param noise: optional noise (if None, random Gaussian is used).
        :return: (true_noise, predicted_noise).
        """
        batch_size = data.shape[0]
        if noise is None:
            noise = torch.randn_like(data)

        # Sample random timesteps
        t = torch.randint(0, self.noise_scheduler.num_timesteps, (batch_size,), device=data.device).long()

        # Add noise to data (x_t)
        x_noised = self.q_sample(data, t, noise=noise)

        # --- classifier-free guidance dropout ---
        if self.training and self.p_null > 0:
            drop_mask = torch.rand(batch_size, 1, device=cond.device) < self.p_null
            cond = cond.clone()
            cond[drop_mask.squeeze()] = 0.0  # drop condition for subset of batch

        # Predict noise with conditional UNet
        predicted_noise = self.model(x_noised, t, cond)
        return noise, predicted_noise

    # -------------------
    # Reverse process (p)
    # -------------------
    def p_sample(self, x, t, cond, cond_scale=1.0, debug=False):
        """
        Single reverse diffusion step p(x_{t-1} | x_t).
        """
        b, *_, device = *x.shape, x.device
        batched_t = torch.full((b,), t, device=device, dtype=torch.long)

        cond = cond.to(device)

        # Conditional prediction
        pred_cond = self.model(x, batched_t, cond)

        if cond_scale != 1.0:
            # Classifier-free guidance
            pred_uncond = self.model(x, batched_t, torch.zeros_like(cond))
            preds = pred_uncond + cond_scale * (pred_cond - pred_uncond)
        else:
            preds = pred_cond

        # Scheduler params
        betas_t = extract(self.noise_scheduler.betas, batched_t, x.shape)
        sqrt_recip_alphas_t = extract(self.noise_scheduler.sqrt_recip_alphas, batched_t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.noise_scheduler.sqrt_one_minus_alphas_cumprod, batched_t, x.shape
        )

        # Predicted mean
        predicted_mean = sqrt_recip_alphas_t * (x - betas_t * preds / sqrt_one_minus_alphas_cumprod_t)

        if debug:
            debug_tensor("betas_t", betas_t, step=t)
            debug_tensor("sqrt_recip_alphas_t", sqrt_recip_alphas_t, step=t)
            debug_tensor("sqrt_one_minus_alphas_cumprod_t", sqrt_one_minus_alphas_cumprod_t, step=t)
            debug_tensor("preds", preds, step=t)
            debug_tensor("predicted_mean", predicted_mean, step=t)

        # Final step → no extra noise
        if t == 0:
            return predicted_mean
        else:
            posterior_variance = extract(self.noise_scheduler.posterior_variance, batched_t, x.shape)
            if debug:
                debug_tensor("posterior_variance", posterior_variance, step=t)
            noise = torch.randn_like(x)
            return predicted_mean + torch.sqrt(posterior_variance) * noise

    @torch.no_grad()
    def sample(self, batch_size, cond, cond_scale=1.0, debug=False):
        """
        Generate samples by iteratively denoising from pure noise.
        """
        shape = (batch_size, *self.image_size)
        samples = torch.randn(shape, device=cond.device)

        cond = cond.to(samples.device)

        num_steps = getattr(self.noise_scheduler, "num_gen_timesteps", self.noise_scheduler.num_timesteps)

        for t in reversed(range(num_steps)):
            samples = self.p_sample(samples, t, cond, cond_scale=cond_scale, debug=debug)

            if debug and (t % max(1, num_steps // 10) == 0):
                debug_tensor("samples", samples, step=t)

        return samples
