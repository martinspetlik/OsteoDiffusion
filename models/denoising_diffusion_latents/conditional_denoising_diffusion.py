import torch
import torch.nn as nn
from tqdm import tqdm
from models.auxiliary_functions import extract
from torch.functional import F


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

    -------------------
    Theoretical background
    -------------------
    Diffusion models define two stochastic processes:

    1. Forward process (q): progressively add Gaussian noise to a clean sample x₀
       until it becomes pure noise x_T.

       Formula:
         q(x_t | x₀) = N(x_t ; sqrt(α̅_t) * x₀ , (1 - α̅_t) I)

       where:
         - α_t = 1 - β_t
         - α̅_t = ∏_{s=1}^t α_s (cumulative product of noise factors)
         - β_t are variance schedule values.

    2. Reverse process (p): train a neural network ε_θ(x_t, t, cond) to predict the noise,
       so we can invert the process.

       Formula:
         p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))

       where μ_θ depends on ε_θ and the diffusion equations.
    """

    def __init__(self, cnn_model, image_size, noise_scheduler, cond_scale=0.1):
        super().__init__()
        self._name = "ConditionalDiffusion"

        self.image_size = image_size
        self.model = cnn_model
        self.noise_scheduler = noise_scheduler

        # Probability to randomly drop conditions during training
        # (important for classifier-free guidance, so the model learns unconditional too)
        self.cond_scale = cond_scale
        #self.parameterization = "x0" #"eps"
        self.log_every_t = 10

    # -------------------
    # Forward process (q)
    # -------------------
    def q_sample(self, x_start, t, noise):
        """
        Diffusion forward process: add noise to clean input x_start at timestep t.

        Formula:
        --------
        q(x_t | x₀) = sqrt(α̅_t) * x₀ + sqrt(1 - α̅_t) * ε

        :param x_start: clean input sample.
        :param t: timestep indices (batch,).
        :param noise: Gaussian noise tensor ε ~ N(0, I).
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

        Steps:
        1. Sample random timestep t ~ Uniform(0, T).
        2. Add Gaussian noise to the clean sample: x_t = q(x_t | x₀).
        3. Randomly drop conditioning vector (classifier-free guidance trick).
        4. Predict noise ε_θ(x_t, t, cond) with CNN backbone.

        :param data: clean input (batch, C, ...).
        :param cond: conditioning vector (batch, cond_dim).
        :param noise: optional noise (if None, random Gaussian is used).
        :return: (true_noise ε, predicted_noise ε_θ).
        """
        batch_size = data.shape[0]
        if noise is None:
            noise = torch.randn_like(data)

        # Sample random timesteps
        t = torch.randint(0, self.noise_scheduler.num_timesteps, (batch_size,), device=data.device).long()

        # Add noise to data (x_t)
        x_noised = self.q_sample(data, t, noise=noise)

        # --- classifier-free guidance dropout ---
        if self.training and self.cond_scale > 0:
            drop_mask = torch.rand(batch_size, 1, device=cond.device) < self.cond_scale
            cond = cond.clone()
            cond[drop_mask.squeeze()] = 0.0  # drop condition for subset of batch

        # Predict noise with conditional UNet
        predicted_noise = self.model(x_noised, t, cond)

        return noise, predicted_noise

    def noise_like(self, shape, device, repeat=False):
        """
        Utility: generate Gaussian noise with given shape.
        If repeat=True → use same noise vector for batch.
        """
        repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
            shape[0], *((1,) * (len(shape) - 1))
        )
        noise = lambda: torch.randn(shape, device=device)
        return repeat_noise() if repeat else noise()

    def q_posterior(self, x_start, x_t, t):
        """
        Compute the true DDPM posterior q(x_{t-1} | x_t, x₀).
        Returns (posterior_mean, posterior_variance, posterior_log_variance_clipped).

        Formula:
        --------
        q(x_{t-1} | x_t, x₀) = N(x_{t-1} ; μ_t(x_t, x₀), β̃_t I)

        where:
          μ_t = (√α̅_{t-1} β_t / (1-α̅_t)) * x₀ + (√α_t (1-α̅_{t-1}) / (1-α̅_t)) * x_t
          β̃_t = (1-α̅_{t-1})/(1-α̅_t) * β_t   (posterior variance)
        """
        # alphas, alphas_cumprod, betas are 1D tensors in scheduler
        # extract per-batch broadcastable versions
        betas_t = extract(self.noise_scheduler.betas, t, x_t.shape)  # beta_t
        alphas_t = extract(1.0 - self.noise_scheduler.betas, t, x_t.shape)  # alpha_t
        alphas_cumprod_t = extract(self.noise_scheduler.alphas_cumprod, t, x_t.shape)  # alpha_bar_t

        # compute alpha_bar_{t-1} by padding alphas_cumprod (on the fly)
        # scheduler.alphas_cumprod is 1D; create alphas_cumprod_prev 1D then extract
        alphas_cumprod_1d = self.noise_scheduler.alphas_cumprod
        alphas_cumprod_prev_1d = F.pad(alphas_cumprod_1d[:-1], (1, 0), value=1.0)
        alphas_cumprod_prev_t = extract(alphas_cumprod_prev_1d, t, x_t.shape)

        # posterior mean coefficients (DDPM textbook)
        coef1 = (betas_t * torch.sqrt(alphas_cumprod_prev_t)) / (1.0 - alphas_cumprod_t + 1e-12)
        coef2 = (torch.sqrt(alphas_t) * (1.0 - alphas_cumprod_prev_t)) / (1.0 - alphas_cumprod_t + 1e-12)

        posterior_mean = coef1 * x_start + coef2 * x_t

        posterior_variance = extract(self.noise_scheduler.posterior_variance, t, x_t.shape)
        # numerical stable log var
        posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Recover x₀ from x_t and predicted noise ε.

        Formula:
        --------
          x₀ = (x_t - sqrt(1 - α̅_t) * ε) / sqrt(α̅_t)

        Note: use scheduler.sqrt_alphas_cumprod (which is sqrt(α̅_t)).
        """
        sqrt_alpha_bar_t = extract(self.noise_scheduler.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_bar_t = extract(self.noise_scheduler.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        # x0 = (x_t - sqrt(1 - alpha_bar) * eps) / sqrt(alpha_bar)
        x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * noise) / (sqrt_alpha_bar_t + 1e-12)
        return x0_pred

    def p_mean_variance(self, x_t, t, clip_denoised: bool):
        """
        Compute model mean & variance for p(x_{t-1} | x_t).

        Steps:
        1. Model predicts ε_θ(x_t, t).
        2. Reconstruct x₀_pred from noise prediction.
        3. Use q_posterior formula to get mean/variance/log-variance.

        :param clip_denoised: whether to clamp x₀_pred to [-1, 1] for stability.
        """
        # model should return eps prediction for shape x_t
        eps_pred = self.model(x_t, t)  # if your model signature is different adapt
        # recover x0
        x0_pred = self.predict_start_from_noise(x_t, t, eps_pred)

        if clip_denoised:
            x0_pred = x0_pred.clamp(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x0_pred, x_t=x_t, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x_t, t, clip_denoised=True, repeat_noise=False, deterministic=False):
        """
        One reverse step:
         - deterministic=False: stochastic DDPM step with posterior variance
         - deterministic=True: return posterior mean (no added noise)  (useful to inspect)

        Formula:
        --------
        x_{t-1} = μ_θ(x_t, t) + σ_t * z,   z ~ N(0, I)
        (unless t=0, then no noise is added)
        """
        b, *_, device = *x_t.shape, x_t.device
        model_mean, posterior_variance, posterior_log_variance = self.p_mean_variance(
            x_t=x_t, t=t, clip_denoised=clip_denoised
        )

        if deterministic:
            return model_mean
        else:
            noise = self.noise_like(x_t.shape, device, repeat=repeat_noise)
            # when t == 0 we should not add noise (nonzero_mask = 0 for t==0)
            nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_t.shape) - 1))).to(device)
            return model_mean + nonzero_mask * torch.sqrt(posterior_variance) * noise

    @torch.no_grad()
    def sample(self, batch_size, image_size, return_intermediates=False, clip_denoised=False, deterministic=False):
        """
        Full sampling loop using reverse steps. Uses scheduler.num_timesteps and sampler p_sample above.

        Steps:
        -------
        1. Start from Gaussian noise x_T ~ N(0, I).
        2. Iteratively denoise: for t = T ... 1
             x_{t-1} ~ p_θ(x_{t-1} | x_t).
        3. Optionally return intermediates for visualization.

        Output: final generated sample x₀.
        """
        device = next(self.model.parameters()).device
        img = torch.randn(batch_size, *image_size, device=device, dtype=torch.float32)
        intermediates = [img.clone()]

        T = int(self.noise_scheduler.num_timesteps)
        for i in tqdm(reversed(range(0, T)), desc='Sampling t', total=T):
            t_tensor = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t_tensor, clip_denoised=clip_denoised, repeat_noise=False,
                                deterministic=deterministic)
            # debug print
            print(f"[t={i}] img min {img.min().item():.6f}, max {img.max().item():.6f}, std {img.std().item():.6f}")
            if (i % max(1, T // 10)) == 0 or i == T - 1:
                intermediates.append(img.clone())

        if return_intermediates:
            return img, intermediates
        return img
