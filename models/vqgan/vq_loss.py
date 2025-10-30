import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils

from models.vqgan.LPIPS_aldm import LPIPSALDM
from models.vqgan.LPIPS_medical_diffusion import LPIPSMedicalDiffusion


def weights_init(m):
    """
    Initialize convolutional layer weights using Kaiming normal initialization.
    :param m: Layer module (expected nn.Conv2d).
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def adopt_weight(weight, global_step, threshold):
    """
    Enable a weight only after a specified training step.
    :param weight: Original weight value.
    :param global_step: Current training step.
    :param threshold: Step after which the weight is applied.
    :return: Weight (float).
    """
    return weight if global_step >= threshold else 0.0


def adopt_weight_ramp(weight, global_step, threshold, ramp_duration=1000):
    """
    Gradually increase a weight from 0 to its target value over a ramp period.
    :param weight: Target weight value.
    :param global_step: Current training step.
    :param threshold: Step at which to start ramping.
    :param ramp_duration: Number of steps over which to ramp up.
    :return: Scaled weight (float).
    """
    if global_step < threshold:
        return 0.0
    elif global_step < threshold + ramp_duration:
        return weight * (global_step - threshold) / ramp_duration
    else:
        return weight


def compute_entropy_loss(codebook_indices, num_codes):
    """
    Compute normalized entropy loss on codebook usage.
    Encourages diverse usage of embeddings in the quantizer.
    :param codebook_indices: Tensor of quantizer indices.
    :param num_codes: Total number of embeddings in the codebook.
    :return: Scalar entropy loss tensor.
    """
    flat = codebook_indices.view(-1)
    one_hot = F.one_hot(flat, num_classes=num_codes).float()
    probs = one_hot.mean(dim=0)
    entropy = -torch.sum(probs * (probs + 1e-8).log())
    entropy_norm = entropy / torch.log(torch.tensor(float(num_codes), device=probs.device))
    return -entropy_norm  # minimize -entropy = encourage diversity


def hinge_d_loss(logits_real, logits_fake):
    """
    Compute hinge loss for discriminator.
    :param logits_real: Discriminator outputs for real samples.
    :param logits_fake: Discriminator outputs for fake samples.
    :return: Scalar discriminator loss.
    """
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    """
    Compute standard (non-hinge) GAN loss using softplus.
    :param logits_real: Discriminator outputs for real samples.
    :param logits_fake: Discriminator outputs for fake samples.
    :return: Scalar discriminator loss.
    """
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class NLayerDiscriminator2D(nn.Module):
    """
    PatchGAN-style 2D discriminator with optional intermediate feature outputs.
    """

    def __init__(self, input_num_channels, num_base_filters=64, n_layers=3,
                 norm_layer=nn.InstanceNorm2d, use_sigmoid=False, getIntermFeat=True):
        """
        Initialize a multi-layer 2D discriminator.
        :param input_num_channels: Number of input channels.
        :param num_base_filters: Number of filters in first conv layer.
        :param n_layers: Number of downsampling layers.
        :param norm_layer: Normalization layer type (default InstanceNorm2d).
        :param use_sigmoid: Whether to add sigmoid at output.
        :param getIntermFeat: If True, returns intermediate activations.
        """
        super(NLayerDiscriminator2D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_num_channels, num_base_filters, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = num_base_filters
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        """
        Forward pass through the 2D discriminator.
        :param input: Input image tensor (B, C, H, W).
        :return: Tuple (final_output, list_of_intermediate_features).
        """
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), _


class NLayerDiscriminator3D(nn.Module):
    """
    PatchGAN-style 3D discriminator with optional intermediate feature outputs.
    """

    def __init__(self, input_num_channels, num_base_filters=64, n_layers=3,
                 norm_layer=nn.InstanceNorm3d, use_sigmoid=False, getIntermFeat=True):
        """
        Initialize a multi-layer 3D discriminator.
        :param input_num_channels: Number of input channels.
        :param num_base_filters: Number of filters in first conv layer.
        :param n_layers: Number of downsampling layers.
        :param norm_layer: Normalization layer type (default InstanceNorm3d).
        :param use_sigmoid: Whether to add sigmoid at output.
        :param getIntermFeat: If True, returns intermediate activations.
        """
        super(NLayerDiscriminator3D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv3d(input_num_channels, num_base_filters, kernel_size=kw,
                               stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = num_base_filters
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=kw,
                                stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        """
        Forward pass through the 3D discriminator.
        :param input: Input tensor (B, C, D, H, W).
        :return: Tuple (final_output, list_of_intermediate_features).
        """
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), _


class NLayerDiscriminator(nn.Module):
    """
    Flexible 3D PatchGAN discriminator supporting normalization, dropout, and spectral norm options.
    """

    def __init__(self,
                 input_num_channels=1,
                 num_base_filters=32,
                 n_layers=3,
                 norm_type="none",
                 spectral_norm="all",
                 dropout_prob=0.0):
        """
        Initialize the 3D PatchGAN discriminator.
        :param input_num_channels: Number of input channels.
        :param num_base_filters: Number of filters in first conv layer.
        :param n_layers: Number of convolutional layers.
        :param norm_type: Normalization type ('none', 'instance', 'group').
        :param spectral_norm: Where to apply spectral norm ('none', 'all', 'last', 'all_but_first').
        :param dropout_prob: Dropout probability after each activation.
        """
        super(NLayerDiscriminator, self).__init__()
        kw = 4
        padw = 1
        use_bias = True

        # Choose normalization
        if norm_type == "instance":
            def norm_layer(channels): return nn.InstanceNorm3d(channels, affine=True)
        elif norm_type == "group":
            def norm_layer(channels): return nn.GroupNorm(1, channels, affine=True)
        else:
            def norm_layer(channels): return nn.Identity()

        def maybe_spectral(conv_layer, layer_idx, total_layers):
            if spectral_norm == "all":
                return nn_utils.spectral_norm(conv_layer)
            elif spectral_norm == "all_but_first" and layer_idx != 0:
                return nn_utils.spectral_norm(conv_layer)
            elif spectral_norm == "last" and layer_idx == total_layers - 1:
                return nn_utils.spectral_norm(conv_layer)
            else:
                return conv_layer

        sequence = []
        nf_mult = 1

        # First layer
        conv1 = nn.Conv3d(input_num_channels, num_base_filters,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias)
        sequence += [
            maybe_spectral(conv1, 0, n_layers + 2),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if dropout_prob > 0:
            sequence.append(nn.Dropout3d(p=dropout_prob))

        # Intermediate layers
        for n in range(1, n_layers):
            nf_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            conv = nn.Conv3d(num_base_filters * nf_prev,
                             num_base_filters * nf_mult,
                             kernel_size=kw, stride=2, padding=padw, bias=use_bias)
            sequence += [
                maybe_spectral(conv, n, n_layers + 2),
                norm_layer(num_base_filters * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            if dropout_prob > 0:
                sequence.append(nn.Dropout3d(p=dropout_prob))

        # Final conv
        nf_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        conv_final = nn.Conv3d(num_base_filters * nf_prev,
                               num_base_filters * nf_mult,
                               kernel_size=kw, stride=1, padding=padw, bias=use_bias)
        sequence += [
            maybe_spectral(conv_final, n_layers, n_layers + 2),
            norm_layer(num_base_filters * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # Output layer
        conv_out = nn.Conv3d(num_base_filters * nf_mult, 1,
                             kernel_size=kw, stride=1, padding=padw)
        sequence += [maybe_spectral(conv_out, n_layers + 1, n_layers + 2)]

        self.main = nn.Sequential(*sequence)

    def forward(self, x):
        """
        Forward pass through the 3D PatchGAN discriminator.
        :param x: Input tensor (B, C, D, H, W).
        :return: Discriminator logits tensor.
        """
        return self.main(x)


class VQLPIPSWithDiscriminator(nn.Module):
    """
    Combined loss module for VQGAN with perceptual, reconstruction, entropy, and adversarial losses.
    Provides both generator and discriminator losses and supports 2D and 3D discriminators.
    """

    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=1, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, entropy_weight=0.1, disc_conditional=False,
                 disc_num_filters=32, disc_loss="hinge", num_codebook_embeddings=256,
                 use_l2=False, perceptual_grad_scaling=1.0, LPIPS_type="aldm",
                 discriminator_type="NLayerDiscriminator", disc_ramp_duration=0,
                 gan_weight_2d=1.0, gan_weight_3d=1.0, g_loss_weight=1.0,
                 disc_spectral_norm="none", disc_norm_type="none", disc_dropout_prob=0.0):
        """
        Initialize the VQGAN loss module with discriminators and perceptual loss models.

        :param disc_start: Step at which adversarial training begins.
        :param codebook_weight: Weight for codebook loss.
        :param pixelloss_weight: Weight for reconstruction loss.
        :param disc_num_layers: Number of convolutional layers in discriminator.
        :param disc_in_channels: Number of input channels to discriminator.
        :param disc_factor: Scaling factor for discriminator contribution.
        :param disc_weight: Weight applied to discriminator loss.
        :param perceptual_weight: Weight for perceptual loss (LPIPS).
        :param entropy_weight: Weight for entropy regularization loss.
        :param disc_conditional: If True, discriminator uses conditional input.
        :param disc_num_filters: Number of base filters in discriminator.
        :param disc_loss: Type of discriminator loss ('hinge' or 'vanilla').
        :param num_codebook_embeddings: Number of embeddings in quantizer.
        :param use_l2: If True, use L2 reconstruction loss instead of L1.
        :param perceptual_grad_scaling: Gradient scaling for perceptual loss.
        :param LPIPS_type: Type of perceptual loss ('aldm' or 'medical_diffusion').
        :param discriminator_type: Discriminator type ('NLayerDiscriminator' or 'ImageNLayerDiscriminator').
        :param disc_ramp_duration: Number of steps to ramp discriminator weight.
        :param gan_weight_2d: Weight of 2D discriminator loss.
        :param gan_weight_3d: Weight of 3D discriminator loss.
        :param g_loss_weight: Scaling factor for generator adversarial loss.
        :param disc_spectral_norm: Spectral normalization mode.
        :param disc_norm_type: Normalization layer type.
        :param disc_dropout_prob: Dropout probability for discriminator.
        """
        super().__init__()
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        self.entropy_weight = entropy_weight
        self.num_codebook_embeddings = num_codebook_embeddings
        self.use_l2 = use_l2
        self.perceptual_grad_scaling = perceptual_grad_scaling
        self.LPIPS_type = LPIPS_type
        self.discriminator_type = discriminator_type
        self.disc_ramp_duration = disc_ramp_duration

        self.discriminator = None
        self.discriminator_2d = None
        self.discriminator_3d = None
        self.gan_weight_2d = gan_weight_2d
        self.gan_weight_3d = gan_weight_3d
        self.g_loss_weight = g_loss_weight

        # Initialize discriminator(s)
        if discriminator_type == "NLayerDiscriminator":
            self.discriminator = NLayerDiscriminator(
                input_num_channels=disc_in_channels,
                n_layers=disc_num_layers,
                num_base_filters=disc_num_filters,
                spectral_norm=disc_spectral_norm,
                norm_type=disc_norm_type,
                dropout_prob=disc_dropout_prob
            ).apply(weights_init)

        elif discriminator_type == "ImageNLayerDiscriminator":
            self.discriminator_2d = NLayerDiscriminator2D(
                input_num_channels=disc_in_channels,
                n_layers=disc_num_layers,
                num_base_filters=disc_num_filters
            ).apply(weights_init)

            self.discriminator_3d = NLayerDiscriminator3D(
                input_num_channels=disc_in_channels,
                n_layers=disc_num_layers,
                num_base_filters=disc_num_filters
            ).apply(weights_init)

        self.discriminator_iter_start = disc_start
        self.disc_conditional = disc_conditional
        self.discriminator_weight = disc_weight
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss

        # Perceptual loss model
        if self.LPIPS_type == "aldm":
            self.perceptual_loss_model = LPIPSALDM().eval()
        elif self.LPIPS_type == "medical_diffusion":
            self.perceptual_loss_model = LPIPSMedicalDiffusion().eval()

        self.disc_factor = disc_factor
        print(f"[VQLPIPS] Perceptual weight: {perceptual_weight}, entropy weight: {entropy_weight}")

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, skip_pass=0, cond=None,
                split="train", codebook_indices=None, freeze_generator=False):
        """
        Compute total generator or discriminator loss for VQGAN.

        :param codebook_loss: Codebook commitment loss tensor.
        :param inputs: Real input tensor (B, C, D, H, W).
        :param reconstructions: Model reconstruction tensor.
        :param optimizer_idx: 0 for generator, 1 for discriminator.
        :param global_step: Current training step.
        :param last_layer: Final generator layer (for perceptual weighting).
        :param skip_pass: Optional loss scaling factor.
        :param cond: Optional conditioning tensor.
        :param split: Phase indicator ('train' or 'val').
        :param codebook_indices: Optional quantizer index tensor.
        :param freeze_generator: If True, skip generator gradient updates.
        :return: Tuple (loss, metrics_dict)
        """

        # === Reconstruction Loss ===
        if self.use_l2:
            rec_loss = F.mse_loss(inputs, reconstructions)
        else:
            rec_loss = F.l1_loss(inputs, reconstructions)
        nll_loss = rec_loss

        # === Perceptual Loss (LPIPS) ===
        if self.LPIPS_type == "medical_diffusion" or self.discriminator_type == "ImageNLayerDiscriminator":
            B, C, T, H, W = inputs.shape
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            frame_idx = torch.randint(0, T, (B,), device=device)
            frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
            frames = torch.gather(inputs, 2, frame_idx_selected).squeeze(2)
            frames_recon = torch.gather(reconstructions, 2, frame_idx_selected).squeeze(2)

        if self.LPIPS_type == "medical_diffusion":
            p_loss = 0
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss_model(frames, frames_recon).mean() * self.perceptual_weight
        else:
            p_loss = torch.tensor(0.0)
            p_loss = p_loss * self.perceptual_grad_scaling

        # === Entropy Regularization ===
        entropy_loss = torch.tensor(0.0, device=inputs.device)
        if codebook_indices is not None:
            entropy_loss = compute_entropy_loss(codebook_indices, self.num_codebook_embeddings)

        g_2d_loss = 0
        g_3d_loss = 0

        # === Generator Update ===
        if optimizer_idx == 0:
            if freeze_generator:
                # Skip generator updates (useful for staged training)
                return None, {
                    "generator_total_loss": torch.tensor(0.0, device=inputs.device),
                    "codebook_loss": torch.tensor(0.0, device=inputs.device),
                    "reconstruction_loss": torch.tensor(0.0, device=inputs.device),
                    "perceptual_loss": torch.tensor(0.0, device=inputs.device),
                    "entropy_loss": torch.tensor(0.0, device=inputs.device),
                    "adversarial_generator_loss": torch.tensor(0.0, device=inputs.device),
                    "disc_factor": torch.tensor(0.0, device=inputs.device),
                    "codebook_weight": torch.tensor(0.0, device=inputs.device),
                    "g_2d_loss": torch.tensor(0.0, device=inputs.device),
                    "g_3d_loss": torch.tensor(0.0, device=inputs.device),
                }

            if self.discriminator_type == "ImageNLayerDiscriminator":
                logits_2d_fake, _ = self.discriminator_2d(frames_recon)
                logits_3d_fake, _ = self.discriminator_3d(reconstructions)
                g_2d_loss = -torch.mean(logits_2d_fake)
                g_3d_loss = -torch.mean(logits_3d_fake)
                g_loss = self.gan_weight_2d * g_2d_loss + self.gan_weight_3d * g_3d_loss
            else:
                logits_fake = self.discriminator(reconstructions)
                g_loss = -torch.mean(logits_fake)

            if torch.isnan(g_loss).any():
                print("NaN in g_loss")
                g_loss = torch.tensor(0.0, device=inputs.device)

            disc_factor = adopt_weight_ramp(self.disc_factor, global_step,
                                            threshold=self.discriminator_iter_start,
                                            ramp_duration=self.disc_ramp_duration)
            codebook_w = adopt_weight(self.codebook_weight, global_step,
                                      threshold=self.discriminator_iter_start)

            if torch.isnan(codebook_loss).any():
                print("NaN in codebook_loss")
                codebook_loss = torch.zeros_like(codebook_loss)

            loss = (
                self.pixel_weight * nll_loss +
                self.perceptual_weight * p_loss +
                codebook_w * codebook_loss.mean() +
                self.entropy_weight * entropy_loss
            )

            if skip_pass != 0 and disc_factor > 0 and torch.isfinite(g_loss):
                loss += skip_pass * self.g_loss_weight * disc_factor * g_loss

            metrics = {
                "generator_total_loss": loss.clone().detach().mean(),
                "codebook_loss": codebook_w * codebook_loss.detach().mean(),
                "reconstruction_loss": rec_loss.detach().mean(),
                "perceptual_loss": p_loss.detach().mean(),
                "entropy_loss": entropy_loss.detach(),
                "adversarial_generator_loss": self.g_loss_weight * disc_factor * g_loss.detach(),
                "disc_factor": torch.tensor(disc_factor),
                "codebook_weight": torch.tensor(codebook_w),
                "g_2d_loss": g_2d_loss,
                "g_3d_loss": g_3d_loss,
            }

            return loss, metrics

        # === Discriminator Update ===
        if optimizer_idx == 1:
            disc_factor = adopt_weight_ramp(self.disc_factor, global_step,
                                            threshold=self.discriminator_iter_start,
                                            ramp_duration=self.disc_ramp_duration)
            d_2d_loss, d_3d_loss = 0, 0
            if self.discriminator_type == "ImageNLayerDiscriminator":
                logits_2d_real, _ = self.discriminator_2d(frames.detach())
                logits_3d_real, _ = self.discriminator_3d(inputs.detach())

                logits_2d_fake, _ = self.discriminator_2d(frames_recon.detach())
                logits_3d_fake, _ = self.discriminator_3d(reconstructions.detach())

                d_2d_loss = self.disc_loss(logits_2d_real, logits_2d_fake)
                d_3d_loss = self.disc_loss(logits_3d_real, logits_3d_fake)
                d_loss = disc_factor * (self.gan_weight_2d * d_2d_loss + self.gan_weight_3d * d_3d_loss)

                log = {
                    "disc_loss": d_loss.detach(),
                    "logits_2d_real": logits_2d_real.mean().detach(),
                    "logits_2d_fake": logits_2d_fake.mean().detach(),
                    "logits_3d_real": logits_3d_real.mean().detach(),
                    "logits_3d_fake": logits_3d_fake.mean().detach(),
                }

            else:
                logits_real = self.discriminator(inputs.detach())
                logits_fake = self.discriminator(reconstructions.detach())
                d_loss = skip_pass * disc_factor * self.disc_loss(logits_real, logits_fake)

                log = {
                    "disc_loss": d_loss.detach(),
                    "logits_2d_real": logits_real.mean().detach(),
                    "logits_2d_fake": logits_fake.mean().detach(),
                    "logits_3d_real": logits_real.mean().detach(),
                    "logits_3d_fake": logits_fake.mean().detach(),
                }

            return d_loss, log
