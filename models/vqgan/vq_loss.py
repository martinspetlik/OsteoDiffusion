import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vqgan.LPIPS_aldm import LPIPSALDM
from models.vqgan.LPIPS_medical_diffusion import LPIPSMedicalDiffusion


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def adopt_weight(weight, global_step, threshold):
    return weight if global_step >= threshold else 0.0

def adopt_weight_ramp(weight, global_step, threshold, ramp_duration=1000):
    if global_step < threshold:
        return 0.0
    elif global_step < threshold + ramp_duration:
        return weight * (global_step - threshold) / ramp_duration
    else:
        return weight


def compute_entropy_loss(codebook_indices, num_codes):
    flat = codebook_indices.view(-1)
    one_hot = F.one_hot(flat, num_classes=num_codes).float()
    probs = one_hot.mean(dim=0)
    entropy = -torch.sum(probs * (probs + 1e-8).log())
    entropy_norm = entropy / torch.log(torch.tensor(float(num_codes), device=probs.device))
    return -entropy_norm  # minimize -entropy = encourage diversity

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class ImageNLayerDiscriminator(nn.Module):
    def __init__(self, input_num_channels, num_base_filters=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True):
        # def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=True):
        super(ImageNLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
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
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), _


class NLayerDiscriminator(nn.Module):
    """
    Defines a 3D PatchGAN discriminator, adapted from Pix2Pix/CycleGAN.
    Uses InstanceNorm3d for stability with batch size 1.
    """

    def __init__(self, input_num_channels=1, num_base_filters=32, n_layers=3):
        """
        Parameters:
            input_nc (int)  -- number of input channels
            ndf (int)       -- base number of filters
            n_layers (int)  -- number of conv layers
        """
        super(NLayerDiscriminator, self).__init__()

        kw = 4
        padw = 1
        norm_layer = nn.InstanceNorm3d
        use_bias = True

        sequence = [
            nn.Conv3d(input_num_channels, num_base_filters, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        nf_mult = 1
        nf_mult_prev = 1

        # Intermediate conv blocks
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(num_base_filters * nf_mult_prev, num_base_filters * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(num_base_filters * nf_mult, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        # Final conv blocks
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(num_base_filters * nf_mult_prev, num_base_filters * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(num_base_filters * nf_mult, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # Output layer
        sequence += [
            nn.Conv3d(num_base_filters * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        self.main = nn.Sequential(*sequence)

    def forward(self, x):
        return self.main(x)


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=1, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, entropy_weight=0.1, disc_conditional=False,
                 disc_num_filters=32, disc_loss="hinge", num_codebook_embeddings=256,
                 use_l2=False, perceptual_grad_scaling=1.0, LPIPS_type="aldm", discriminator_type="NLayerDiscriminator", disc_ramp_duration=0):
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

        if discriminator_type == "NLayerDiscriminator":
            # Discriminator
            self.discriminator = NLayerDiscriminator(input_num_channels=disc_in_channels,
                                                     n_layers=disc_num_layers,
                                                     num_base_filters=disc_num_filters).apply(weights_init)
        elif discriminator_type == "ImageNLayerDiscriminator":
            self.discriminator = ImageNLayerDiscriminator(input_num_channels=disc_in_channels,
                                                     n_layers=disc_num_layers,
                                                     num_base_filters=disc_num_filters).apply(weights_init)

        self.discriminator_iter_start = disc_start
        self.disc_conditional = disc_conditional
        self.discriminator_weight = disc_weight

        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss

        # Perceptual loss
        if self.LPIPS_type == "aldm":
            self.perceptual_loss_model = LPIPSALDM().eval()
        elif self.LPIPS_type == "medical_diffusion":
            self.perceptual_loss_model = LPIPSMedicalDiffusion().eval()

        self.disc_factor = disc_factor
        print(f"[VQLPIPS] Perceptual weight: {perceptual_weight}, entropy weight: {entropy_weight}")

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, skip_pass=0, cond=None,
                split="train", codebook_indices=None):

        # Pixel loss
        if self.use_l2:
            rec_loss = F.mse_loss(inputs, reconstructions)
        else:
            rec_loss = F.l1_loss(inputs, reconstructions)
        nll_loss = rec_loss

        # # Perceptual loss (with optional gradient scaling)
        if self.LPIPS_type == "medical_diffusion" or self.discriminator_type == "ImageNLayerDiscriminator":
            B, C, T, H, W = inputs.shape
            # Selects one random 2D image from each 3D Image
            frame_idx = torch.randint(0, T, [B]).cuda()
            frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
            frames = torch.gather(inputs, 2, frame_idx_selected).squeeze(2)
            frames_recon = torch.gather(reconstructions, 2, frame_idx_selected).squeeze(2)

        if self.LPIPS_type == "medical_diffusion":
            # Perceptual loss
            p_loss = 0
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss_model(frames, frames_recon).mean() * self.perceptual_weight
        else:
            # with torch.amp.autocast('cuda', enabled=False):
            #     p_loss = self.perceptual_loss(inputs, reconstructions).mean()
            # with torch.no_grad():
            #     p_loss = self.perceptual_loss(inputs, reconstructions)
            # p_loss = p_loss.detach().mean()
            # p_loss = self.perceptual_loss(inputs, reconstructions)
            # p_loss = p_loss.mean()
            p_loss = torch.tensor(0.0)
            p_loss = p_loss * self.perceptual_grad_scaling

        # Entropy loss
        entropy_loss = torch.tensor(0.0, device=inputs.device)
        if codebook_indices is not None:
            entropy_loss = compute_entropy_loss(codebook_indices, self.num_codebook_embeddings)

        # Generator update
        if optimizer_idx == 0:
            if self.discriminator_type == "ImageNLayerDiscriminator":
                logits_fake, _ = self.discriminator(frames_recon)
            else:
                logits_fake = self.discriminator(reconstructions)
            g_loss = -torch.mean(logits_fake)

            print("g loss ", g_loss)

            if torch.isnan(g_loss).any():
                print("NaN in g_loss")
                g_loss = torch.tensor(0.0, device=inputs.device)

            disc_factor = adopt_weight_ramp(self.disc_factor, global_step, threshold=self.discriminator_iter_start, ramp_duration=self.disc_ramp_duration)
            codebook_w = adopt_weight(self.codebook_weight, global_step, threshold=self.discriminator_iter_start)

            if torch.isnan(codebook_loss).any():
                print("NaN in codebook_loss")
                codebook_loss = torch.zeros_like(codebook_loss)

            print("skip_pass ", skip_pass)
            print("disc_factor ", disc_factor)

            loss = (
                    self.pixel_weight * nll_loss +
                    self.perceptual_weight * p_loss +
                    codebook_w * codebook_loss.mean() +
                    self.entropy_weight * entropy_loss
            )

            print("loss ", loss)
            if skip_pass != 0 and disc_factor > 0 and torch.isfinite(g_loss):
                loss += skip_pass * disc_factor * g_loss

            metrics = {
                "generator_total_loss": loss.clone().detach().mean(),
                "codebook_loss": codebook_loss.detach().mean(),
                "reconstruction_loss": rec_loss.detach().mean(),
                "perceptual_loss": p_loss.detach().mean(),
                "entropy_loss": entropy_loss.detach(),
                "adversarial_generator_loss": g_loss.detach(),
                "disc_factor": torch.tensor(disc_factor),
                "codebook_weight": torch.tensor(codebook_w),
            }
            return loss, metrics

        # Discriminator update
        if optimizer_idx == 1:
            if self.discriminator_type == "ImageNLayerDiscriminator":
                logits_real, _ = self.discriminator(frames.detach())
            else:
                logits_real = self.discriminator(inputs.detach())
            if self.discriminator_type == "ImageNLayerDiscriminator":
                logits_fake, _ = self.discriminator(frames_recon.detach())
            else:
                logits_fake = self.discriminator(reconstructions.detach())

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = skip_pass * disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "disc_loss": d_loss.detach(),
                "logits_real": logits_real.mean().detach(),
                "logits_fake": logits_fake.mean().detach(),
            }

            return d_loss, log
