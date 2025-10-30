import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cnn_diffusion.aldm.discriminator import NLayerDiscriminator, weights_init

"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple


class LPIPSALDM(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        # self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = torch.hub.load("Warvito/MedicalNet-models", model="medicalnet_resnet10_23datasets", verbose=False, )
        for param in self.parameters():
            param.requires_grad = False

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        model.load_state_dict(
            torch.hub.load("Warvito/MedicalNet-models",
                           model="medicalnet_resnet10_23datasets",
                           verbose=False, ).state_dict(), strict=False)
        return model

    def forward(self, input, target):
        in0_input, in1_input = normalize_tensor(input), normalize_tensor(
            target)  # (self.scaling_layer(input), self.scaling_layer(target))
        # Convert to bfloat16 just before passing to self.net
        # in0_input = in0_input.to(torch.float32)
        # in1_input = in1_input.to(torch.float32)

        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1 = normalize_tensor(outs0), normalize_tensor(outs1)
        diffs = (feats0 - feats1) ** 2
        res = spatial_average_3d(diffs, keepdim=True)
        val = res

        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class BasicBlock(nn.Module):
    def __init__(self, chn_in, chn_out, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(chn_in, chn_out, kernel_size=3, stride=stride, padding=2, dilation=2, bias=False)
        self.bn1 = nn.BatchNorm3d(chn_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(chn_out, chn_out, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.bn2 = nn.BatchNorm3d(chn_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        if downsample is True:
            self.downsample = nn.Sequential(
                nn.Conv3d(chn_in, chn_out, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm3d(chn_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv3d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


def spatial_average_3d(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return x.mean([2, 3, 4], keepdim=keepdim)

class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


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


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=1, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_num_filters=32, disc_loss="hinge"):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        # self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_num_filters
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, skip_pass=0, cond=None, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + skip_pass * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()

            metrics = {
                "generator_total_loss": loss.clone().detach().mean(),
                "codebook_loss": codebook_loss.detach().mean(),
                "reconstruction_loss": rec_loss.detach().mean(),
                "perceptual_loss": p_loss.detach().mean(),
                "entropy_loss": torch.tensor(0.0),
                "adversarial_generator_loss": g_loss.detach().mean(),
                "disc_factor": torch.tensor(disc_factor),
                "codebook_weight": torch.tensor(0.0),
            }
            return loss, metrics

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = skip_pass * disc_factor * self.disc_loss(logits_real, logits_fake)

            metrics = {
                "disc_loss": d_loss.detach(),
                "logits_real": logits_real.mean().detach(),
                "logits_fake": logits_fake.mean().detach(),
            }
            return d_loss, metrics
