###########
## LPIPS from https://github.com/jongdory/ALDM/blob/main/VQ-GAN/taming/modules/losses/lpips.py ##
###########
import os
import torch
import torch.nn as nn


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


def spatial_average_3d(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return x.mean([2, 3, 4], keepdim=keepdim)


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
        in0_input, in1_input = normalize_tensor(input), normalize_tensor(target)  # (self.scaling_layer(input), self.scaling_layer(target))

        # Convert to bfloat16 just before passing to self.net
        in0_input = in0_input.to(torch.float32)
        in1_input = in1_input.to(torch.float32)

        outs0, outs1 = self.net(in0_input), self.net(in1_input)

        feats0, feats1 = normalize_tensor(outs0), normalize_tensor(outs1)
        diffs = (feats0 - feats1) ** 2
        res = spatial_average_3d(diffs, keepdim=True)
        val = res

        return val

