# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# def nonlinearity(x):
#     # swish
#     return x*torch.sigmoid(x)

def Normalize(channels, num_groups=8):
    # Clamp num_groups to valid range
    num_groups = min(num_groups, channels)
    while channels % num_groups != 0 and num_groups > 1:
        num_groups -= 1

    print(f"Creating Normalize layer with {channels} channels")
    return nn.GroupNorm(num_groups, channels, eps=1e-6, affine=True)

# def Normalize(in_channels):
#     return torch.nn.GroupNorm(num_groups=8, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)
    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0, conv_shortcut=False):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        print("ResnetBlock in channels ", in_channels)

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            if conv_shortcut:
                self.shortcut = nn.Conv3d(in_channels, out_channels, 3, padding=1)
            else:
                self.shortcut = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x), inplace=True))
        h = self.conv2(self.dropout(F.silu(self.norm2(h), inplace=True)))
        return self.shortcut(x) + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w,d = q.shape
        q = q.reshape(b,c,h*w*d)
        q = q.permute(0,2,1)   # b,hwd,c
        k = k.reshape(b,c,h*w*d) # b,c,hwd
        w_ = torch.bmm(q,k)     # b,hwd,hwd    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w*d)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w,d)

        h_ = self.proj_out(h_)

        return x+h_


class Encoder(nn.Module):
    def __init__(self, *, base_channels, channel_mults=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels=1,
                 resolution, z_channels, double_z=True, use_attention, **ignore_kwargs):
        super().__init__()
        self.ch = base_channels
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.use_attention = use_attention
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        # dimensions
        dims = [*map(lambda m: base_channels * m, channel_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        print("Encoder in_out ", in_out)

        # downsampling
        self.conv_in = torch.nn.Conv3d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        self.down = nn.ModuleList()
        curr_res = resolution
        for i_level, (dim_in, dim_out) in enumerate(in_out):
            is_last = i_level >= (len(in_out) - 1)

            blocks = nn.ModuleList()
            attns = nn.ModuleList()

            print("dim in: {}, dim out: {}".format(dim_in, dim_out))

            for i_block in range(num_res_blocks):
                in_c = dim_in if i_block == 0 else dim_out
                blocks.append(ResnetBlock(in_channels=in_c, out_channels=dim_out, dropout=dropout))

                if curr_res in attn_resolutions and self.use_attention:
                    attns.append(AttnBlock(dim_out))
                else:
                    attns.append(nn.Identity())

            downsample = Downsample(dim_out, resamp_with_conv)# if not is_last else nn.Identity()

            self.downs.append(nn.ModuleDict({
                "blocks": blocks,
                "attns": attns,
                "downsample": downsample
            }))

            if not is_last:
                curr_res = curr_res // 2

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=dim_out, out_channels=dim_out, dropout=dropout)
        self.mid.attn_1 = AttnBlock(dim_out)
        self.mid.block_2 = ResnetBlock(in_channels=dim_out, out_channels=dim_out, dropout=dropout)

        # output
        self.norm_out = Normalize(dim_out)
        self.conv_out = torch.nn.Conv3d(dim_out,
                                        2 * z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        h = self.conv_in(x)

        for down in self.downs:
            for block, attn in zip(down["blocks"], down["attns"]):
                h = block(h)
                h = attn(h)
            h = down["downsample"](h)

        h = self.mid.block_1(h)
        if self.use_attention:
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        h = self.norm_out(h)
        h = F.silu(h, inplace=True)
        h = self.conv_out(h)

        return h


class Decoder(nn.Module):
    def __init__(self, *, base_channels, channel_mults=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, out_channels=1,
                 resolution, z_channels, use_attention, **ignore_kwargs):
        super().__init__()
        self.ch = base_channels
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.out_channels = out_channels
        self.z_channels = z_channels
        self.use_attention = use_attention
        self.ups = nn.ModuleList([])

        dims = [base_channels * m for m in channel_mults]
        in_out = list(zip(dims[::-1][:-1], dims[::-1][1:]))

        print("Decoder in_out ", in_out)

        # input projection from latent space
        self.conv_in = nn.Conv3d(z_channels, dims[-1], kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=dims[-1], out_channels=dims[-1], dropout=dropout)
        self.mid.attn_1 = AttnBlock(dims[-1]) if (resolution // 2**(len(channel_mults)-1)) in attn_resolutions else nn.Identity()
        self.mid.block_2 = ResnetBlock(in_channels=dims[-1], out_channels=dims[-1], dropout=dropout)

        # upsampling
        curr_res = resolution // 2**(len(channel_mults)-1)
        for i_level, (dim_in, dim_out) in enumerate(in_out):
            blocks = nn.ModuleList()
            attns = nn.ModuleList()

            for i_block in range(num_res_blocks):
                in_c = dim_in if i_block == 0 else dim_out
                blocks.append(ResnetBlock(in_channels=in_c, out_channels=dim_out, dropout=dropout))

                if curr_res in attn_resolutions and self.use_attention:
                    attns.append(AttnBlock(dim_out))
                else:
                    attns.append(nn.Identity())

            upsample = Upsample(dim_out, resamp_with_conv)
            curr_res *= 2

            self.ups.append(nn.ModuleDict({
                "blocks": blocks,
                "attns": attns,
                "upsample": upsample
            }))


        # final normalization and output conv
        self.norm_out = Normalize(dim_out)
        self.conv_out = nn.Conv3d(dim_out, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        if self.use_attention:
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for up in self.ups:
            for i, block in enumerate(up["blocks"]):
                h = block(h)
                h = up["attns"][i](h)
            h = up["upsample"](h)

        h = self.norm_out(h)
        h = F.silu(h, inplace=True)
        h = self.conv_out(h)

        return h
