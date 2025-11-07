import torch
import torch.nn as nn
import torch.nn.functional as F


def Normalize(channels, num_groups=8):
    """
    Group normalization layer with automatic group number adjustment.
    :param channels: Number of input channels.
    :param num_groups: Desired number of groups (default 8). Adjusted automatically if not divisible by channels.
    :return: nn.GroupNorm layer.
    """
    num_groups = min(num_groups, channels)
    while channels % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return nn.GroupNorm(num_groups, channels, eps=1e-5, affine=True)


class Upsample(nn.Module):
    """
    3D upsampling block with optional convolution.
    """
    def __init__(self, in_channels, with_conv):
        """
        Initialize the upsampling block.
        :param in_channels: Number of input channels.
        :param with_conv: If True, applies a convolution after upsampling.
        """
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        """
        Forward pass of the upsampling layer.
        :param x: Input tensor.
        :return: Upsampled (and optionally convolved) tensor.
        """
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    3D downsampling block with optional convolution.
    """
    def __init__(self, in_channels, with_conv):
        """
        Initialize the downsampling block.
        :param in_channels: Number of input channels.
        :param with_conv: If True, applies a convolution with stride=2 for downsampling.
        """
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        """
        Forward pass of the downsampling layer.
        :param x: Input tensor.
        :return: Downsampled tensor.
        """
        if self.with_conv:
            pad = (0, 1, 0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    """
    3D residual block with normalization, dropout, and skip connection.
    """
    def __init__(self, in_channels, out_channels=None, dropout=0.0, conv_shortcut=False):
        """
        Initialize the residual block.
        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels (default same as in_channels).
        :param dropout: Dropout rate (default 0.0).
        :param conv_shortcut: If True, use 3x3 conv for skip connection instead of 1x1.
        """
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
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
        """
        Forward pass through the residual block.
        :param x: Input tensor.
        :return: Output tensor after residual connection.
        """
        h = self.norm1(x)
        h = F.silu(h, inplace=True)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h, inplace=True)
        h = self.dropout(h)
        h = self.conv2(h)
        x_short = self.shortcut(x)
        out = x_short + h
        return out


class AttnBlock(nn.Module):
    """
    3D self-attention block for feature refinement.
    """
    def __init__(self, in_channels):
        """
        Initialize the attention block.
        :param in_channels: Number of input channels.
        """
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.k = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.v = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = torch.nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass for the attention block.
        :param x: Input tensor.
        :return: Tensor with attention applied.
        """
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w, d = q.shape
        q = q.reshape(b, c, h * w * d).permute(0, 2, 1)
        k = k.reshape(b, c, h * w * d)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        v = v.reshape(b, c, h * w * d)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w, d)
        h_ = self.proj_out(h_)
        return x + h_


class Encoder(nn.Module):
    """
    3D convolutional encoder with residual and attention blocks.
    """
    def __init__(self, *, base_channels, channel_mults=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels=1,
                 resolution, z_channels, double_z=True, use_attention=True, **ignore_kwargs):
        """
        Initialize the 3D encoder network.
        :param base_channels: Base number of feature channels.
        :param channel_mults: Multipliers for channel count at each level.
        :param num_res_blocks: Number of residual blocks per level.
        :param attn_resolutions: List of resolutions where attention is applied.
        :param dropout: Dropout rate.
        :param resamp_with_conv: Use convolutional resampling if True.
        :param in_channels: Number of input channels.
        :param resolution: Input resolution.
        :param z_channels: Number of latent channels.
        :param double_z: If True, output twice as many channels (for mean and logvar).
        :param use_attention: Whether to use attention blocks.
        """
        super().__init__()
        self.ch = base_channels
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.use_attention = use_attention
        self.downs = nn.ModuleList([])

        dims = [base_channels * m for m in channel_mults]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.conv_in = nn.Conv3d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        for i_level, (dim_in, dim_out) in enumerate(in_out):
            is_last = i_level >= (len(in_out) - 1)
            blocks = nn.ModuleList()
            attns = nn.ModuleList()

            for i_block in range(num_res_blocks):
                in_c = dim_in if i_block == 0 else dim_out
                blocks.append(ResnetBlock(in_channels=in_c, out_channels=dim_out, dropout=dropout))
                if curr_res in attn_resolutions and self.use_attention:
                    attns.append(AttnBlock(dim_out))
                else:
                    attns.append(nn.Identity())

            downsample = Downsample(dim_out, resamp_with_conv)
            self.downs.append(nn.ModuleDict({
                "blocks": blocks,
                "attns": attns,
                "downsample": downsample
            }))
            if not is_last:
                curr_res //= 2

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(dim_out, dim_out, dropout=dropout)
        self.mid.attn_1 = AttnBlock(dim_out)
        self.mid.block_2 = ResnetBlock(dim_out, dim_out, dropout=dropout)

        self.norm_out = Normalize(dim_out)
        self.conv_out = nn.Conv3d(dim_out, 2 * z_channels if double_z else z_channels, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Forward pass through the encoder.
        :param x: Input tensor.
        :return: Encoded latent tensor.
        """
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
    """
    3D convolutional decoder with residual and attention blocks.
    """
    def __init__(self, *, base_channels, channel_mults=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, out_channels=1,
                 resolution, z_channels, use_attention=True, out_activation=None, **ignore_kwargs):
        """
        Initialize the 3D decoder network.
        :param base_channels: Base number of feature channels.
        :param channel_mults: Multipliers for channel count at each level.
        :param num_res_blocks: Number of residual blocks per level.
        :param attn_resolutions: List of resolutions where attention is applied.
        :param dropout: Dropout rate.
        :param resamp_with_conv: Use convolutional resampling if True.
        :param out_channels: Number of output channels.
        :param resolution: Target output resolution.
        :param z_channels: Number of latent input channels.
        :param use_attention: Whether to use attention blocks.
        :param out_activation: Final activation ('tanh', 'sigmoid', or None).
        """
        super().__init__()
        self.ch = base_channels
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.out_channels = out_channels
        self.z_channels = z_channels
        self.use_attention = use_attention
        self.ups = nn.ModuleList([])

        if out_activation == 'tanh':
            self.output_activation = nn.Tanh()
        elif out_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        elif out_activation is None:
            self.output_activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported output activation: {out_activation}")

        dims = [base_channels * m for m in channel_mults]
        in_out = list(zip(dims[::-1][:-1], dims[::-1][1:]))

        self.conv_in = nn.Conv3d(z_channels, dims[-1], kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(dims[-1], dims[-1], dropout=dropout)
        self.mid.attn_1 = AttnBlock(dims[-1]) if (resolution // 2 ** (len(channel_mults) - 1)) in attn_resolutions else nn.Identity()
        self.mid.block_2 = ResnetBlock(dims[-1], dims[-1], dropout=dropout)

        curr_res = resolution // 2 ** (len(channel_mults) - 1)
        for dim_in, dim_out in in_out:
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for i_block in range(num_res_blocks):
                in_c = dim_in if i_block == 0 else dim_out
                blocks.append(ResnetBlock(in_c, dim_out, dropout=dropout))
                if curr_res in attn_resolutions and self.use_attention:
                    attns.append(AttnBlock(dim_out))
                else:
                    attns.append(nn.Identity())
            upsample = Upsample(dim_out, resamp_with_conv)
            curr_res *= 2
            self.ups.append(nn.ModuleDict({"blocks": blocks, "attns": attns, "upsample": upsample}))

        self.norm_out = Normalize(dim_out)
        self.conv_out = nn.Conv3d(dim_out, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        """
        Forward pass through the decoder.
        :param z: Latent tensor.
        :return: Reconstructed output tensor.
        """
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        if self.use_attention:
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        for up in self.ups:
            for i, block in enumerate(up["blocks"]):
                h = block(h)
                h = up["attns"][i](h)
            h = up["upsample"](h)

        h = self.norm_out(h)
        h = F.silu(h, inplace=True)
        h = self.conv_out(h)
        return self.output_activation(h)
