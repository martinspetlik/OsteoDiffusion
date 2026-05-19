# Largely taken and adapted from
# https://github.com/FirasGit/medicaldiffusion/blob/master/ddpm/diffusion.py, which was build on https://github.com/lucidrains/video-diffusion-pytorch"
import math
import torch
from torch import nn, einsum
from functools import partial
from einops import rearrange
from einops_exts import check_shape, rearrange_many
from rotary_embedding_torch import RotaryEmbedding

from models.denoising_diffusion_latents.Unet3D_conditioning import (
    CONDITIONING_MODES,
    ResnetBlockFiLM,
    CrossAttentionConditioning,
    ConcatConditioning,
)

# ---------------------------
# Utility functions
# ---------------------------

def exists(x):
    """Check if a variable is not None."""
    return x is not None


def is_odd(n):
    """Return True if a number is odd."""
    return (n % 2) == 1


def default(val, d):
    """
    Return val if it exists, otherwise return default d.
    If d is a callable, call it and return its result.
    """
    if exists(val):
        return val
    return d() if callable(d) else d


# ---------------------------
# Core building blocks
# ---------------------------

class Residual(nn.Module):
    """
    Wraps a function (fn) with a residual connection.
    Output = fn(x) + x
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    """
    Creates sinusoidal positional embeddings.
    Useful for injecting position or time information into the network.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        # Compute frequencies using log scale
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # Outer product between input and frequencies
        emb = x[:, None] * emb[None, :]
        # Concatenate sin and cos embeddings
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# ---------------------------
# Upsampling and downsampling layers
# ---------------------------

# def Upsample(dim):
#     return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))
#
#
# def Downsample(dim):
#     return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

def Upsample(dim):
    """3D transposed convolution for upsampling (in time and space)."""
    return nn.ConvTranspose3d(dim, dim, (2, 4, 4), (2, 2, 2), (0, 1, 1))

def Downsample(dim):
    """3D convolution for downsampling (in time and space)."""
    return nn.Conv3d(dim, dim, (2, 4, 4), (2, 2, 2), (0, 1, 1))


# ---------------------------
# Normalization layers
# ---------------------------

class LayerNorm(nn.Module):
    """
    Custom LayerNorm for 3D feature maps.
    Normalizes across channels with learnable scale gamma.
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        # Learnable scaling parameter, one per channel
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        # Compute mean and variance across channels
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)

        # Normalize and scale
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    """
    Apply LayerNorm before passing input through a function.
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


# ---------------------------
# Convolutional blocks
# ---------------------------

class Block(nn.Module):
    """
    Basic convolutional block:
    Conv3D -> GroupNorm -> optional scale/shift -> SiLU activation.
    """
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        # Apply FiLM-like conditioning if scale/shift is given
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time embedding must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.dropout(h)
        h = self.block2(h)
        return h + self.res_conv(x)



# Spatial Linear Attention for 3D inputs (B, C, F, H, W)
# -------------------------------------------------------
# Graph theory lens:
# - Each pixel = a node
# - Attention = dynamic adjacency (who attends to whom)
# - Values (V) = messages passed
# - Queries (Q) & Keys (K) = determine edge weights
#
# Linear attention replaces the O(n^2) adjacency matrix
# with a factorized approximation, reducing cost.

class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()

        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        # Project to Q, K, V using 1x1 conv
        # - Graph view:
        #   W_Q, W_K, W_V = learn how nodes query, advertise, and send messages
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        # Merge heads back into original channel dimension
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        """
        x: [B, C, F, H, W]
        B = batch, C = channels, F = frames, H/W = spatial dims
        """
        b, c, f, h, w = x.shape

        # Treat each frame independently
        # - Graph view: each frame = its own graph
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        # Compute Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=1)

        # Rearrange into heads
        # q, k, v: [(B*F), heads, dim_head, H*W]
        q, k, v = [
            rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads)
            for t in qkv
        ]

        q = q.softmax(dim=-2)   # normalize across channels
        k = k.softmax(dim=-1)   # normalize across positions
        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)

        # Reshape back: merge heads, restore spatial layout
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)

        # Final projection back to original dim
        out = self.to_out(out)

        # Reassemble frames
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(
            tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(
            x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, rotary_emb=None):
        super().__init__()

        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb

        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x, pos_bias=None, focus_present_mask=None):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if focus_present_mask is not None and focus_present_mask.all():
            values = qkv[-1]
            return self.to_out(values)

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)

        q = q * self.scale

        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        if pos_bias is not None:
            sim = sim + pos_bias

        if focus_present_mask is not None and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)


class EMA():
    """
    Exponential Moving Average (EMA) for model parameters.
    Keeps a smoothed version of the model weights during training.

    :param beta: decay rate for EMA update (typically 0.995–0.9999).
                 Higher beta = slower updates, smoother weights.
    """

    def __init__(self, beta):
        super().__init__()
        self.beta = beta  # smoothing factor

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight = ma_params.data
            up_weight = current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new



class UNet3D(nn.Module):
    def __init__(
            self,
            dim,
            cond_dim=0,
            conditioning_mode="add_original",
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            attn_heads=8,
            attn_dim_head=32,
            init_dim=None,
            init_kernel_size=7,
            use_sparse_linear_attn=True,
            resnet_groups=8,
            use_rotary_emb=True,
            use_temporal_attention=True,
            # cross_attn parameters (only used when conditioning_mode="cross_attn")
            cross_attn_heads=4,
            cross_attn_dim_head=32,
    ):
        super().__init__()
        self._name = "MedicalDiffusionUNet3DOwn"
        self.channels = channels
        self.dim = dim
        self.cond_dim = cond_dim                          #  store for CFG zeros
        self.conditioning_mode = conditioning_mode
        self._use_temporal_attention = use_temporal_attention

        print("conditioning_mode ", conditioning_mode)

        # validate mode
        assert conditioning_mode in CONDITIONING_MODES, (
            f"conditioning_mode must be one of {CONDITIONING_MODES}, "
            f"got '{conditioning_mode}'"
        )

        rotary_emb = None
        if use_rotary_emb:
            rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        def temporal_attn(dim):
            return EinopsToAndFrom(
                'b c f h w', 'b (h w) f c',
                Attention(dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb)
            )

        # --- Initial convolution ---
        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)
        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(
            channels, init_dim, (1, init_kernel_size, init_kernel_size),
            padding=(0, init_padding, init_padding)
        )

        # --- Dimensions per level ---
        dims = [init_dim, *map(lambda m: int(dim * m), dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # --- Time embedding ---
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # --- Condition embedding ---
        # Other modes set up their own modules below.
        self.cond_mlp        = None   # used by "add_original" and "add"
        self.cond_gate       = None   # used by "add" only
        self.cond_cross_attn = None   # used by "cross_attn" only
        self.cond_concat     = None   # used by "concat" only

        mid_dim = dims[-1]

        if cond_dim > 0:
            cond_hidden  = max(64, cond_dim * 16)
            cond_emb_dim = time_dim

            if conditioning_mode == "add_original":
                # Exact original cond_mlp — no change in behaviour
                self.cond_mlp = nn.Sequential(
                    nn.Linear(cond_dim, time_dim),
                    nn.GELU(),
                    nn.Linear(time_dim, time_dim),
                )

            elif conditioning_mode == "add":
                # Hidden bottleneck prevents near-zero gradients for small cond_dim
                self.cond_mlp = nn.Sequential(
                    nn.Linear(cond_dim,    cond_hidden),
                    nn.GELU(),
                    nn.Linear(cond_hidden, cond_emb_dim),
                    nn.GELU(),
                    nn.Linear(cond_emb_dim, cond_emb_dim),
                )
                self.cond_gate = nn.Parameter(torch.ones(1))

            elif conditioning_mode == "film":
                # Same improved projection; passed to ResnetBlockFiLM blocks
                self.cond_mlp = nn.Sequential(
                    nn.Linear(cond_dim,    cond_hidden),
                    nn.GELU(),
                    nn.Linear(cond_hidden, cond_emb_dim),
                    nn.GELU(),
                    nn.Linear(cond_emb_dim, cond_emb_dim),
                )

            elif conditioning_mode == "cross_attn":
                self.cond_cross_attn = CrossAttentionConditioning(
                    spatial_dim=mid_dim,
                    cond_dim=cond_dim,
                    heads=cross_attn_heads,
                    dim_head=cross_attn_dim_head,
                )

            elif conditioning_mode == "concat":
                self.cond_concat = ConcatConditioning(
                    spatial_dim=mid_dim,
                    cond_dim=cond_dim,
                )

        # choose block factory based on mode
        _film_cond_dim = time_dim if (cond_dim > 0 and conditioning_mode == "film") else None

        if conditioning_mode == "film" and cond_dim > 0:
            def make_block(dim_in, dim_out):
                return ResnetBlockFiLM(
                    dim_in, dim_out,
                    time_emb_dim=time_dim,
                    cond_emb_dim=_film_cond_dim,
                    groups=resnet_groups,
                )
        else:
            def make_block(dim_in, dim_out):
                return ResnetBlock(
                    dim_in, dim_out,
                    time_emb_dim=time_dim,
                    groups=resnet_groups,
                )

        # --- Block definitions ---
        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=time_dim)

        # --- Encoder (downs) ---
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                make_block(dim_in, dim_out),
                make_block(dim_out, dim_out),
                Residual(PreNorm(dim_out, SpatialLinearAttention(dim_out, heads=attn_heads)))
                if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        # --- Bottleneck (mid) ---
        self.mid_block = nn.ModuleList([
            nn.ModuleList([
                make_block(mid_dim, mid_dim),
                Residual(PreNorm(mid_dim, EinopsToAndFrom(
                    'b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads)
                ))),
                Residual(PreNorm(mid_dim, temporal_attn(mid_dim))),
                make_block(mid_dim, mid_dim)
            ])
        ])

        # --- Decoder (ups) ---
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                make_block(dim_out * 2, dim_in),
                make_block(dim_in, dim_in),
                Residual(PreNorm(dim_in, SpatialLinearAttention(dim_in, heads=attn_heads)))
                if use_sparse_linear_attn else nn.Identity(),
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        # --- Final convolution ---
        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )

    # block forward dispatch
    def _block_forward(self, block, x, t_emb, cond_emb):
        """
        Dispatch block forward based on type.
        ResnetBlockFiLM receives both t_emb and cond_emb.
        ResnetBlock receives only t_emb.
        """
        if isinstance(block, ResnetBlockFiLM):
            return block(x, time_emb=t_emb, cond_emb=cond_emb)
        return block(x, time_emb=t_emb)

    def forward(self, x, time, cond=None):
        x = x.float()
        batch, device = x.shape[0], x.device

        # --- time embedding ---
        t_emb = self.time_mlp(time)

        # mode-dependent conditioning setup.
        # Always compute cond_in as zeros when cond is None (clean uncond pass).
        cond_in = cond.float() if cond is not None else torch.zeros(
            batch, self.cond_dim, device=device
        )
        cond_emb = None

        if self.cond_mlp is not None:
            cond_emb = self.cond_mlp(cond_in)

        if self.conditioning_mode == "add_original" and cond_emb is not None:
            # Original behaviour: direct addition to t_emb
            t_emb = t_emb + cond_emb

        elif self.conditioning_mode == "add" and cond_emb is not None:
            # Improved: gate-scaled addition
            t_emb = t_emb + self.cond_gate * cond_emb

        # "film":       cond_emb passed through _block_forward to every block
        # "cross_attn": injected at bottleneck below
        # "concat":     injected at bottleneck below

        # --- init conv ---
        x = self.init_conv(x)
        r = x.clone()

        # --- encoder (downs) ---
        h = []
        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = self._block_forward(block1, x, t_emb, cond_emb)
            x = self._block_forward(block2, x, t_emb, cond_emb)
            x = spatial_attn(x) + x
            x = temporal_attn(x) + x
            h.append(x)
            x = downsample(x)

        # --- bottleneck ---
        for block1, spatial_attn, temporal_attn, block2 in self.mid_block:
            x = self._block_forward(block1, x, t_emb, cond_emb)
            x = spatial_attn(x) + x
            x = temporal_attn(x) + x

            # bottleneck-only conditioning modes
            if self.conditioning_mode == "cross_attn" and self.cond_cross_attn is not None:
                x = self.cond_cross_attn(x, cond_in)
            if self.conditioning_mode == "concat" and self.cond_concat is not None:
                x = self.cond_concat(x, cond_in)

            x = self._block_forward(block2, x, t_emb, cond_emb)

        # --- decoder (ups) ---
        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = self._block_forward(block1, x, t_emb, cond_emb)
            x = self._block_forward(block2, x, t_emb, cond_emb)
            x = spatial_attn(x) + x
            x = temporal_attn(x) + x
            x = upsample(x)

        # --- output ---
        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)