# Largely taken and adapted from
# https://github.com/FirasGit/medicaldiffusion/blob/master/ddpm/diffusion.py, which was build on https://github.com/lucidrains/video-diffusion-pytorch"
import math
import torch
from torch import nn, einsum
from functools import partial
from einops import rearrange
from einops_exts import check_shape, rearrange_many
from rotary_embedding_torch import RotaryEmbedding

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
    """
    ResNet-style block with optional time embedding conditioning.
    Two Block layers + skip connection.
    If time_emb_dim is provided, generates scale/shift from time embedding.
    """
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        # If time embedding exists, project it to scale and shift
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        # 1x1 conv if input/output channels differ
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time embedding must be passed in'
            # Project time embedding to scale and shift
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        # Two convolutional blocks with optional FiLM conditioning
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        # Add residual connection
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

        # Normalize queries and keys (kernel trick)
        # - Standard attention: A = softmax(QK^T / sqrt(d))
        # - Linear attention: replace softmax(QK^T) with separable kernel
        #   q = softmax(Q, dim=features)
        #   k = softmax(K, dim=positions)
        #
        #   This allows factorization:
        #   (QK^T)V  -->  Q(K^T V)
        #   ✔ No n×n adjacency matrix ever built!
        q = q.softmax(dim=-2)   # normalize across channels
        k = k.softmax(dim=-1)   # normalize across positions
        q = q * self.scale

        # Compute context = K^T V
        # - Step 1: aggregate messages across all nodes
        # - Graph view: compress the whole graph into a "global context"
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        # Compute output = context * Q
        # - Step 2: each node pulls messages back using its query
        # - Graph view: diffusion of information guided by Q
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


# -------------------------------------------------------------------
# Standard Attention (used here for temporal dimension)
#
# Graph theory lens:
# - Nodes = frames (for a fixed pixel location over time)
# - Edges = attention weights, built dynamically from Q and K
# - Values (V) = messages each frame carries
# - Attention = message passing on a temporal graph
# -------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, rotary_emb=None):
        super().__init__()

        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb

        # Linear projections for Q, K, V
        # - Graph view:
        #   W_Q, W_K, W_V define how each frame queries,
        #   how it advertises itself, and what message it sends.
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)

        # Final projection after merging heads
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x, pos_bias=None, focus_present_mask=None):
        """
        x: [batch, sequence_length=n, dim]
        In temporal use-case:
          - batch = B * (H*W)    (each pixel trajectory is a "graph")
          - n = F                (frames)
          - dim = feature dimension
        """

        n, device = x.shape[-2], x.device

        # Project to Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # Special case: if focus_present_mask is all True,
        # skip attention and just forward V
        if focus_present_mask is not None and focus_present_mask.all():
            values = qkv[-1]
            return self.to_out(values)

        # Rearrange into multi-head form
        # q, k, v: [batch, heads, n, dim_head]
        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)

        # Scale queries (standard in dot-product attention)
        q = q * self.scale

        # Optionally rotate positions with rotary embeddings
        # - This encodes temporal order into Q and K
        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # Compute raw similarity scores
        # sim[i, j] = dot(Q_i, K_j)
        # Shape: [batch, heads, n, n]
        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # Add relative positional bias if given
        if pos_bias is not None:
            sim = sim + pos_bias

        # Apply masks if necessary
        # - e.g. only self-attend or causal masking
        if focus_present_mask is not None and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # Numerical stability: subtract row-wise max before softmax
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()

        # Normalize to get attention weights
        # A = softmax(QK^T)
        attn = sim.softmax(dim=-1)  # [batch, heads, n, n]

        # Aggregate values with adjacency A
        # out[i] = sum_j A[i,j] * V[j]
        # - Graph view:
        #   Each frame pulls messages from all others
        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)

        # Merge heads back
        out = rearrange(out, '... h n d -> ... n (h d)')

        # Final linear projection
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
        """
        Update EMA model parameters using the current model parameters.

        :param ma_model: model holding EMA-smoothed weights.
        :param current_model: model being trained (raw weights).
        """
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight = ma_params.data  # EMA (previous smoothed weight)
            up_weight = current_params.data  # new weight from current training step
            # update EMA weight
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        """
        Perform EMA update for a single parameter tensor.

        :param old: previous EMA weight.
        :param new: new model weight.
        :return: updated EMA weight.
        """
        if old is None:
            return new
        # EMA formula: ema = beta * ema + (1 - beta) * new
        return old * self.beta + (1 - self.beta) * new



class UNet3D(nn.Module):
    def __init__(
            self,
            dim,
            cond_dim=0,
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
            use_temporal_attention=True
    ):
        super().__init__()
        self._name = "MedicalDiffusionUNet3DOwn"
        self.channels = channels
        self.dim = dim
        self._use_temporal_attention = use_temporal_attention


        rotary_emb = None
        if use_rotary_emb:
            rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        # --- Temporal Attention ---
        # Note: temporal attention runs across the depth dimension (F slices).
        # - Quadratic cost is O(F^2), but F=40 is small → perfectly fine.
        # - Standard softmax preserves a true probability distribution across slices,
        #   which is important for CT scans (adjacent slices highly correlated).
        # - Factorization here would save almost nothing and reduce expressivity.
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
        if cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )
        else:
            self.cond_mlp = None

        # --- Block definitions ---
        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=time_dim)

        # --- Encoder (downs) ---
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                block_klass_cond(dim_out, dim_out),

                # Spatial attention: FACTORIZED version (linear attention)
                # - Complexity of full spatial attention: O((H*W)^2)
                # - For H=24, W=40 → 960^2 ≈ 921,600, far too expensive.
                # - Factorization reduces cost to O(H*W), making it feasible.
                Residual(PreNorm(dim_out, SpatialLinearAttention(dim_out, heads=attn_heads)))
                if use_sparse_linear_attn else nn.Identity(),

                # Temporal attention: keep SOFTMAX version (see reasoning above)
                Residual(PreNorm(dim_out, temporal_attn(dim_out))),

                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        # --- Bottleneck (mid) ---
        mid_dim = dims[-1]
        self.mid_block = nn.ModuleList([
            nn.ModuleList([
                block_klass_cond(mid_dim, mid_dim),

                # Optional spatial attention at bottleneck
                Residual(PreNorm(mid_dim, EinopsToAndFrom(
                    'b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads)
                ))),

                # Temporal softmax attention again
                Residual(PreNorm(mid_dim, temporal_attn(mid_dim))),

                block_klass_cond(mid_dim, mid_dim)
            ])
        ])

        # --- Decoder (ups) ---
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                block_klass_cond(dim_in, dim_in),

                # Factorized spatial attention again
                Residual(PreNorm(dim_in, SpatialLinearAttention(dim_in, heads=attn_heads)))
                if use_sparse_linear_attn else nn.Identity(),

                # Standard temporal attention
                Residual(PreNorm(dim_in, temporal_attn(dim_in))),

                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        # --- Final convolution ---
        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),
            nn.Conv3d(dim, out_dim, 1)
        )

    def forward(self, x, time, cond=None):
        """
        Forward pass of UNet3D for denoising diffusion.

        Input:
        - x: (batch, C, D, H, W) noised input volume
        - time: (batch,) diffusion timesteps
        - cond: (batch, cond_dim) conditioning vector (e.g. class embedding)

        Workflow:
        1. Encode with downsampling, using ResNet blocks + spatial & temporal attention.
           - Spatial attention is factorized (linear) → handles large H×W efficiently.
           - Temporal attention is softmax → preserves strong slice-to-slice correlations.
        2. Bottleneck with attention.
        3. Decode with upsampling, again applying both attentions.
        4. Final conv projects back to output channels.
        """
        x = x.float()
        batch, device = x.shape[0], x.device

        # --- time embedding ---
        t_emb = self.time_mlp(time)

        # --- condition embedding ---
        if self.cond_mlp is not None:
            if cond is None:
                # Produce zero embedding matching t_emb dimension
                cond_emb = torch.zeros(batch, t_emb.shape[-1], device=device)
            else:
                cond_emb = self.cond_mlp(cond)
            t_emb = t_emb + cond_emb

        # --- init conv ---
        x = self.init_conv(x)
        r = x.clone()

        # --- encoder (downs) ---
        h = []
        for block1, block2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t_emb)
            x = block2(x, t_emb)
            x = spatial_attn(x) + x
            x = temporal_attn(x) + x
            h.append(x)
            x = downsample(x)

        # --- bottleneck ---
        for block1, spatial_attn, temporal_attn, block2 in self.mid_block:
            x = block1(x, t_emb)
            x = spatial_attn(x) + x
            x = temporal_attn(x) + x
            x = block2(x, t_emb)

        # --- decoder (ups) ---
        for block1, block2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t_emb)
            x = block2(x, t_emb)
            x = spatial_attn(x) + x
            x = temporal_attn(x) + x
            x = upsample(x)

        # --- output ---
        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)

