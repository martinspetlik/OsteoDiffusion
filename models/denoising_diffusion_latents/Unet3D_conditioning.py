"""
Extra conditioning modules for UNet3D.
Imported by Unet3D.py — do not modify Unet3D.py beyond the minimal diff.

Five conditioning modes
-----------------------
"add_original"  — original cond_mlp, added to t_emb directly (no changes to blocks)
"add"           — improved MLP with hidden bottleneck + learnable scalar gate
"film"          — FiLM scale/shift injected into every ResNet block (ResnetBlockFiLM)
"cross_attn"    — cross-attention at the bottleneck (CrossAttentionConditioning)
"concat"        — spatial broadcast + 1x1 conv at the bottleneck (ConcatConditioning)
"""

import torch
import torch.nn as nn
from einops import rearrange


CONDITIONING_MODES = ("add_original", "add", "film", "cross_attn", "concat")


# =============================================================================
# ResnetBlockFiLM
# =============================================================================

class ResnetBlockFiLM(nn.Module):
    """
    ResNet block conditioned on BOTH a time embedding AND a FiLM conditioning
    embedding applied separately:
      - time  -> scale/shift in block1  (denoising dynamics)
      - cond  -> scale/shift in block2  (biological signal)

    Keeping them separate prevents the two signals from competing.
    Used only when conditioning_mode="film".

    Drop-in replacement for ResnetBlock when film mode is active.
    Accepts the same (x, time_emb) call signature PLUS an optional cond_emb.
    """

    def __init__(self, dim, dim_out, *, time_emb_dim=None, cond_emb_dim=None,
                 groups=8, dropout=0.0):
        super().__init__()

        # Reuse Block from Unet3D — imported at usage site to avoid circular import
        from models.denoising_diffusion_latents.Unet3D import Block

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2),
        ) if time_emb_dim is not None else None

        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_emb_dim, dim_out * 2),
        ) if cond_emb_dim is not None else None

        self.block1   = Block(dim,     dim_out, groups=groups)
        self.block2   = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.dropout  = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, time_emb=None, cond_emb=None):
        time_scale_shift = None
        if self.time_mlp is not None and time_emb is not None:
            te = self.time_mlp(time_emb)
            te = rearrange(te, 'b c -> b c 1 1 1')
            time_scale_shift = te.chunk(2, dim=1)

        cond_scale_shift = None
        if self.cond_mlp is not None and cond_emb is not None:
            ce = self.cond_mlp(cond_emb)
            ce = rearrange(ce, 'b c -> b c 1 1 1')
            cond_scale_shift = ce.chunk(2, dim=1)

        h = self.block1(x,  scale_shift=time_scale_shift)
        h = self.dropout(h)
        h = self.block2(h,  scale_shift=cond_scale_shift)
        return h + self.res_conv(x)


# =============================================================================
# CrossAttentionConditioning
# =============================================================================

class CrossAttentionConditioning(nn.Module):
    """
    Cross-attention block inserted at the UNet bottleneck.

    Spatial features (flattened over D*H*W) attend to a single conditioning
    token. Every spatial position gets direct access to the biological signal.

    Unconditional pass: zero cond -> near-zero K/V -> residual preserves x.
    """

    def __init__(self, spatial_dim, cond_dim, heads=4, dim_head=32):
        super().__init__()
        hidden_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(spatial_dim, hidden_dim, bias=False)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.to_kv  = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, spatial_dim),
            nn.LayerNorm(spatial_dim),
        )
        self.norm_x    = nn.LayerNorm(spatial_dim)
        self.norm_cond = nn.LayerNorm(hidden_dim)

    def forward(self, x, cond):
        b, c, d, h, w = x.shape

        x_flat   = rearrange(x, 'b c d h w -> b (d h w) c')
        x_normed = self.norm_x(x_flat)

        cond_tok = self.cond_proj(cond).unsqueeze(1)
        cond_tok = self.norm_cond(cond_tok)

        q    = self.to_q(x_normed)
        k, v = self.to_kv(cond_tok).chunk(2, dim=-1)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        sim  = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out  = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out  = rearrange(out, 'b h n d -> b n (h d)')
        out  = self.to_out(out)

        x_flat = x_flat + out
        return rearrange(x_flat, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)


# =============================================================================
# ConcatConditioning
# =============================================================================

class ConcatConditioning(nn.Module):
    """
    Spatially broadcast the conditioning vector, concatenate to the bottleneck
    feature map channels, then fuse with a 1x1x1 conv.

    Most explicit method: conditioning values are present at every spatial
    position, making them impossible for the model to ignore.

    Unconditional pass: zero cond -> zero extra channels -> handled gracefully.
    """

    def __init__(self, spatial_dim, cond_dim):
        super().__init__()
        proj_dim  = max(spatial_dim // 4, cond_dim)
        self.proj = nn.Sequential(
            nn.Linear(cond_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.fuse = nn.Conv3d(spatial_dim + proj_dim, spatial_dim, 1)
        self.norm = nn.GroupNorm(8, spatial_dim)
        self.act  = nn.SiLU()

    def forward(self, x, cond):
        b, c, d, h, w = x.shape
        cond_feat = self.proj(cond)
        cond_feat = cond_feat[:, :, None, None, None].expand(b, -1, d, h, w)
        x_cat = torch.cat([x, cond_feat], dim=1)
        return self.act(self.norm(self.fuse(x_cat)))