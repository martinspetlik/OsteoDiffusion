import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange


class VectorQuantizer(nn.Module):
    """
    Customized VectorQuantizer based on VectorQuantizer2:
    https://github.com/jongdory/ALDM/blob/main/VQ-GAN/taming/modules/vqvae/quantize.py
    """

    def __init__(self, num_embed, dim_embed, beta, remap=None, unknown_index="random",
                 sane_index_shape=False):
        super().__init__()
        self.num_embed = num_embed          # Number of codebook vectors
        self.dim_embed = dim_embed          # Dimensionality of each embedding vector
        self.beta = beta                    # Weight for commitment loss term
        self.unknown_index = unknown_index

        # Initialize embedding (codebook) with uniform values
        self.embedding = nn.Embedding(self.num_embed, self.dim_embed)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embed, 1.0 / self.num_embed)

        self.remap = remap
        if self.remap is not None:
            self.set_used_indices(self.remap)
        else:
            self.re_embed = num_embed

        self.sane_index_shape = sane_index_shape  # Option to reshape indices for later processing

    def set_used_indices(self, used_indices_files):
        self.remap = used_indices_files

        # Load a pre-specified set of used indices for remapping
        self.register_buffer("used", torch.tensor(np.load(self.remap)))
        self.re_embed = self.used.shape[0]
        self.unknown_index = self.unknown_index  # "random", "extra", or a specific index

        if self.unknown_index == "extra":
            # Add an extra token for unknown indices
            self.unknown_index = self.re_embed
            self.re_embed += 1

        print(f"Remapping {self.num_embed} indices to {self.re_embed} indices. "
              f"Using {self.unknown_index} for unknown indices.")


    def remap_to_used(self, inds):
        # Map raw indices to a known subset of indices (used embeddings)
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds.device)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)

        # Mark unknowns
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index

        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        # Map reduced embedding indices back to original embedding indices
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)

        used = self.used.to(inds.device)

        if self.re_embed > self.used.shape[0]:  # extra token exists
            inds[inds >= self.used.shape[0]] = 0  # default to 0 for invalid

        # Reverse remapping via gather
        back = torch.gather(used[None, :].expand(inds.shape[0], -1), 1, inds)
        return back.reshape(ishape)

    def forward(self, z):
        # Rearrange z to (B, H, W, D, C) for distance computation
        z = rearrange(z, 'b c h w d -> b h w d c').contiguous()
        z_flattened = z.view(-1, self.dim_embed)  # Flatten for dot-product

        # Compute L2 distance between input and codebook vectors
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        # Get nearest codebook index for each vector
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # Optional outputs (not used here)
        perplexity = None
        min_encodings = None

        # Compute VQ-VAE loss:
        # (1) commitment loss: ||z_q.detach() - z||^2
        # (2) codebook loss: ||z_q - z.detach()||^2
        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
               torch.mean((z_q - z.detach()) ** 2)

        # Straight-through estimator to preserve gradients
        z_q = z + (z_q - z).detach()

        # Restore original shape (B, C, H, W, D)
        z_q = rearrange(z_q, 'b h w d c -> b c h w d').contiguous()

        print("z_q.shape ", z_q.shape)

        # Optionally remap indices to used subset
        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[-3], z_q.shape[-2], z_q.shape[-1])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # Retrieve codebook embeddings based on indices
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)

        # Lookup embeddings
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q