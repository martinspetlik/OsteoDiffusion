import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


class EmbeddingEMA(nn.Module):
    """
    Class implementing an Exponential Moving Average (EMA) codebook used for vector quantization.
    Maintains running averages of cluster centers for stable updates.
    """
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5):
        """
        Initialize the EMA embedding codebook.
        :param num_tokens: Number of embeddings in the codebook.
        :param codebook_dim: Dimensionality of each embedding vector.
        :param decay: EMA decay rate (default 0.99).
        :param eps: Small epsilon to avoid division by zero (default 1e-5).
        """
        super().__init__()
        self.decay = decay
        self.eps = eps
        weight = torch.randn(num_tokens, codebook_dim)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)

    def forward(self, embed_id):
        """
        Forward pass to return embeddings for given indices.
        :param embed_id: Tensor of embedding indices.
        :return: Corresponding embedding vectors.
        """
        return F.embedding(embed_id, self.weight)

    def update(self, embed_onehot, flat_inputs):
        """
        Update the embedding weights using exponential moving averages.
        :param embed_onehot: One-hot encoded tensor of shape (N, num_tokens).
        :param flat_inputs: Flattened input tensor of shape (N, codebook_dim).
        :return: None
        """
        new_cluster_size = embed_onehot.sum(0)
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

        embed_sum = torch.matmul(embed_onehot.t(), flat_inputs)
        self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

        n = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.eps) / (n + self.cluster_size.shape[0] * self.eps) * n
        normalized_embed = self.embed_avg / smoothed.unsqueeze(1)
        self.weight.data.copy_(normalized_embed)


class EMAVectorQuantizer(nn.Module):
    """
    Vector quantizer with Exponential Moving Average (EMA) codebook updates.
    Supports index remapping and codebook usage tracking.
    """
    def __init__(self, num_embed, dim_embed, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, decay=0.99, eps=1e-5):
        """
        Initialize the EMA vector quantizer.
        :param num_embed: Number of codebook embeddings.
        :param dim_embed: Dimensionality of embeddings.
        :param beta: Weight for commitment loss.
        :param remap: Optional path to .npy file with subset of used indices.
        :param unknown_index: How to handle unknown indices ('random' or 'extra').
        :param sane_index_shape: If True, reshape indices to match input dimensions.
        :param decay: EMA decay rate.
        :param eps: Small epsilon for stability.
        """
        super().__init__()
        self.num_embed = num_embed
        self.dim_embed = dim_embed
        self.beta = beta
        self.unknown_index = unknown_index
        self.sane_index_shape = sane_index_shape

        self.embedding = EmbeddingEMA(num_embed, dim_embed, decay=decay, eps=eps)

        self.remap = remap
        if self.remap is not None:
            self.set_used_indices(self.remap)
        else:
            self.re_embed = num_embed

        self.register_buffer("index_usage_counts", torch.zeros(num_embed, dtype=torch.long))

    def set_used_indices(self, used_indices_files):
        """
        Set allowed subset of embedding indices from file.
        :param used_indices_files: Path to .npy file containing valid indices.
        :return: None
        """
        self.remap = used_indices_files
        self.register_buffer("used", torch.tensor(np.load(self.remap)))
        self.re_embed = self.used.shape[0]
        if self.unknown_index == "extra":
            self.unknown_index = self.re_embed
            self.re_embed += 1

    def remap_to_used(self, inds):
        """
        Remap raw indices to the known subset of used indices.
        :param inds: Tensor of embedding indices.
        :return: Remapped indices tensor.
        """
        ishape = inds.shape
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds.device)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        """
        Reverse remapping back to original embedding indices.
        :param inds: Remapped indices tensor.
        :return: Original embedding indices tensor.
        """
        ishape = inds.shape
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds.device)
        if self.re_embed > self.used.shape[0]:
            inds[inds >= self.used.shape[0]] = 0
        back = torch.gather(used[None, :].expand(inds.shape[0], -1), 1, inds)
        return back.reshape(ishape)

    def forward(self, z):
        """
        Quantize input tensor and update EMA codebook.
        :param z: Input tensor of shape (B, C, H, W, D).
        :return: Tuple (quantized tensor, loss, (None, None, indices)).
        """
        z = rearrange(z, 'b c h w d -> b h w d c').contiguous()
        flat_input = z.view(-1, self.dim_embed)

        weight = self.embedding.weight
        d = (flat_input ** 2).sum(dim=1, keepdim=True) + (weight ** 2).sum(dim=1) - 2 * torch.matmul(flat_input, weight.t())

        encoding_indices = torch.argmin(d, dim=1)

        with torch.no_grad():
            unique, counts = encoding_indices.unique(return_counts=True)
            self.index_usage_counts.index_add_(0, unique, counts)

        z_q = self.embedding(encoding_indices).view(z.shape)

        one_hot = F.one_hot(encoding_indices, self.num_embed).type(flat_input.dtype)
        self.embedding.update(one_hot, flat_input)

        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, 'b h w d c -> b c h w d').contiguous()

        if self.remap is not None:
            encoding_indices = encoding_indices.reshape(z.shape[0], -1)
            encoding_indices = self.remap_to_used(encoding_indices)
            encoding_indices = encoding_indices.reshape(-1, 1)

        if self.sane_index_shape:
            encoding_indices = encoding_indices.reshape(z_q.shape[0], z_q.shape[-3], z_q.shape[-2], z_q.shape[-1])

        return z_q, loss, (None, None, encoding_indices)

    def get_codebook_entry(self, indices, shape):
        """
        Retrieve embedding vectors corresponding to given indices.
        :param indices: Tensor of embedding indices.
        :param shape: Desired output tensor shape.
        :return: Tensor of quantized embeddings.
        """
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(*shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

    def reset_index_usage(self):
        """
        Reset tracked codebook usage statistics.
        :return: None
        """
        self.index_usage_counts.zero_()


class VectorQuantizer(nn.Module):
    """
    Standard vector quantizer module (non-EMA) similar to the one used in VQ-VAE.
    """
    def __init__(self, num_embed, dim_embed, beta, remap=None, unknown_index="random",
                 sane_index_shape=False):
        """
        Initialize the vector quantizer module.
        :param num_embed: Number of embeddings.
        :param dim_embed: Embedding dimensionality.
        :param beta: Weight for commitment loss term.
        :param remap: Optional path to file defining used subset of indices.
        :param unknown_index: Strategy for unknown indices ('random', 'extra', or int).
        :param sane_index_shape: If True, reshape indices to match spatial dimensions.
        """
        super().__init__()
        self.num_embed = num_embed
        self.dim_embed = dim_embed
        self.beta = beta
        self.unknown_index = unknown_index

        self.embedding = nn.Embedding(self.num_embed, self.dim_embed)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embed, 1.0 / self.num_embed)

        self.remap = remap
        if self.remap is not None:
            self.set_used_indices(self.remap)
        else:
            self.re_embed = num_embed

        self.sane_index_shape = sane_index_shape

    def set_used_indices(self, used_indices_files):
        """
        Load subset of valid embedding indices for remapping.
        :param used_indices_files: Path to .npy file with valid indices.
        :return: None
        """
        self.remap = used_indices_files
        self.register_buffer("used", torch.tensor(np.load(self.remap)))
        self.re_embed = self.used.shape[0]

        if self.unknown_index == "extra":
            self.unknown_index = self.re_embed
            self.re_embed += 1

        print(f"Remapping {self.num_embed} indices to {self.re_embed}. Unknown index policy: {self.unknown_index}")

    def remap_to_used(self, inds):
        """
        Remap raw embedding indices to used subset.
        :param inds: Input tensor of indices.
        :return: Remapped tensor of indices.
        """
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds.device)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        """
        Reverse the remapping from used subset to full index space.
        :param inds: Remapped indices tensor.
        :return: Original embedding indices tensor.
        """
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds.device)
        if self.re_embed > self.used.shape[0]:
            inds[inds >= self.used.shape[0]] = 0
        back = torch.gather(used[None, :].expand(inds.shape[0], -1), 1, inds)
        return back.reshape(ishape)

    def forward(self, z):
        """
        Quantize input tensor using nearest-neighbor lookup.
        :param z: Input tensor of shape (B, C, H, W, D).
        :return: Tuple (quantized tensor, loss, (None, None, indices)).
        """
        z = rearrange(z, 'b c h w d -> b h w d c').contiguous()
        z_flattened = z.view(-1, self.dim_embed)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)

        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, 'b h w d c -> b c h w d').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[-3], z_q.shape[-2], z_q.shape[-1])

        return z_q, loss, (None, None, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        """
        Retrieve embeddings based on indices.
        :param indices: Tensor of embedding indices.
        :param shape: Desired output shape.
        :return: Quantized tensor with given shape.
        """
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)

        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q
