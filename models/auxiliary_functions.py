import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def extract(a, t, x_shape):
    """
    Extract values from tensor `a` at indices `t` and reshape to match `x_shape`.

    :param a: source tensor to gather from.
    :param t: indices to extract.
    :param x_shape: target shape for broadcasting.
    :return: extracted values reshaped to match `x_shape`.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def forward_latent_transform(h, vqgan):
    """
    Normalize VQGAN encoder outputs to [-1, 1].

    :param h: VQGAN encoder output tensor.
    :param vqgan: VQGAN model with quantize.embedding.weight.
    :return: normalized latent representation.
    """
    w_min = vqgan.quantize.embedding.weight.min().item()
    w_max = vqgan.quantize.embedding.weight.max().item()
    return ((h - w_min) / (w_max - w_min)) * 2.0 - 1.0


def inverse_latent_transform(latents, vqgan):
    """
    Inverse of forward_latent_transform: maps normalized latents back to VQGAN space.

    :param latents: normalized latent representation.
    :param vqgan: VQGAN model with quantize.embedding.weight.
    :return: original latent representation.
    """
<<<<<<< HEAD
=======
    print("latents.shape ", latents.shape)
>>>>>>> e7f76be51110a205240956a772fda3f2da904176
    w_min = vqgan.quantize.embedding.weight.min().item()
    w_max = vqgan.quantize.embedding.weight.max().item()
    return ((latents + 1.0) / 2.0) * (w_max - w_min) + w_min


def reshape_to_tensors(tn_array, dim=2):
    """
    Convert an array of upper-triangular values into a symmetric tensor.

    :param tn_array: values to fill in the upper triangular part of the matrix.
    :param dim: dimension of the target tensor (default=2).
    :return: symmetric tensor of shape (dim, dim).
    """
    tn = np.eye(dim)
    tn[np.triu_indices(dim)] = tn_array

    diagonal_values = np.diag(tn)
    symmetric_tn = tn + tn.T
    np.fill_diagonal(symmetric_tn, diagonal_values)
    return symmetric_tn


class CosineSimilarity(nn.Module):
    """
    Cosine similarity loss = 1 - cosine_similarity(y_pred, y_true).
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        """
        :param y_pred: predicted tensor.
        :param y_true: target tensor.
        :return: cosine similarity loss.
        """
        criterion = nn.CosineSimilarity()
        return 1 - criterion(y_pred, y_true)


class WeightedMSELoss(nn.Module):
    """
    Weighted mean squared error loss.
    """
    def __init__(self, weights):
        """
        :param weights: list or tensor of per-element weights.
        """
        super().__init__()
        self.weights = torch.Tensor(weights)
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()

    def forward(self, y_pred, y_true):
        """
        :param y_pred: predicted tensor.
        :param y_true: target tensor.
        :return: weighted MSE loss (scalar).
        """
        mse_loss = F.mse_loss(y_pred, y_true, reduction='none')

        # Ensure weights are on the same device
        if mse_loss.device.type == "cpu":
            self.weights = self.weights.cpu()

        weighted_mse_loss = torch.mean(self.weights * mse_loss)
        return weighted_mse_loss


class WeightedMSELossSum(nn.Module):
    """
    Weighted mean squared error loss, with summation across dimensions.
    """
    def __init__(self, weights):
        """
        :param weights: list or tensor of per-element weights.
        """
        super().__init__()
        self.weights = torch.Tensor(weights)
        if torch.cuda.is_available():
            self.weights = self.weights.cuda()

    def forward(self, y_pred, y_true):
        """
        :param y_pred: predicted tensor.
        :param y_true: target tensor.
        :return: weighted MSE loss (scalar).
        """
        mse_loss = F.mse_loss(y_pred, y_true, reduction='none')

        # Ensure weights are on the same device
        if mse_loss.device.type == "cpu":
            self.weights = self.weights.cpu()

        # Apply weights, summing if loss has >1 dimension
        if len(mse_loss.shape) == 1:
            mse_loss_sum = self.weights * mse_loss
        else:
            mse_loss_sum = torch.sum(self.weights * mse_loss, dim=1)

        weighted_mse_loss = torch.mean(mse_loss_sum)
        return weighted_mse_loss


class LossX(nn.Module):
    """
    Cross-entropy loss for node attribute prediction.
    """
    def __init__(self, num_attrs_X):
        """
        :param num_attrs_X: number of node attributes.
        """
        super().__init__()
        self.num_attrs_X = num_attrs_X

    def forward(self, true_X, logit_X):
        """
        :param true_X: ground truth (F, |V|, 2), one-hot encoding of node attributes.
        :param logit_X: predicted logits (|V|, F, 2).
        :return: cross-entropy loss (scalar).
        """
        true_X = true_X.transpose(0, 1)               # (|V|, F, 2)
        true_X = true_X.reshape(-1, true_X.size(-1))  # (|V| * F, 2)
        logit_X = logit_X.reshape(true_X.size(0), -1) # (|V| * F, 2)

        true_X = torch.argmax(true_X, dim=-1)         # (|V| * F)
        return F.cross_entropy(logit_X, true_X)


class KLLoss(nn.Module):
    """
    Wrapper for cross-entropy loss (used as KL divergence surrogate).
    """
    def __init__(self):
        super().__init__()

    def forward(self, true_G, logit_G):
        """
        :param true_G: ground truth labels.
        :param logit_G: predicted logits.
        :return: cross-entropy loss.
        """
        return F.cross_entropy(logit_G, true_G)


def get_loss_fn(loss_function):
    """
    Factory method for retrieving loss functions by name.

    :param loss_function: tuple (loss_name, params).
    :return: instantiated loss function module.
    """
    loss_fn_name, loss_fn_params = loss_function

    if loss_fn_name in ("MSE", "L2"):
        return nn.MSELoss()
    elif loss_fn_name == "L1":
        return nn.L1Loss()
    elif loss_fn_name == "MSEweighted":
        return WeightedMSELoss(loss_fn_params)
    elif loss_fn_name == "MSEweightedSum":
        return WeightedMSELossSum(loss_fn_params)
    elif loss_fn_name in ("KL", "cross_entropy"):
        return KLLoss()
    elif loss_fn_name == "CrossEntropy":
        return nn.CrossEntropyLoss()
