import torch
import torch.nn.functional                                      as F
from pathlib import Path

file_dir = Path(__file__).parents[1]

def angle_to_matrix(theta, ndim):
    """
    Transforms angles to orthogonal matrices.

    Takes shapes [..., Stiefel_dim] to
    [..., ndim, ndim].
    """

    Y = torch.eye(ndim).view(1, ndim, ndim).repeat(*theta.shape[:-1], 1, 1)
    d = ndim * (ndim - 1) // 2 - 1

    for i in range(ndim-1, -1, -1):
        for j in range(ndim-1, i, -1):
            Y_i = torch.cos(theta[..., d]).unsqueeze(-1) * Y[..., i, :] - torch.sin(theta[..., d]).unsqueeze(-1) * Y[..., j, :]
            Y_j = torch.sin(theta[..., d]).unsqueeze(-1) * Y[..., i, :] + torch.cos(theta[..., d]).unsqueeze(-1) * Y[..., j, :]
            Y = Y.clone()
            Y[..., i, :] = Y_i
            Y[..., j, :] = Y_j
            d -= 1

    return Y

def mix_weights(beta):  
    """
    Converts the DP's betas to the mixture weights

    Note that beta_T=1 here to terminate the weights.
    """
    beta1m_cumprod = (1 - beta).cumprod(-1)
    return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)
