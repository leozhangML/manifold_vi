from pyro.distributions.torch_transform                         import TransformModule
from pyro.distributions.torch_distribution                      import TorchDistribution
from pyro.distributions                                         import Uniform, MultivariateNormal, TransformedDistribution

import torch
import torch.nn                                                 as nn
from torch.distributions                                        import constraints
from torch.distributions.utils                                  import _sum_rightmost

import numpy                                                    as np


class UniformStiefel(TorchDistribution):
    """
    Implements uniform distribution over 
    the Givens representation from:
    https://arxiv.org/pdf/1710.09443.pdf.

    Note that the log probs will be off by 
    some constant factor.
    """

    arg_constraints = {}
    support = constraints.real_vector

    def __init__(self, T, ndim, validate_args=None):
        if not isinstance(ndim, int) or ndim < 1:
            raise ValueError(
                "ndims should be an integer > 1"
                )
        stiefel_dim = ndim * (ndim - 1) // 2
        batch_shape = torch.Size([T])
        event_shape = torch.Size([stiefel_dim])
        super().__init__(batch_shape, 
                         event_shape, 
                         validate_args=validate_args)
        self.T = T
        self.ndim = ndim
        self.stiefel_dim = stiefel_dim

    def log_prob(self, value):
        """
        value has shape [..., T, stiefel_dim] 
        and we reduce over the last dims.
        """
        ind = torch.triu_indices(self.ndim, self.ndim, offset=1)
        ind = ind[1] - ind[0] - 1
        ind = torch.broadcast_to(ind, value.shape)
        return torch.sum(torch.log(torch.abs(torch.cos(value))) * ind, 
                         dim=-1)
