import torch
from torch import nn


class PeriodicFeaturesElementwise(nn.Module):
    """
    Converts a specified part of the input to periodic features by
    replacing those features f with
    w1 * sin(scale * f) + w2 * cos(scale * f).

    Note that this operation is done elementwise and, therefore,
    some information about the feature can be lost.
    """

    def __init__(self, ndim, ind, scale=1.0, bias=False, activation=None):
        """Constructor

        Args:
          ndim (int): number of dimensions
          ind (iterable): indices of input elements to convert to periodic features
          scale: Scalar or iterable, used to scale inputs before converting them to periodic features
          bias: Flag, whether to add a bias
          activation: Function or None, activation function to be applied

        PeriodicFeaturesElementwise(features, ind_circ, scale_pf)
        scale_pf = np.pi / tail_bound[ind_circ]  hence 1 for circular
        self.preprocessing(inputs)  [..., N]
        """
        super(PeriodicFeaturesElementwise, self).__init__()

        # Set up indices and permutations
        self.ndim = ndim
        if torch.is_tensor(ind):
            self.register_buffer("ind", torch._cast_Long(ind))  # ind_circ
        else:
            self.register_buffer("ind", torch.tensor(ind, dtype=torch.long))

        ind_ = []  # non-circular indices
        for i in range(self.ndim):
            if not i in self.ind:
                ind_ += [i]
        self.register_buffer("ind_", torch.tensor(ind_, dtype=torch.long))

        perm_ = torch.cat((self.ind, self.ind_))
        inv_perm_ = torch.zeros_like(perm_)  # reverse permutation to restore orderings
        for i in range(self.ndim):
            inv_perm_[perm_[i]] = i
        self.register_buffer("inv_perm", inv_perm_)

        self.weights = nn.Parameter(torch.ones(len(self.ind), 2))  # [N, 2]
        if torch.is_tensor(scale):
            self.register_buffer("scale", scale)  # scale_pf
        else:
            self.scale = scale

        self.apply_bias = bias
        if self.apply_bias:
            self.bias = nn.Parameter(torch.zeros(len(self.ind)))

        if activation is None:
            self.activation = lambda input: input
        else:
            self.activation = activation

    def forward(self, inputs):
        inputs_ = inputs[..., self.ind]
        inputs_ = self.scale * inputs_
        inputs_ = self.weights[:, 0] * torch.sin(inputs_) + self.weights[
            :, 1
        ] * torch.cos(inputs_)
        if self.apply_bias:
            inputs_ = inputs_ + self.bias
        inputs_ = self.activation(inputs_)
        out = torch.cat((inputs_, inputs[..., self.ind_]), -1)  # batching does work
        return out[..., self.inv_perm]

