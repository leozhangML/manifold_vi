import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn


class Autoregressive(nn.Module):
    """Transforms each input variable with an invertible elementwise transformation.

    The parameters of each invertible elementwise transformation can be functions of previous input
    variables, but they must not depend on the current or any following input variables.

    **NOTE** Calculating the inverse transform is D times slower than calculating the
    forward transform, where D is the dimensionality of the input to the transform.
    """

    def __init__(self, autoregressive_net):
        super(Autoregressive, self).__init__()
        self.autoregressive_net = autoregressive_net

    def forward(self, inputs):  # [..., T, sdim]
        context = torch.broadcast_to(self.context, [*inputs.shape[:-1], self.context.shape[-1]])  # [T, M] to [..., T, M]
        #context = None
        autoregressive_params = self.autoregressive_net(inputs, context)  # inputs shape: [..., N]
        outputs, logabsdet = self._elementwise_forward(inputs, autoregressive_params)  # logabsdet shape: [...]
        return outputs, logabsdet

    def inverse(self, inputs):
        #num_inputs = np.prod(inputs.shape[1:])  # should change to inputs.shape[-1] previous was due to image shapes
        context = torch.broadcast_to(self.context, [*inputs.shape[:-1], self.context.shape[-1]])
        #context = None
        num_inputs = inputs.shape[-1]
        outputs = torch.zeros_like(inputs)
        logabsdet = None
        for _ in range(num_inputs):
            autoregressive_params = self.autoregressive_net(outputs, context)  # check inverse
            outputs, logabsdet = self._elementwise_inverse(
                inputs, autoregressive_params
            )
        return outputs, logabsdet

    def _output_dim_multiplier(self):
        raise NotImplementedError()

    def _elementwise_forward(self, inputs, autoregressive_params):
        raise NotImplementedError()

    def _elementwise_inverse(self, inputs, autoregressive_params):
        raise NotImplementedError()

