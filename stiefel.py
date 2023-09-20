from pyro.distributions.torch_transform                         import TransformModule
from pyro.distributions.torch_distribution                      import TorchDistribution

import torch
import torch.nn                                                 as nn
from torch.distributions                                        import constraints

from mprqat                                                     import MaskedPiecewiseRationalQuadraticAutoregressive


class Circular(TransformModule):
        """
        Base will be:
        Uniform(torch.ones([self.T, self.stiefel_dim]) * -np.pi,
                torch.ones([self.T, self.stiefel_dim]) * np.pi).to_event(1)
        hence never a scalar.
        """

        domain = constraints.real_vector
        codomain = constraints.real_vector
        bijective = True
        sign = +1
        autoregressive = True

        def __init__(
            self,
            num_input_channels,
            context_features,
            num_blocks,
            num_hidden_channels,
            num_bins=8,
            activation=nn.ReLU,
            dropout_probability=0.0,
            permute_mask=True,
            init_identity=True,
            **kwargs
        ): 
            super().__init__(cache_size=1)

            self.mprqat = MaskedPiecewiseRationalQuadraticAutoregressive(
                features=num_input_channels,
                hidden_features=num_hidden_channels,
                context_features=context_features,
                num_bins=num_bins,
                num_blocks=num_blocks,
                use_residual_blocks=True,
                random_mask=False,
                permute_mask=permute_mask,
                activation=activation(),
                dropout_probability=dropout_probability,
                use_batch_norm=False,
                init_identity=init_identity,
            )

            # different shape conventions for 1D flows
            if num_input_channels == 1:
                self.prob_shape = [-1, 1]
            else:
                self.prob_shape = [1, -1] 

            self.context_features = context_features

        def _call(self, x):

            x_shape = x.shape
            if x.dim() < 2:
                x = x.view(*self.prob_shape)

            x, log_detJ = self.mprqat(x)
            self._cache_log_detJ = log_detJ  
            return x.view(x_shape)

        def _inverse(self, y):

            y_shape = y.shape
            if y.dim() < 2:
                y = y.view(*self.prob_shape)

            y, log_detJ = self.mprqat.inverse(y) 
            self._cache_log_detJ = -log_detJ
            return y.view(y_shape)

        def log_abs_det_jacobian(self, x, y):

            x_old, y_old = self._cached_x_y
            if x is not x_old or y is not y_old:
                self._call(x)

            if x.dim() < 2:
                return self._cache_log_detJ.view(torch.Size([]))
            else:
                return self._cache_log_detJ
