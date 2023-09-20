import numpy as np
import torch
from torch.nn import functional as F

from autoregressive import Autoregressive
from made import MADE
import splines
from nn import PeriodicFeaturesElementwise


class MaskedPiecewiseRationalQuadraticAutoregressive(Autoregressive):

    def __init__(
        self,
        features,
        hidden_features,
        context_features,
        num_bins=10,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        permute_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        init_identity=True,
        min_bin_width=splines.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=splines.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=splines.DEFAULT_MIN_DERIVATIVE,
    ):
        # no tails or tail_bounds

        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        scale_pf = 1.
        ind_circ = [i for i in range(features)]
        preprocessing = PeriodicFeaturesElementwise(features, ind_circ, scale_pf)
        tail_bound = np.pi

        # allows batching
        autoregressive_net = MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            permute_mask=permute_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
            preprocessing=preprocessing,
        )

        if init_identity:
            torch.nn.init.constant_(autoregressive_net.final_layer.weight, 0.0)
            torch.nn.init.constant_(
                autoregressive_net.final_layer.bias,
                np.log(np.exp(1 - min_derivative) - 1),
            )

        super().__init__(autoregressive_net)

        self.tail_bound = tail_bound
        self.context_features = context_features
        self.context = torch.eye(context_features)  # for context set at T with one hot encoding

    def _output_dim_multiplier(self):
        return self.num_bins * 3

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size, features = inputs.shape[:-1], inputs.shape[-1] 
        transform_params = autoregressive_params.view(
            *batch_size, features, self._output_dim_multiplier()
        )
        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        if hasattr(self.autoregressive_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.autoregressive_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.autoregressive_net.hidden_features)
        spline_fn = splines.unconstrained_rational_quadratic_spline  # uses this
        spline_kwargs = {"tail_bound": self.tail_bound}

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs 
        )

        return outputs, torch.sum(logabsdet, dim=-1)

    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)
