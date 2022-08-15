"""
    Layers used for up-sampling or down-sampling images.

    Many functions are ported from https://github.com/NVlabs/stylegan2.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from op import upfirdn2d


# Function ported from StyleGAN2
def get_weight(module, shape, weight_var="weight", kernel_init=None):
    """Get/create weight tensor for a convolution or fully-connected layer."""
    return module.param(weight_var, kernel_init, shape)


class Conv2d(nn.Module):
    """Conv2d layer with optimal upsampling and downsampling (StyleGAN2)."""

    def _init__(
        self,
        in_ch,
        out_ch,
        kernel,
        up=False,
        down=False,
        resample_kernel=(1, 3, 3, 1),
        use_bias=True,
        kernel_init=None,
    ):
        super().__init__()
        assert not (up and down)
        assert kernel >= 1 and kernel % 2 == 1
        self.weight = nn.Parameter(torch.zeros(out_ch, in_ch, kernel, kernel))
        if kernel_init is not None:
            self.weight.data = kernel_init(self.weight.data.shape)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_ch))

        self.up = up
        self.down = down
        self.resample_kernel = resample_kernel
        self.kernel = kernel
        self.use_bias = use_bias
