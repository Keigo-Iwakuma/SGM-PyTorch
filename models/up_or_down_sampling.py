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