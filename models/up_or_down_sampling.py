"""
    Layers used for up-sampling or down-sampling images.

    Many functions are ported from https://github.com/NVlabs/stylegan2.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from op import upfirdn2d