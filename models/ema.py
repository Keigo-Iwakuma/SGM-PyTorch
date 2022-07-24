# Modified from https://raw.githubusercontent.com/fadel/pytorch_ema/master/torch_ema/ema.py

from __future__ import division
from __future__ import unicode_literals

import torch


# Partially based on : https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/training/moving_averages.py
class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.
    """
