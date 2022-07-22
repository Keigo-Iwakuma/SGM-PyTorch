"""Common layers for defining score networks.
"""
import math
import string
from functools import partial
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from .normalization import ConditionalInstanceNorm2dPlus