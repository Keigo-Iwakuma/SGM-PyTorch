"""DDPM model.

This code is the pytorch equivalent of:
https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py
"""
import torch
import torch.nn as nn
import functools

from . import utils, layers, normalization

RefineBlock = layers.RefineBlock
ResidualBlock = layers.ResidualBlock
ResnetBlockDDPM = layers.ResnetBlockDDPM
Upsample = layers.Upsample
Downsample = layers.Downsample
conv3x3 = layers.ddpm_conv3x3
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init
