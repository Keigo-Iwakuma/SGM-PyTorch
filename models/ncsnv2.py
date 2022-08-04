""""The NCSNv2 model."""
import torch
import torch.nn as nn
import functools

from .utils import get_sigmas, register_model
from .layers import (
    CondRefineBlock,
    RefineBlock,
    ncsn_conv3x3,
    ConditionalResidualBlock,
    get_act,
)
from .normalization import get_normalization

CondResidualBlock = ConditionalResidualBlock
conv3x3 = ncsn_conv3x3


def get_network(config):
    if config.data.image_size < 96:
        return functools.partial(NCSNv2, config=config)
    elif 96 <= config.data.image_size <= 128:
        return functools.partial(NCSNv2_128, config=config)
    elif 128 <= config.data.image_size <= 256:
        return functools.partial(NCSNv2_256, config=config)
    else:
        raise NotImplementedError(
            f"No network suitable for {config.data.image_size}px implemented yet."
        )


@register_model(name="ncsnv2_64")
class NCSNv2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.centered = config.data.centered
        self.norm = get_normalization(config)
        self.nf = nf = config.model.nf


@register_model(name="ncsn")
class NCSN(nn.Module):
    def __init__(self, config):
        super().__init__()



@register_model(name="ncsnv2_128")
class NCSNv2_128(nn.Module):
    """NCSNv2 model architecture for 128px images."""
    def __init__(self, config):
        super().__init__()


@register_model(name="ncsnv2_256")
class NCSNv2_256(nn.Module):
    """NCSNv2 model architecture for 256px images."""
    def __init__(self, config):
        super().__init__()