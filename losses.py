"""
All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import numpy as np
from models import utils as mutils
from sde_lib import VESDE, VPSDE


def get_optimizer(config, params):
    """Returns a flax optimizer object based on `config`."""
    if config.optim.optimizer == "Adam":
        optimizer = optim.Adam(
            params,
            lr=config.optim.lr,
            betas=(config.optim.beta1, 0.999),
            eps=config.optim.eps,
            weight_decay=config.optim.weight_decay
        )
    else:
        raise NotImplementedError(
            f"Optimizer {config.optim.optimizer} not supported yet!"
        )

    return optimizer