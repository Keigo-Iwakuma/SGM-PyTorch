"""ALL functions and modules related to model definition.
"""

import torch
import sde_lib
import numpy as np


_MODELS = {}


def register_model(cls=None, *, name=None):
    """A decorator for regsitering model classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)