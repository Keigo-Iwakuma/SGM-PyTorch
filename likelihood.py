"""Various sampling methods."""

import torch
import numpy as np
from scipy import integrate
from models import utils as mutils


def get_div_fn(fn):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(x, t, eps):
        with torch.enable_grad():
            x.requires_grad_(True)
            fn_eps = torch.sum(fn(x, t) * eps)
            grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
        x.requires_grad_(False)
        return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))
    
    return div_fn


def get_likelihood_fn(sde, inverse_scaler, hutchinson_type="Rademacher", rtol=1e-5, atol=1e-5, method="RK45", eps=1e-5):
    """
    Create a function to compute the unbiased log-likelihood estimate of a given data point.

    Args:
        sde: A `sde_lib.SDE` object that represents the forward SDE.
        inverse_scaler: The inverse data normalizer.
        hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
        rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
        atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
        method: A `str`. The algorithm for the black-box ODE solver.
            See documentation for `scipy.integrate.solve_ivp`.
        eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.

    Returns:
        A function that a batch of data points and returns the log-likelihood in bits/dim,
            the latent code, and the number of function evaluations cost by computation.
    """