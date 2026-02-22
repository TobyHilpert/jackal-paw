"""Stress computation from strain derivatives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def compute_stress(energy_vs_cell_fn, cell):
    cell0 = jnp.asarray(cell)

    def e_eta(eta_flat):
        eta = eta_flat.reshape(3, 3)
        cell_eta = (jnp.eye(3) + eta) @ cell0
        return energy_vs_cell_fn(cell_eta)

    grad = jax.grad(e_eta)(jnp.zeros(9))
    eta_grad = grad.reshape(3, 3)
    vol = abs(np.linalg.det(np.asarray(cell)))
    sigma = eta_grad / max(vol, 1e-12)
    return np.asarray(sigma)
