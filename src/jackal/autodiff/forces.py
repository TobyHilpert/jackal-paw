"""Force computation utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def compute_forces(energy_fn, positions, cell, numbers):
    pos = jnp.asarray(positions)
    cell_arr = jnp.asarray(cell)
    z = jnp.asarray(numbers)

    def wrapped(p):
        return energy_fn(p, cell_arr, z)

    grad = jax.grad(wrapped)(pos)
    return -np.asarray(grad)
