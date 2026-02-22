"""Compact PBE-like enhancement on top of LDA.

The implementation is intentionally lightweight but shape-stable and useful for
validation and workflow plumbing.
"""

from __future__ import annotations

import numpy as np

from jackal.xc import lda


def energy_and_potential(rho, grad_rho=None):
    rho_arr = np.asarray(rho, dtype=float)
    eps_lda, v_lda = lda.energy_and_potential(rho_arr)

    if grad_rho is None:
        return eps_lda, v_lda

    grad = np.asarray(grad_rho, dtype=float)
    grad2 = np.sum(np.square(grad), axis=0) if grad.ndim > rho_arr.ndim else np.square(grad)
    s2 = grad2 / np.clip(rho_arr ** (8.0 / 3.0), 1e-14, None)
    enhancement = 1.0 + 0.12 * s2 / (1.0 + 0.24 * s2)

    eps = eps_lda * enhancement
    v = v_lda * enhancement
    return eps, v
