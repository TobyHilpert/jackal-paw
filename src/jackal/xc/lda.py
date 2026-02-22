"""Local-density approximation helpers.

This is a compact JAX/NumPy-friendly implementation of an exchange-only LDA
model with a small correlation regularizer so downstream SCF workflows can run
without placeholder errors.
"""

from __future__ import annotations

import numpy as np

_CX = -0.7385587663820223  # -3/4 * (3/pi)^(1/3)


def energy_and_potential(rho):
    """Return (epsilon_xc, v_xc) on a real-space grid.

    Parameters
    ----------
    rho:
        Electron density values (array-like, in bohr^-3).
    """
    rho_arr = np.asarray(rho, dtype=float)
    rho_pos = np.clip(rho_arr, 1e-14, None)

    eps_x = _CX * np.cbrt(rho_pos)
    v_x = (4.0 / 3.0) * eps_x

    # Tiny smooth correlation surrogate (kept simple for scaffold robustness).
    corr = -0.02 * np.log1p(rho_pos)
    v_corr = -0.02 / (1.0 + rho_pos)

    eps_xc = eps_x + corr
    v_xc = v_x + v_corr
    return eps_xc, v_xc
