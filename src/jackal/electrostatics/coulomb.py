"""Coulomb kernels and G=0 conventions."""

from __future__ import annotations

import numpy as np


def coulomb_kernel_g(g2: np.ndarray, *, zero_value: float = 0.0) -> np.ndarray:
    """Return reciprocal-space Coulomb kernel ``4π / |G|²``.

    Parameters
    ----------
    g2
        Squared reciprocal-vector magnitudes.
    zero_value
        Value to use for ``G=0`` to enforce a chosen gauge/neutrality policy.
    """
    g2_arr = np.asarray(g2, dtype=float)
    out = np.empty_like(g2_arr, dtype=float)
    mask = g2_arr > 0.0
    out[mask] = 4.0 * np.pi / g2_arr[mask]
    out[~mask] = float(zero_value)
    return out
