"""Radial interpolation tables and reciprocal transforms."""

from __future__ import annotations

import numpy as np


def interp_radial(r: np.ndarray, values: np.ndarray, r_query: np.ndarray) -> np.ndarray:
    return np.interp(np.asarray(r_query, dtype=float), np.asarray(r, dtype=float), np.asarray(values, dtype=float))


def radial_to_reciprocal(r: np.ndarray, f_r: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Spherical transform ``F(q)=4π∫ r² f(r) sin(qr)/(qr) dr``."""
    r_arr = np.asarray(r, dtype=float)
    f_arr = np.asarray(f_r, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    qr = q_arr[:, None] * r_arr[None, :]
    sinc = np.ones_like(qr)
    mask = np.abs(qr) > 1e-12
    sinc[mask] = np.sin(qr[mask]) / qr[mask]
    integrand = 4.0 * np.pi * (r_arr[None, :] ** 2) * f_arr[None, :] * sinc
    return np.trapz(integrand, r_arr, axis=1)
