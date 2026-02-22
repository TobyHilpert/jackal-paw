"""Electron density construction from orbitals, occupations, and k-point weights."""

from __future__ import annotations

import numpy as np


def density_from_orbitals(psi_r: np.ndarray, occupations: np.ndarray, k_weights: np.ndarray) -> np.ndarray:
    """Build real-space density ``rho(r)=Σ_k w_k Σ_n f_nk |ψ_nk(r)|²``."""
    psi = np.asarray(psi_r)
    occ = np.asarray(occupations, dtype=float)
    wk = np.asarray(k_weights, dtype=float)

    if psi.ndim != 3:
        raise ValueError("psi_r must have shape (nk, nbands, ngrid)")
    if occ.shape != psi.shape[:2]:
        raise ValueError("occupations shape must match (nk, nbands)")
    if wk.shape[0] != psi.shape[0]:
        raise ValueError("k_weights length must match nk")

    band_density = np.abs(psi) ** 2
    return np.einsum("k,kn,kng->g", wk, occ, band_density, optimize=True).real
