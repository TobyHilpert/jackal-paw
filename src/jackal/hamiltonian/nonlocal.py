"""USPP/PAW nonlocal projector term application."""

from __future__ import annotations

import numpy as np


def apply_nonlocal(psi: np.ndarray, beta_proj: np.ndarray, d_matrix: np.ndarray) -> np.ndarray:
    """Apply separable nonlocal operator β D β† to a state vector/matrix."""
    psi_arr = np.asarray(psi)
    beta = np.asarray(beta_proj)
    dmat = np.asarray(d_matrix)
    proj = beta.conj().T @ psi_arr
    return beta @ (dmat @ proj)
