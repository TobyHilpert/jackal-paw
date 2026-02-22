"""Projected H/S subspace build and Rayleigh-Ritz solve."""

from __future__ import annotations

import numpy as np


def build_projected_matrices(h_mat: np.ndarray, s_mat: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b = np.asarray(basis)
    h = np.asarray(h_mat)
    s = np.asarray(s_mat)
    return b.conj().T @ h @ b, b.conj().T @ s @ b


def solve_rayleigh_ritz(h_proj: np.ndarray, s_proj: np.ndarray, nroots: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    h = 0.5 * (np.asarray(h_proj) + np.asarray(h_proj).conj().T)
    s = 0.5 * (np.asarray(s_proj) + np.asarray(s_proj).conj().T)

    evals_s, evecs_s = np.linalg.eigh(s)
    evals_s = np.clip(evals_s, 1e-12, None)
    s_inv_sqrt = (evecs_s / np.sqrt(evals_s)) @ evecs_s.conj().T
    h_tilde = s_inv_sqrt.conj().T @ h @ s_inv_sqrt
    evals, y = np.linalg.eigh(h_tilde)
    vecs = s_inv_sqrt @ y

    if nroots is not None:
        evals = evals[:nroots]
        vecs = vecs[:, :nroots]
    return evals, vecs
