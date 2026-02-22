"""Nonlocal projector construction and application helpers."""

from __future__ import annotations

import numpy as np


def projector_overlaps(beta_proj: np.ndarray, psi: np.ndarray) -> np.ndarray:
    return np.asarray(beta_proj).conj().T @ np.asarray(psi)


def apply_projector_operator(beta_proj: np.ndarray, d_matrix: np.ndarray, psi: np.ndarray) -> np.ndarray:
    overlaps = projector_overlaps(beta_proj, psi)
    return np.asarray(beta_proj) @ (np.asarray(d_matrix) @ overlaps)
