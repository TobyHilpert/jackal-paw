"""Generalized overlap operator S for USPP/PAW."""

from __future__ import annotations

import numpy as np


def apply_overlap(psi: np.ndarray, overlap_matrix: np.ndarray | None = None) -> np.ndarray:
    psi_arr = np.asarray(psi)
    if overlap_matrix is None:
        return psi_arr
    return np.asarray(overlap_matrix) @ psi_arr
