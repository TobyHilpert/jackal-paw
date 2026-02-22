"""Local pseudopotential generation in reciprocal and real-space representations."""

from __future__ import annotations

import numpy as np


def structure_factor(gvecs: np.ndarray, positions_bohr: np.ndarray) -> np.ndarray:
    phase = np.asarray(gvecs) @ np.asarray(positions_bohr).T
    return np.exp(-1j * phase).sum(axis=1)


def local_potential_g(v_atom_g: np.ndarray, gvecs: np.ndarray, positions_bohr: np.ndarray) -> np.ndarray:
    return np.asarray(v_atom_g) * structure_factor(gvecs, positions_bohr)
