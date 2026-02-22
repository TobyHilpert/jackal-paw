"""Local ionic + Hartree + XC potential assembly."""

from __future__ import annotations

import numpy as np


def combine_local_potential(v_ionic_r: np.ndarray, v_hartree_r: np.ndarray, v_xc_r: np.ndarray) -> np.ndarray:
    return np.asarray(v_ionic_r) + np.asarray(v_hartree_r) + np.asarray(v_xc_r)
