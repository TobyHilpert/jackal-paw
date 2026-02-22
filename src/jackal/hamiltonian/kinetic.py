"""Kinetic-energy term in plane-wave basis."""

from __future__ import annotations

import numpy as np


def kinetic_diagonal(g2: np.ndarray) -> np.ndarray:
    return 0.5 * np.asarray(g2, dtype=float)


def apply_kinetic(psi_g: np.ndarray, g2: np.ndarray) -> np.ndarray:
    return kinetic_diagonal(g2) * np.asarray(psi_g)


def kinetic_energy(psi_g: np.ndarray, g2: np.ndarray) -> float:
    psi = np.asarray(psi_g)
    tpsi = apply_kinetic(psi, g2)
    return float(np.real(np.vdot(psi, tpsi)))
