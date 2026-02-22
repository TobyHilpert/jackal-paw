"""Hartree potential and energy from reciprocal-space density."""

from __future__ import annotations

import numpy as np

from jackal.electrostatics.coulomb import coulomb_kernel_g


def hartree_potential_g(rho_g: np.ndarray, g2: np.ndarray, *, g0_value: float = 0.0) -> np.ndarray:
    rho_g_arr = np.asarray(rho_g)
    kernel = coulomb_kernel_g(g2, zero_value=g0_value)
    return kernel * rho_g_arr


def hartree_energy(rho_g: np.ndarray, g2: np.ndarray, volume_bohr3: float, *, g0_value: float = 0.0) -> float:
    rho_g_arr = np.asarray(rho_g)
    v_h = hartree_potential_g(rho_g_arr, g2, g0_value=g0_value)
    e = 0.5 / float(volume_bohr3) * np.sum(np.conj(rho_g_arr) * v_h)
    return float(np.real(e))
