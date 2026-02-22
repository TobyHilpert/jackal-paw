"""Implicit differentiation / custom-VJP helpers for the SCF fixed point."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def scf_residual(rho: np.ndarray, fixed_point_map: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    rho_arr = np.asarray(rho)
    return np.asarray(fixed_point_map(rho_arr)) - rho_arr


def fixed_point_jacobian_fd(rho: np.ndarray, fixed_point_map: Callable[[np.ndarray], np.ndarray], eps: float = 1e-6) -> np.ndarray:
    rho_arr = np.asarray(rho, dtype=float)
    base = np.asarray(fixed_point_map(rho_arr), dtype=float)
    jac = np.zeros((rho_arr.size, rho_arr.size), dtype=float)
    for i in range(rho_arr.size):
        pert = rho_arr.copy()
        pert[i] += eps
        col = (np.asarray(fixed_point_map(pert), dtype=float) - base) / eps
        jac[:, i] = col.ravel()
    return jac - np.eye(rho_arr.size)
