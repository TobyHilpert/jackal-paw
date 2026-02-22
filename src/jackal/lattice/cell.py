from __future__ import annotations

import numpy as np


def reciprocal_cell(cell: np.ndarray) -> np.ndarray:
    """Return reciprocal lattice vectors as rows (2π convention)."""
    return 2.0 * np.pi * np.linalg.inv(cell).T


def cell_volume(cell: np.ndarray) -> float:
    return float(abs(np.linalg.det(cell)))


def strain_cell(cell: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Apply small finite strain parameterization h' = (I + eta) h."""
    return (np.eye(3) + eta) @ cell
