"""Ion-ion electrostatics helpers."""

from __future__ import annotations

import numpy as np

from jackal.core.units import BOHR_TO_ANG


def _minimum_image(displacement_bohr: np.ndarray, cell_bohr: np.ndarray) -> np.ndarray:
    frac = np.linalg.solve(cell_bohr.T, displacement_bohr)
    frac -= np.round(frac)
    return frac @ cell_bohr


def ion_ion_energy(system_cell_ang: np.ndarray, positions_ang: np.ndarray, numbers: np.ndarray) -> float:
    """Compute pairwise ion-ion Coulomb energy in Hartree with minimum-image PBC."""
    cell_b = np.asarray(system_cell_ang, dtype=float) / BOHR_TO_ANG
    pos_b = np.asarray(positions_ang, dtype=float) / BOHR_TO_ANG
    z = np.asarray(numbers, dtype=float)

    energy = 0.0
    n = len(z)
    for i in range(n):
        for j in range(i + 1, n):
            dr = _minimum_image(pos_b[i] - pos_b[j], cell_b)
            rij = max(np.linalg.norm(dr), 1e-12)
            energy += z[i] * z[j] / rij
    return float(energy)
