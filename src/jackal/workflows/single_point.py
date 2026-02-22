from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jaxpw.core.types import System
from jaxpw.io.yaml_input import InputParams


@dataclass
class SinglePointResult:
    energy_ev: float
    free_energy_ev: float
    forces_ev_per_ang: np.ndarray
    stress_voigt_ev_per_ang3: np.ndarray
    metadata: dict


def run_single_point(system: System, params: InputParams) -> SinglePointResult:
    """Top-level single-point workflow scaffold.

    Real implementation should:
      1) parse/load UPF data for species
      2) build lattice, k-points, G-vectors, FFT grids
      3) initialize density/wavefunctions
      4) run SCF (Davidson + DIIS)
      5) evaluate energies, forces, stress
    """
    raise NotImplementedError(
        "Single-point workflow is a scaffold only. Implement basis setup, SCF, and AD post-processing."
    )
