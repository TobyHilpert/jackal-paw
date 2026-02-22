from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class System:
    cell: np.ndarray
    positions: np.ndarray
    numbers: np.ndarray
    pbc: tuple[bool, bool, bool]
    charge: float = 0.0
    spin_polarized: bool = False


@dataclass(frozen=True)
class KPointGrid:
    kpts: np.ndarray
    weights: np.ndarray
    gamma_only: bool = False


@dataclass(frozen=True)
class BasisSet:
    ecutwfc_ry: float
    ecutrho_ry: float
    fft_shape: tuple[int, int, int]
    gvecs_int: np.ndarray


@dataclass(frozen=True)
class PseudopotentialData:
    symbol: str
    pp_type: str  # NC, USPP, PAW
    z_valence: float
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class SCFState:
    rho_r: np.ndarray | None = None
    v_eff_r: np.ndarray | None = None
    eigvals: np.ndarray | None = None
    occs: np.ndarray | None = None
    fermi_level: float | None = None
    converged: bool = False
    iteration: int = 0
    mixer_history: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class EnergyBreakdown:
    kinetic: float = 0.0
    e_local: float = 0.0
    e_nonlocal: float = 0.0
    e_hartree: float = 0.0
    e_xc: float = 0.0
    e_ion_ion: float = 0.0
    e_aug: float = 0.0
    e_entropy: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.kinetic
            + self.e_local
            + self.e_nonlocal
            + self.e_hartree
            + self.e_xc
            + self.e_ion_ion
            + self.e_aug
        )

    @property
    def free_energy(self) -> float:
        return self.total - self.e_entropy
