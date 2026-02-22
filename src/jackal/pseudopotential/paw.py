"""PAW basis construction helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jackal.core.types import PseudopotentialData


@dataclass(frozen=True)
class PAWBasis:
    beta_projectors: tuple[np.ndarray, ...]
    dij: np.ndarray
    qij: np.ndarray


def build_paw_basis(pp: PseudopotentialData) -> PAWBasis:
    if pp.pp_type != "PAW":
        raise ValueError(f"Expected PAW pseudopotential, got {pp.pp_type}")

    nonlocal_data = pp.raw.get("nonlocal", {})
    beta = tuple(nonlocal_data.get("beta_projectors", []))
    dij = np.asarray(nonlocal_data.get("dij", np.array([], dtype=float)), dtype=float)
    qij = np.asarray(nonlocal_data.get("qij", np.array([], dtype=float)), dtype=float)

    if not beta:
        raise ValueError("PAW pseudopotential is missing nonlocal beta projectors")

    nproj = len(beta)
    if dij.size and dij.shape != (nproj, nproj):
        raise ValueError(f"PAW D_ij has shape {dij.shape}, expected {(nproj, nproj)}")
    if qij.size and qij.shape != (nproj, nproj):
        raise ValueError(f"PAW Q_ij has shape {qij.shape}, expected {(nproj, nproj)}")

    return PAWBasis(beta_projectors=beta, dij=dij, qij=qij)
