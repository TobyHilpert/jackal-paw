from __future__ import annotations

import numpy as np


def estimate_gmax_from_ecut(ecut_ry: float) -> float:
    """Crude internal helper for scaffold only.

    In atomic units, kinetic energy T = |G+k|^2 / 2 (Hartree). Input cutoff is in Ry.
    Since 1 Ry = 0.5 Ha, ecut_Ha = ecut_ry / 2.
    Then |G|_max ~ sqrt(2 * ecut_Ha) = sqrt(ecut_ry).
    """
    return float(np.sqrt(max(ecut_ry, 0.0)))


def generate_gvectors(cell: np.ndarray, ecutwfc_ry: float, margin: int = 1) -> np.ndarray:
    """Return a simple integer G-grid superset (scaffold implementation).

    Real implementation should generate reciprocal shells using the reciprocal metric
    and produce per-k basis masks.
    """
    gmax = estimate_gmax_from_ecut(ecutwfc_ry)
    n = int(np.ceil(gmax)) + margin
    ii = np.arange(-n, n + 1)
    g = np.stack(np.meshgrid(ii, ii, ii, indexing="ij"), axis=-1).reshape(-1, 3)
    return g.astype(int)
