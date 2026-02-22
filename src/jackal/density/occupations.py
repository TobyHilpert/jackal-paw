from __future__ import annotations

import numpy as np


def fermi_dirac_occupations(eps: np.ndarray, mu: float, kT: float) -> np.ndarray:
    if kT <= 0:
        return (eps <= mu).astype(float)
    x = (eps - mu) / kT
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (np.exp(x) + 1.0)
