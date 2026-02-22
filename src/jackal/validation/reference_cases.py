"""Reference benchmark cases and expected tolerances."""

from __future__ import annotations

REFERENCE_CASES: dict[str, dict] = {
    "si_pbe_gamma": {
        "reference": {"energy_ev": -100.0},
        "tolerances": {"energy_ev": 10.0},
    },
    "al_pbe_kmesh": {
        "reference": {"energy_ev": -50.0},
        "tolerances": {"energy_ev": 10.0},
    },
}
