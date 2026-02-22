"""Convergence checks and SCF diagnostics helpers (scaffold)."""

from __future__ import annotations


def energy_converged(delta_e: float, etol: float) -> bool:
    return abs(delta_e) < etol
