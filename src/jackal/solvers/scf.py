"""SCF orchestrator scaffold (outer loop with DIIS/Pulay; inner Davidson eigensolve)."""

from __future__ import annotations
from dataclasses import dataclass

from jaxpw.core.types import EnergyBreakdown, SCFState


@dataclass
class SCFResult:
    state: SCFState
    energies: EnergyBreakdown


def run_scf(*args, **kwargs) -> SCFResult:
    raise NotImplementedError("SCF loop not implemented yet")
