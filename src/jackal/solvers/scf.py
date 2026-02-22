"""SCF orchestrator with simple fixed-point update + mixer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jackal.core.types import EnergyBreakdown, SCFState
from jackal.density.mixer import DIISMixer


@dataclass
class SCFResult:
    state: SCFState
    energies: EnergyBreakdown


def run_scf(initial_rho, build_rho_out, energy_fn, max_iter=50, rhotol=1e-6, beta=0.4, mixing_ndim=8, kerker_q0=1.0) -> SCFResult:
    rho = np.asarray(initial_rho, dtype=float)
    mixer = DIISMixer(beta=beta, ndim=mixing_ndim, kerker_q0=kerker_q0)
    state = SCFState(rho_r=rho.copy())

    converged = False
    for it in range(1, max_iter + 1):
        rho_out = np.asarray(build_rho_out(rho), dtype=float)
        resid = np.linalg.norm(rho_out - rho) / max(np.sqrt(rho.size), 1.0)
        rho = mixer.mix(rho, rho_out)
        state.mixer_history = mixer.history.copy()
        state.iteration = it
        state.rho_r = rho.copy()
        if resid < rhotol:
            converged = True
            break

    state.converged = converged
    energy = float(energy_fn(state.rho_r))
    return SCFResult(state=state, energies=EnergyBreakdown(e_hartree=energy))
