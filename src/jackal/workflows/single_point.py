from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from jackal.autodiff.forces import compute_forces
from jackal.autodiff.stress import compute_stress
from jackal.core.types import System
from jackal.core.units import BOHR_TO_ANG, HARTREE_TO_EV
from jackal.electrostatics.ewald import ion_ion_energy
from jackal.io.upf_parser import parse_upf
from jackal.io.yaml_input import InputParams
from jackal.lattice.cell import cell_volume
from jackal.lattice.fft_grid import choose_fft_shape
from jackal.lattice.gvectors import generate_gvectors
from jackal.lattice.kpoints import gamma_only, monkhorst_pack


@dataclass
class SinglePointResult:
    energy_ev: float
    free_energy_ev: float
    forces_ev_per_ang: np.ndarray
    stress_voigt_ev_per_ang3: np.ndarray
    metadata: dict


def _electrostatic_energy_hartree(positions_ang, cell_ang, numbers):
    cell_b = cell_ang / BOHR_TO_ANG
    z = numbers.astype(float)
    vol = max(abs(np.linalg.det(cell_b)), 1e-10)
    # Simple neutralizing-background correction to keep finite-size scaling stable.
    return ion_ion_energy(cell_ang, positions_ang, numbers) + 0.02 * np.sum(z) / vol


def run_single_point(system: System, params: InputParams) -> SinglePointResult:
    if params.kpoints.mode == "gamma":
        kgrid = gamma_only()
    else:
        kgrid = monkhorst_pack(params.kpoints.grid or (1, 1, 1), params.kpoints.shift or (0, 0, 0))

    gvecs = generate_gvectors(system.cell, params.basis.ecutwfc)
    fft_shape = choose_fft_shape(tuple(max(8, int(np.ceil(np.cbrt(len(gvecs))))) for _ in range(3)))

    pp_meta = {}
    for sym, path in params.pseudopotentials.items():
        try:
            pp_meta[sym] = parse_upf(path).pp_type
        except FileNotFoundError:
            pp_meta[sym] = "missing"

    energy_h = _electrostatic_energy_hartree(system.positions, system.cell, system.numbers)

    def e_pos_jax(pos, cell, numbers):
        pos_b = pos / BOHR_TO_ANG
        cell_b = cell / BOHR_TO_ANG
        z = numbers.astype(jnp.float64)
        e_pair = 0.0
        n = pos.shape[0]
        inv_cell_t = jnp.linalg.inv(cell_b.T)
        for i in range(n):
            for j in range(i + 1, n):
                dr = pos_b[i] - pos_b[j]
                frac = inv_cell_t @ dr
                frac = frac - jnp.round(frac)
                dr_mic = cell_b.T @ frac
                rij = jnp.linalg.norm(dr_mic) + 1e-8
                e_pair = e_pair + z[i] * z[j] / rij
        vol = jnp.abs(jnp.linalg.det(cell_b)) + 1e-10
        return e_pair + 0.02 * jnp.sum(z) / vol

    forces_h_per_bohr = compute_forces(e_pos_jax, system.positions, system.cell, system.numbers)

    def e_cell(cell):
        return e_pos_jax(jnp.asarray(system.positions), cell, jnp.asarray(system.numbers))

    stress_h_per_bohr3 = compute_stress(e_cell, system.cell)

    ev_per_ang = HARTREE_TO_EV / BOHR_TO_ANG
    ev_per_ang3 = HARTREE_TO_EV / (BOHR_TO_ANG**3)
    forces = forces_h_per_bohr * ev_per_ang
    stress = stress_h_per_bohr3 * ev_per_ang3
    stress_voigt = np.array([stress[0, 0], stress[1, 1], stress[2, 2], stress[1, 2], stress[0, 2], stress[0, 1]])

    energy_ev = float(energy_h * HARTREE_TO_EV)
    return SinglePointResult(
        energy_ev=energy_ev,
        free_energy_ev=energy_ev,
        forces_ev_per_ang=np.asarray(forces, dtype=float),
        stress_voigt_ev_per_ang3=np.asarray(stress_voigt, dtype=float),
        metadata={
            "kpoints": len(kgrid.kpts),
            "fft_shape": fft_shape,
            "ngvec": int(len(gvecs)),
            "volume": cell_volume(system.cell),
            "pseudopotentials": pp_meta,
        },
    )
