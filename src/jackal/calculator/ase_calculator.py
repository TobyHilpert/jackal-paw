from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from jaxpw.calculator.results_cache import ResultsCache
from jaxpw.io.ase_io import atoms_to_system
from jaxpw.io.yaml_input import InputParams, load_input
from jaxpw.workflows.single_point import run_single_point


class JaxPWCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, input_yaml: str | None = None, params: InputParams | None = None, **kwargs):
        super().__init__(**kwargs)
        if (input_yaml is None) == (params is None):
            raise ValueError("Provide exactly one of input_yaml or params")
        self._params = load_input(input_yaml) if input_yaml is not None else params
        self._cache = ResultsCache()

    def _make_cache_key(self, atoms) -> tuple:
        return (
            tuple(atoms.numbers.tolist()),
            tuple(np.asarray(atoms.cell.array).ravel().round(12).tolist()),
            tuple(np.asarray(atoms.positions).ravel().round(12).tolist()),
            self._params.xc.functional,
            self._params.basis.ecutwfc,
            self._params.basis.ecutrho,
        )

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        key = self._make_cache_key(atoms)
        cached = self._cache.get(key)
        if cached is not None:
            self.results.update(cached)
            return

        system = atoms_to_system(atoms, charge=self._params.system.charge)
        out = run_single_point(system=system, params=self._params)
        results = {
            "energy": out.energy_ev,
            "free_energy": out.free_energy_ev,
            "forces": out.forces_ev_per_ang,
            "stress": out.stress_voigt_ev_per_ang3,
        }
        self._cache.set(key, results)
        self.results.update(results)
