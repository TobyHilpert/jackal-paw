from __future__ import annotations

import numpy as np
from ase import Atoms

from jaxpw.core.types import System


def atoms_to_system(atoms: Atoms, charge: float = 0.0) -> System:
    return System(
        cell=np.array(atoms.cell.array, dtype=float),
        positions=np.array(atoms.positions, dtype=float),
        numbers=np.array(atoms.numbers, dtype=int),
        pbc=tuple(bool(x) for x in atoms.pbc),
        charge=charge,
    )
