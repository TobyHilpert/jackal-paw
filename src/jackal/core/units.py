"""Unit conversions.

Internal unit convention is planned to be Hartree / bohr.
ASE interface returns eV / Å and eV / Å^3.
"""

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.529177210903
RY_TO_HARTREE = 0.5
RY_TO_EV = RY_TO_HARTREE * HARTREE_TO_EV


def hartree_to_ev(x: float) -> float:
    return x * HARTREE_TO_EV


def ev_to_hartree(x: float) -> float:
    return x / HARTREE_TO_EV
