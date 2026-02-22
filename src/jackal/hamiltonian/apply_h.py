"""Matrix-free Hamiltonian/overlap application helpers."""

from __future__ import annotations

import numpy as np


def apply_h(psi, kinetic=None, v_local=None, nonlocal_op=None):
    psi_arr = np.asarray(psi)
    out = np.zeros_like(psi_arr, dtype=np.result_type(psi_arr, complex))

    if kinetic is not None:
        out = out + np.asarray(kinetic) * psi_arr
    if v_local is not None:
        out = out + np.asarray(v_local) * psi_arr
    if nonlocal_op is not None:
        out = out + np.asarray(nonlocal_op(psi_arr))
    return out


def apply_s(psi, overlap_op=None):
    psi_arr = np.asarray(psi)
    if overlap_op is None:
        return psi_arr
    return np.asarray(overlap_op(psi_arr))
