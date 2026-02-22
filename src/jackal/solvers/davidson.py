"""Small dense blocked-Davidson style wrapper.

For scaffold use we solve the Rayleigh-Ritz problem in one shot using explicit
matrices when callbacks are linear over the canonical basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class DavidsonResult:
    eigvals: Any
    eigvecs: Any
    residual_norms: Any
    converged: bool


def _build_matrix(apply_op: Callable, n: int) -> np.ndarray:
    eye = np.eye(n)
    cols = [np.asarray(apply_op(eye[:, i])) for i in range(n)]
    return np.column_stack(cols)


def solve_blocked_davidson(apply_h: Callable, apply_s: Callable, guess, params) -> DavidsonResult:
    guess_arr = np.asarray(guess)
    n = guess_arr.shape[0]
    h_mat = _build_matrix(apply_h, n)
    s_mat = _build_matrix(apply_s, n)

    # Regularize overlap and map to standard Hermitian problem.
    s_mat = 0.5 * (s_mat + s_mat.conj().T)
    h_mat = 0.5 * (h_mat + h_mat.conj().T)
    evals_s, evecs_s = np.linalg.eigh(s_mat)
    evals_s = np.clip(evals_s, 1e-10, None)
    s_inv_sqrt = (evecs_s / np.sqrt(evals_s)) @ evecs_s.conj().T
    h_tilde = s_inv_sqrt.conj().T @ h_mat @ s_inv_sqrt

    eigvals, y = np.linalg.eigh(h_tilde)
    eigvecs = s_inv_sqrt @ y

    nroots = int(getattr(params, "nbands", min(guess_arr.shape[-1], n)))
    eigvals = eigvals[:nroots]
    eigvecs = eigvecs[:, :nroots]

    res = h_mat @ eigvecs - s_mat @ eigvecs * eigvals[None, :]
    res_norm = np.linalg.norm(res, axis=0)
    tol = float(getattr(params, "residual_tol", 1e-8))
    return DavidsonResult(eigvals=eigvals, eigvecs=eigvecs, residual_norms=res_norm, converged=bool(np.all(res_norm < tol)))
