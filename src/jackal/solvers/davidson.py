"""Blocked Davidson generalized eigensolver scaffold.

Target: solve H C = S C ε using matrix-free apply_H/apply_S callbacks.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class DavidsonResult:
    eigvals: Any
    eigvecs: Any
    residual_norms: Any
    converged: bool


def solve_blocked_davidson(apply_h: Callable, apply_s: Callable, guess, params) -> DavidsonResult:
    raise NotImplementedError("Blocked Davidson solver not implemented yet")
