from __future__ import annotations

import math


def _next_even(n: int) -> int:
    return n if n % 2 == 0 else n + 1


def choose_fft_shape(base: tuple[int, int, int]) -> tuple[int, int, int]:
    """Scaffold heuristic: round each dimension to next even integer."""
    return tuple(_next_even(int(x)) for x in base)
