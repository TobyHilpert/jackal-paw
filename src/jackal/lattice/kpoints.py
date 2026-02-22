from __future__ import annotations

import itertools
import numpy as np

from jaxpw.core.types import KPointGrid


def gamma_only() -> KPointGrid:
    return KPointGrid(kpts=np.zeros((1, 3)), weights=np.ones(1), gamma_only=True)


def monkhorst_pack(grid: tuple[int, int, int], shift: tuple[float, float, float] = (0, 0, 0)) -> KPointGrid:
    nx, ny, nz = grid
    sx, sy, sz = shift
    pts = []
    for i, j, k in itertools.product(range(nx), range(ny), range(nz)):
        pts.append(((i + sx) / nx - 0.5, (j + sy) / ny - 0.5, (k + sz) / nz - 0.5))
    kpts = np.asarray(pts, dtype=float)
    weights = np.full(len(kpts), 1.0 / len(kpts))
    return KPointGrid(kpts=kpts, weights=weights, gamma_only=False)
