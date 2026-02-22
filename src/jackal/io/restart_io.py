"""Checkpoint/restart I/O scaffold (npz first, HDF5/Zarr later)."""

from __future__ import annotations
from pathlib import Path
from typing import Any

import numpy as np


def save_restart(path: str | Path, payload: dict[str, Any]) -> None:
    np.savez(Path(path), **payload)


def load_restart(path: str | Path) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=True) as data:
        return {k: data[k] for k in data.files}
