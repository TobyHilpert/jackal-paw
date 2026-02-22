"""S-orthonormalization routines."""

from __future__ import annotations

import numpy as np


def s_orthonormalize(vectors: np.ndarray, overlap: np.ndarray | None = None, eps: float = 1e-12) -> np.ndarray:
    vecs = np.asarray(vectors)
    s = np.eye(vecs.shape[0], dtype=vecs.dtype) if overlap is None else np.asarray(overlap)
    gram = vecs.conj().T @ s @ vecs
    gram = 0.5 * (gram + gram.conj().T)
    evals, evecs = np.linalg.eigh(gram)
    keep = evals > eps
    if not np.any(keep):
        raise ValueError("All vectors are linearly dependent under the overlap metric")
    inv_sqrt = (evecs[:, keep] / np.sqrt(evals[keep])) @ evecs[:, keep].conj().T
    return vecs @ inv_sqrt
