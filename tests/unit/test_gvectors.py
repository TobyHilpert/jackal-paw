import numpy as np
from jaxpw.lattice.gvectors import generate_gvectors


def test_generate_gvectors_shape():
    cell = np.eye(3)
    g = generate_gvectors(cell, ecutwfc_ry=16.0)
    assert g.ndim == 2 and g.shape[1] == 3
    assert g.dtype.kind in ("i", "u")
