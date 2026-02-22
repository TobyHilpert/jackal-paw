import numpy as np
from jackal.xc import lda


def test_lda_returns_finite_arrays():
    rho = np.array([0.1, 0.5, 1.0])
    eps, vxc = lda.energy_and_potential(rho)
    assert eps.shape == rho.shape
    assert vxc.shape == rho.shape
    assert np.all(np.isfinite(eps))
    assert np.all(np.isfinite(vxc))
