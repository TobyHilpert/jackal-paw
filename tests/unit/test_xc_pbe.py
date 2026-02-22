import numpy as np
from jackal.xc import pbe


def test_pbe_returns_finite_arrays():
    rho = np.array([0.1, 0.5, 1.0])
    grad = np.zeros((3, rho.size))
    eps, vxc = pbe.energy_and_potential(rho, grad)
    assert eps.shape == rho.shape
    assert vxc.shape == rho.shape
    assert np.all(np.isfinite(eps))
    assert np.all(np.isfinite(vxc))
