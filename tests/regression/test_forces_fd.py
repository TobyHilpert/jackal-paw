import numpy as np

from jackal.autodiff.forces import compute_forces


def test_forces_match_harmonic_analytic():
    def energy(pos, cell, numbers):
        return 0.5 * np.sum(pos**2)

    pos = np.array([[0.1, -0.2, 0.05]])
    cell = np.eye(3) * 5.0
    numbers = np.array([1])

    forces = compute_forces(energy, pos, cell, numbers)
    assert np.allclose(forces, -pos, atol=1e-10)
