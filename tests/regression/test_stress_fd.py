import jax.numpy as jnp
import numpy as np

from jackal.autodiff.stress import compute_stress


def test_stress_matches_volume_derivative():
    bulk_modulus = 2.5

    def energy(cell):
        vol = jnp.abs(jnp.linalg.det(cell))
        return bulk_modulus * vol

    cell = np.eye(3) * 4.0
    stress = compute_stress(energy, cell)
    expected = bulk_modulus * np.eye(3)
    assert np.allclose(stress, expected, atol=5e-6)
