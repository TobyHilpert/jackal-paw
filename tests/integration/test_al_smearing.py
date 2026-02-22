import numpy as np

from jackal.density.occupations import fermi_dirac_occupations


def test_fermi_dirac_smearing_monotonic_and_bounded():
    eps = np.linspace(-1.0, 1.0, 9)
    occ = fermi_dirac_occupations(eps, mu=0.0, kT=0.1)
    assert np.all((occ >= 0.0) & (occ <= 1.0))
    assert np.all(np.diff(occ) <= 0.0)
