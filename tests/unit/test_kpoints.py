from jaxpw.lattice.kpoints import gamma_only, monkhorst_pack


def test_gamma_only():
    kp = gamma_only()
    assert kp.kpts.shape == (1, 3)
    assert kp.gamma_only


def test_monkhorst_pack_weights_sum():
    kp = monkhorst_pack((2, 2, 2))
    assert abs(kp.weights.sum() - 1.0) < 1e-12
    assert len(kp.kpts) == 8
