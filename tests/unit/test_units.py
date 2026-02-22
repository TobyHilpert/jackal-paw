from jackal.core.units import hartree_to_ev, ev_to_hartree


def test_unit_roundtrip():
    x = 1.234
    assert abs(ev_to_hartree(hartree_to_ev(x)) - x) < 1e-12
