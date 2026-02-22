from jackal.validation.compare_qe import compare_scalar_results


def test_qe_comparison_tolerances():
    got = {"energy_ev": -10.01, "pressure_kbar": 2.0}
    ref = {"energy_ev": -10.0, "pressure_kbar": 1.5}
    tol = {"energy_ev": 0.02, "pressure_kbar": 1.0}

    summary = compare_scalar_results(got, ref, tol)
    assert summary["ok"]
    assert summary["failures"] == []
