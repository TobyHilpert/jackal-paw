import pytest

ase = pytest.importorskip("ase")
from ase.build import bulk
from jackal.calculator.ase_calculator import JaxPWCalculator


def test_si_gamma_scaffold_computes_energy(tmp_path):
    yaml_path = tmp_path / "in.yaml"
    yaml_path.write_text(
        """
        basis:
          ecutwfc: 30.0
        xc:
          functional: pbe
        pseudopotentials:
          Si: /tmp/Si.UPF
        """,
        encoding="utf-8",
    )
    atoms = bulk("Si", "diamond", a=5.43)
    calc = JaxPWCalculator(input_yaml=str(yaml_path))
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    assert isinstance(energy, float)
