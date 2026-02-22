import pytest

ase = pytest.importorskip("ase")

from inspect import isclass
from jaxpw.calculator.ase_calculator import JaxPWCalculator


def test_calculator_class_exists():
    assert isclass(JaxPWCalculator)
    assert "energy" in JaxPWCalculator.implemented_properties
