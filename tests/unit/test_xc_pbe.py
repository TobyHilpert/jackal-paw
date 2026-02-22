import pytest
from jaxpw.xc import pbe


def test_pbe_placeholder():
    with pytest.raises(NotImplementedError):
        pbe.energy_and_potential(None)
