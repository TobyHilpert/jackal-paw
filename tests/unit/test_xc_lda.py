import pytest
from jaxpw.xc import lda


def test_lda_placeholder():
    with pytest.raises(NotImplementedError):
        lda.energy_and_potential(None)
