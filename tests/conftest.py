import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure editable-style imports also work without installation.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def silicon_cell():
    a = 5.43
    return np.array([[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]], dtype=float)
