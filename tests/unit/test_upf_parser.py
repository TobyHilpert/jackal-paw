from pathlib import Path

import numpy as np

from jackal.io.upf_parser import parse_upf
from jackal.pseudopotential.paw import build_paw_basis
from jackal.pseudopotential.uspp import build_uspp_basis


def test_parse_minimal_upf(tmp_path: Path):
    p = tmp_path / "X.UPF"
    p.write_text(
        "<UPF><PP_HEADER element='X' z_valence='4.0' is_ultrasoft='F' is_paw='F'/></UPF>",
        encoding="utf-8",
    )
    pp = parse_upf(p)
    assert pp.symbol == "X"
    assert pp.z_valence == 4.0
    assert pp.pp_type == "NC"


def test_parse_uspp_and_build_basis(tmp_path: Path):
    p = tmp_path / "Si.UPF"
    p.write_text(
        """
<UPF>
  <PP_HEADER element='Si' z_valence='4.0' is_ultrasoft='T' is_paw='F' number_of_proj='2'/>
  <PP_MESH>
    <PP_R>0.1 0.2 0.3</PP_R>
    <PP_RAB>0.1 0.1 0.1</PP_RAB>
  </PP_MESH>
  <PP_LOCAL>1.0 2.0 3.0</PP_LOCAL>
  <PP_NONLOCAL>
    <PP_BETA.1 angular_momentum='0'>0.5 0.6 0.7</PP_BETA.1>
    <PP_BETA.2 angular_momentum='1'>1.5 1.6 1.7</PP_BETA.2>
    <PP_DIJ>1.0 0.0 0.0 2.0</PP_DIJ>
    <PP_QIJ>0.1 0.0 0.0 0.2</PP_QIJ>
  </PP_NONLOCAL>
</UPF>
        """.strip(),
        encoding="utf-8",
    )

    pp = parse_upf(p)
    assert pp.pp_type == "USPP"
    assert np.allclose(pp.raw["mesh"]["r"], np.array([0.1, 0.2, 0.3]))
    assert np.allclose(pp.raw["local_potential"], np.array([1.0, 2.0, 3.0]))

    basis = build_uspp_basis(pp)
    assert len(basis.beta_projectors) == 2
    assert basis.dij.shape == (2, 2)
    assert basis.qij.shape == (2, 2)


def test_parse_paw_and_build_basis(tmp_path: Path):
    p = tmp_path / "O.UPF"
    p.write_text(
        """
<UPF>
  <PP_HEADER element='O' z_valence='6.0' is_ultrasoft='F' is_paw='T' number_of_proj='1'/>
  <PP_NONLOCAL>
    <PP_BETA.1 angular_momentum='0'>0.2 0.4</PP_BETA.1>
    <PP_DIJ>3.0</PP_DIJ>
    <PP_QIJ>0.3</PP_QIJ>
  </PP_NONLOCAL>
</UPF>
        """.strip(),
        encoding="utf-8",
    )

    pp = parse_upf(p)
    assert pp.pp_type == "PAW"

    basis = build_paw_basis(pp)
    assert len(basis.beta_projectors) == 1
    assert np.allclose(basis.dij, np.array([[3.0]]))
    assert np.allclose(basis.qij, np.array([[0.3]]))
