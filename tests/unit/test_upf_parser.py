from pathlib import Path
from jaxpw.io.upf_parser import parse_upf


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
