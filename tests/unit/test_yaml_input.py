from pathlib import Path
from jackal.io.yaml_input import load_input


def test_load_minimal_yaml(tmp_path: Path):
    p = tmp_path / "in.yaml"
    p.write_text("""basis:
  ecutwfc: 30.0
""", encoding="utf-8")
    inp = load_input(p)
    assert inp.basis.ecutwfc == 30.0
    assert inp.basis.ecutrho >= 120.0
