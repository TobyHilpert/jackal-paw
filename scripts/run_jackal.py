from __future__ import annotations

import argparse
from pathlib import Path

from ase.io import read

from jaxpw.io.yaml_input import load_input
from jaxpw.calculator.ase_calculator import JaxPWCalculator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run jaxpw single-point calculation from YAML input")
    parser.add_argument("input_yaml", type=Path)
    args = parser.parse_args()

    params = load_input(args.input_yaml)
    if params.structure.path is None:
        raise SystemExit("input YAML must define structure.path for this script")
    structure_path = (args.input_yaml.parent / params.structure.path).resolve()
    atoms = read(structure_path)
    atoms.calc = JaxPWCalculator(input_yaml=str(args.input_yaml))
    try:
        energy = atoms.get_potential_energy()
        print(f"Energy (eV): {energy:.12f}")
    except NotImplementedError as exc:
        print(f"Scaffold placeholder: {exc}")


if __name__ == "__main__":
    main()
