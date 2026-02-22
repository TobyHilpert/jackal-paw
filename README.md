# jackal-paw

A plane-wave DFT code implemented in Python using JAX, with ASE integration.

## Status
Scaffold / architecture skeleton (not a complete solver yet).

## Goals
- Plane-wave Kohn-Sham DFT (LDA/PBE; staged HF/EXX support)
- UPF pseudopotentials (NC/USPP/PAW)
- Blocked Davidson + DIIS/Pulay SCF
- Forces and stress via JAX automatic differentiation
- ASE calculator interface

## Install (editable)
```bash
pip install -e .[dev]
```

## Docs
- Full code specification: `docs/code_spec.md`
- Input reference: `docs/input_reference.md`

## Examples
```bash
python scripts/run_jackal.py examples/si_pbe_gamma/input.yaml
python examples/ase_singlepoint.py
```
