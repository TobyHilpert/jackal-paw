# Input reference (planned YAML schema)

## Top-level sections
- `structure`
- `pseudopotentials`
- `basis`
- `kpoints`
- `xc`
- `occupations`
- `scf`
- `diagonalization`
- `autodiff`
- `runtime`

## Example
```yaml
basis:
  ecutwfc: 40.0   # Ry
  ecutrho: 320.0  # Ry
kpoints:
  mode: gamma
xc:
  functional: pbe
```
