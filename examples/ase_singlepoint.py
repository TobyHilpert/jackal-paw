from ase.build import bulk
from jaxpw.calculator.ase_calculator import JaxPWCalculator

atoms = bulk("Si", "diamond", a=5.43)
calc = JaxPWCalculator(input_yaml="examples/si_pbe_gamma/input.yaml")
atoms.calc = calc

# This will fail until the physics kernels are implemented, but shows intended API usage.
try:
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    s = atoms.get_stress()
    print("Energy (eV):", e)
    print("Forces (eV/Å):", f)
    print("Stress (eV/Å^3):", s)
except NotImplementedError as exc:
    print("Scaffold placeholder:", exc)
