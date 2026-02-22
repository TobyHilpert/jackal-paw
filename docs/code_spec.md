# JAX Plane-Wave DFT Code Specification

This document defines a research-grade code specification for a plane-wave DFT code implemented in Python + JAX, with ASE integration and support for modern pseudopotential methods (USPP/PAW), multiple XC choices (LDA/PBE/HF), SCF minimization (blocked Davidson + DIIS), and forces/stress via automatic differentiation.

---

## 1) Comparison of requirements to existing plane-wave DFT codes (especially QE and VASP)

### 1.1 Feature-by-feature comparison

#### A. Structure I/O via ASE
- **Quantum ESPRESSO (QE):** Native input formats; ASE supports I/O and wrappers around QE.
- **VASP:** Native POSCAR/CONTCAR; ASE supports VASP I/O and calculator wrappers.
- **Design implication:** Using ASE as the structure interface is strongly recommended for interoperability and workflow integration.

#### B. YAML input file
- **QE/VASP:** Use custom text formats (namelists, INCAR/KPOINTS/POSCAR/POTCAR).
- **Design implication:** YAML is cleaner for Python, but must be paired with a validated schema and explicit unit handling.

#### C. Plane-wave basis and k-point grid
- **QE/VASP:** Mature implementations with FFT grids, symmetry-reduced k-points, gamma optimizations, multiple solvers.
- **Design implication:** Core requirement is feasible, but requires explicit handling of FFT grids, G-vectors, cutoffs, weights, and occupations.

#### D. Read/use open-source UltraSoft and PAW potential files
- **QE:** Uses UPF (NC/USPP/PAW support available in modern workflows).
- **VASP:** Uses POTCAR (license-restricted; not open-source).
- **Design implication:** Primary target format should be **UPF (v2.x)**. Comparison to VASP behavior is useful, but POTCAR should not be the parser target.

#### E. Evaluate Hartree-Fock, LDA, and PBE energies
- **QE/VASP:** LDA/PBE standard; exact exchange/hybrids supported with substantial complexity.
- **Design implication:**
  - LDA/PBE are natural for a JAX plane-wave code.
  - Hartree-Fock (periodic, plane-wave, k-point sampled, PAW/USPP) is substantially more complex and should be staged.

#### F. Self-consistent minimization using blocked Davidson and DIIS
- **QE/VASP:** Multiple robust diagonalizers and density mixing schemes (Pulay/Broyden/DIIS, etc.).
- **Design implication:** Must explicitly separate:
  1. Inner eigensolver (blocked Davidson)
  2. Outer SCF fixed-point iteration (DIIS/Pulay mixing)

#### G. Forces and stress via JAX automatic differentiation
- **QE/VASP:** Traditionally implemented with hand-derived analytic expressions.
- **Design implication:** This is a key differentiator. Requires JAX-friendly code structure and an explicit SCF differentiation strategy.

#### H. ASE calculator interface
- **QE/VASP via ASE:** Available through wrappers.
- **Design implication:** Excellent choice; requires caching, unit conventions, and robust repeated-call behavior.

### 1.2 Missing but necessary features compared to QE/VASP
The original requirement list omits several capabilities that are essential in practical plane-wave DFT codes:
1. Occupations, smearing, and Fermi level determination
2. Charge-density / potential mixing details
3. FFT grid policy and aliasing controls
4. Generalized overlap operator (S) for USPP/PAW
5. Ionic electrostatics (Ewald or equivalent)
6. Convergence control and restart/checkpoint support
7. Symmetry and irreducible k-point reduction (may be deferred)
8. Spin treatment (at least collinear planned)
9. Validation against reference codes
10. Precision and performance policy (JAX, complex arithmetic, FFTs)

---

## 2) Updated and extended requirement set

### 2.1 Core functional requirements (updated)

#### Input/Output and user interface
1. Structure I/O via ASE
2. YAML input with typed/validated schema
3. ASE calculator interface returning energy/free energy/forces/stress

#### Electronic-structure model
4. 3D periodic plane-wave Kohn-Sham DFT (spin-unpolarized v1; collinear v2)
5. Plane-wave basis generation
6. k-point sampling (gamma-only and Monkhorst-Pack)
7. FFT-based reciprocal/real-space grids with `ecutwfc` and `ecutrho`

#### Pseudopotentials / PAW / USPP
8. UPF parser (v2.x target)
9. Ultrasoft pseudopotential support
10. PAW support
11. Pseudopotential validation checks

#### Energies / functionals
12. Total energy decomposition
13. LDA and PBE XC implementations (JAX-native)
14. Hartree-Fock (exact exchange) module with staged support

#### SCF and minimization
15. Nested SCF algorithm (blocked Davidson inner + DIIS/Pulay outer)
16. Occupation update + Fermi level solve
17. SCF convergence controls
18. Preconditioning (Davidson kinetic + Kerker-like mixing preconditioner)

#### Forces and stress
19. Forces via JAX AD
20. Stress via JAX AD
21. Explicit SCF differentiation strategy
22. Basis-set differentiability strategy (stress / Pulay issues)

#### Software engineering and validation
23. Precision policy (`jax_enable_x64=True`)
24. Modular architecture
25. Checkpoint/restart support
26. Logging and diagnostics
27. Verification and validation
28. Performance portability (CPU/GPU via JAX)

### 2.2 Scope assumptions
- v1 target: periodic solids and supercells, scalar-relativistic pseudopotentials, no SOC
- spin-unpolarized first; collinear spin planned
- symmetry hooks in architecture; full irreducible k-point reduction may be deferred
- HF/EXX energy required; full PAW+k-point EXX forces/stress can be later
- pseudopotential primary format is UPF, not POTCAR

---

## 3) High-level implementation structure

### 3.1 Proposed package layout
```text
jackal/
  calculator/
  io/
  core/
  lattice/
  pseudopotential/
  density/
  hamiltonian/
  xc/
  electrostatics/
  solvers/
  autodiff/
  workflows/
  validation/
```

### 3.2 Core data model
Use immutable dataclasses (registered as JAX pytrees where appropriate) for:
- `System`
- `InputParams`
- `PseudopotentialData`
- `BasisSet`
- `SCFState`
- `EnergyBreakdown`

### 3.3 High-level execution flow
1. Read YAML and validate schema
2. Read structure via ASE
3. Parse UPF files
4. Build lattice, k-points, basis, FFT grids
5. Initialize density/wavefunctions
6. Run SCF loop
7. Evaluate energy breakdown
8. Evaluate forces/stress (AD)
9. Return results and optional restart

### 3.4 Separation of numerical concerns
- Inner loop: generalized eigenproblem `Hc = εSc` (USPP/PAW)
- Outer loop: SCF fixed point with DIIS/Pulay mixing

---

## 4) Complete code specification (module-level details)

### 4.1 Scope and feature tiers
#### v1 (usable implementation)
- ASE I/O and ASE calculator
- YAML schema and parsing
- plane-wave basis and k-points
- UPF parsing (USPP/PAW metadata included)
- LDA/PBE energies
- SCF with blocked Davidson + DIIS/Pulay
- Forces and stress via JAX AD (LDA/PBE pathways)
- Energy/forces/stress delivery to ASE

#### v1.5 / v2 (maturity targets)
- broad USPP/PAW coverage
- general-k-point HF/EXX and accelerations
- symmetry reduction, collinear spin
- advanced smearing and hybrid functionals

### 4.2 Input specification (YAML schema)
Top-level sections:
- `structure`, `pseudopotentials`, `basis`, `kpoints`, `xc`, `occupations`, `scf`, `diagonalization`, `autodiff`, `runtime`

Validation rules include:
- `ecutrho >= 4*ecutwfc` for NC (higher defaults for USPP/PAW)
- `nbands >= nocc + buffer`
- HF compatibility checks
- species consistency between structure and PP map

### 4.3 ASE integration specification
`JaxPWCalculator(ase.calculators.calculator.Calculator)` must:
- expose `energy`, `free_energy`, `forces`, `stress`
- accept YAML path or parsed params
- cache by atomic configuration + key params
- return ASE units (eV, eV/Å, eV/Å^3)

### 4.4 Structure and lattice handling
- `io/ase_io.py`: `ase.Atoms` ↔ internal `System`
- `lattice/cell.py`: reciprocal lattice, volume, metrics, strain parameterization `h'=(I+η)h`

### 4.5 k-point and plane-wave basis
- `lattice/kpoints.py`: Monkhorst-Pack, gamma-only, future symmetry hooks
- `lattice/gvectors.py`: G-vectors, per-k kinetic energies, basis masks, AD-aware cutoff strategy
- `lattice/fft_grid.py`: FFT grid dimensions and mappings

### 4.6 Pseudopotential / USPP / PAW specification
- `io/upf_parser.py`: parse UPF v2.x metadata, local/nonlocal, augmentation, PAW data
- `pseudopotential/radial.py`: interpolation and reciprocal-space transform tables
- nonlocal and overlap modules must provide matrix-free `V_NL|ψ⟩` and `S|ψ⟩`
- `pseudopotential/paw.py`: PAW projector overlaps, on-site density matrices, one-center corrections, compensation charges

### 4.7 Density, Hartree, XC, electrostatics
- `density/rho.py`: density from orbitals/occupations/k-weights
- `density/augmentation.py`: USPP/PAW augmentation charges
- `electrostatics/hartree.py`: reciprocal-space Poisson solve with explicit `G=0` policy
- `xc/lda.py`, `xc/pbe.py`: JAX-native XC energy density and potential implementations
- `electrostatics/ewald.py`: ion-ion energy and stress

### 4.8 Hamiltonian and overlap application
`hamiltonian/apply_h.py` applies matrix-free:
1. kinetic
2. local potential (FFT to real/multiply/FFT back)
3. nonlocal projectors
4. USPP/PAW corrections

`hamiltonian/overlap.py` handles identity (NC) or generalized overlap (USPP/PAW).

### 4.9 Eigensolver specification: blocked Davidson
`solvers/davidson.py` solves generalized eigenproblem `HC = SCε` using:
- block iterations
- residuals `r_i = Hc_i - ε_i Sc_i`
- kinetic preconditioner
- subspace expansion/restart
- S-orthonormalization
- Rayleigh-Ritz in small dense subspaces
- residual convergence criteria

### 4.10 SCF loop and DIIS/Pulay mixing
`solvers/scf.py` outer loop:
1. build `v_eff`
2. solve bands (Davidson)
3. update occupations/Fermi level
4. build output density
5. compute residual
6. mix (DIIS/Pulay)
7. check convergence

`density/mixer.py` must include linear fallback and Kerker preconditioning.

### 4.11 Occupations and smearing
`density/occupations.py` provides:
- electron counting from valence + charge
- Fermi level solve
- fixed / Fermi-Dirac / Gaussian smearing
- entropy term for free energy

### 4.12 Hartree-Fock / exact exchange (EXX)
`electrostatics/exx.py` evaluates periodic exact exchange with:
- occupied-state and k-point sums
- Coulomb singularity treatment (`q→0`)
- occupation/k-point weights
- documented smearing / partial-occupation treatment

Staged performance:
- v1 small-system implementation (gamma-only or restricted k-point)
- v2 accelerated methods (e.g., ACE)

PAW/USPP EXX augmentation corrections should be explicitly separated from pseudo-space EXX.

### 4.13 Forces and stress via JAX AD
#### Forces (`autodiff/forces.py`)
Compute `F_a = -∂E/∂R_a` via `jax.grad` on a converged energy wrapper.

SCF differentiation strategy (must be explicit):
1. implicit differentiation of fixed-point SCF (preferred)
2. custom VJP for SCF solver
3. unrolled SCF differentiation (debug only)

#### Stress (`autodiff/stress.py`)
Differentiate w.r.t. strain and convert to tensor / ASE Voigt form.

Critical issue: basis discontinuity under strain at fixed cutoff. Required handling includes at least one of:
- fixed G-superset + masks
- smooth cutoff window in differentiable mode
- explicit Pulay stress correction

Validation: compare AD stress to finite-difference strain derivatives.

### 4.14 JAX engineering specification
- default precision: `float64` / `complex128`
- JIT shape-stable kernels (`apply_H`, `apply_S`, density build, Hartree/XC)
- keep parsing/logging/file I/O outside JIT
- vectorize over k-points/bands with `vmap` where practical

### 4.15 Restart / checkpoint specification
`io/restart_io.py` stores:
- density
- optional wavefunctions
- eigenvalues
- occupations
- input/system hash and PP metadata hash

Formats: `.npz` initially; HDF5/Zarr later.

### 4.16 Testing and validation specification
#### Unit tests
- YAML schema
- UPF parsing (NC/USPP/PAW)
- k-points, G-vectors, FFT grids
- Ewald energy
- LDA/PBE kernels

#### Integration tests (small systems, compare against QE)
- Si diamond (PBE)
- Al fcc (metal + smearing)
- MgO or NaCl
- H2O in box (gamma-only)
- open PAW and USPP UPF benchmark cases

#### AD validation tests
- forces vs finite differences
- stress vs finite differences
- tests for converged SCF and fixed-iteration debug mode

Define explicit tolerances for energy, forces, stress (looser initially for HF and early PAW/USPP stages).

### 4.17 Documentation artifacts required in repository
1. Theory notes
2. Developer architecture docs
3. Input reference
4. Validation report

### 4.18 Suggested implementation order
1. ASE + YAML + lattice/kpoints + FFT grid
2. NC UPF subset + LDA/PBE + SCF + Davidson + DIIS
3. AD forces/stress for NC + LDA/PBE
4. USPP (overlap + augmentation)
5. PAW
6. HF/EXX (gamma-only → k-point)
7. Robustness, performance, symmetry, spin
