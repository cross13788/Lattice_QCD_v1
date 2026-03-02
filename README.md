# Lattice_QCD_v1

A C++ implementation of **lattice quantum chromodynamics** (LQCD) with the Wilson gauge action, Wilson fermions, and Hybrid Monte Carlo.

Developed by [Christian Ross](mailto:christian.ross@vanderbilt.edu) at Vanderbilt University. Independent research project.

## Overview

Lattice_QCD_v1 performs non-perturbative QCD calculations on a discrete four-dimensional Euclidean spacetime lattice. The code implements SU(3) gauge theory from scratch — all group operations, Dirac algebra, and linear solvers are written without external QCD libraries.

**Key features:**
- SU(3) gauge field generation via heat bath (Kennedy-Pendleton / Cabibbo-Marinari), overrelaxation, and Metropolis updates
- Wilson fermion propagators with CG and BiCGstab solvers
- Even/odd preconditioning for improved solver convergence
- Hybrid Monte Carlo (HMC) for dynamical fermion simulations
- Meson spectroscopy from 2-point correlators
- Polyakov loop for confinement/deconfinement detection
- Wilson loops and static quark potential extraction
- APE cooling and action density visualization
- OpenMP parallelization with checkerboard decomposition

### Simulation Phases

| Phase | Description | Status |
|-------|-------------|--------|
| Pure Gauge | Quenched SU(3) Monte Carlo, plaquette, Polyakov loop, Wilson loops | Complete |
| Wilson Fermions | Quark propagators, meson correlators, spectroscopy | Complete |
| Hybrid Monte Carlo | Dynamical fermions via pseudofermion HMC | Complete |

## Requirements

- **Compiler:** C++17 (g++ 7.0+)
- **Parallelization:** OpenMP 4.5+
- **Optional:** Python 3 with numpy/matplotlib (for analysis scripts)
- No external QCD libraries required

## Quick Start

```bash
vim run/lqcd.inp               # Configure lattice, coupling, run type
./build.sh                     # Build with parallel compilation
./run.sh                       # Run with OpenMP
```

## Input Configuration

All parameters are set in `run/lqcd.inp` — a section-based key-value file. Pre-configured input files are available in `inputs/`.

### Input Sections

| Section | Parameters |
|---------|-----------|
| `[Lattice]` | Spatial and temporal dimensions (nX, nY, nZ, nT) |
| `[Gauge Action]` | Coupling constant (beta), update method, start type |
| `[Monte Carlo]` | Thermalization sweeps, configurations, overrelaxation |
| `[Measurements]` | Wilson loops, Polyakov loop, correlator toggles |
| `[Wilson Fermions]` | Hopping parameter (kappa), solver type, tolerance |
| `[HMC]` | MD steps, step size, pseudofermion configuration |

### Example Input Files

| File | Description |
|------|-------------|
| `quenched_8x8x8x8_b56.inp` | Quenched gauge at beta = 5.6 |
| `fermion_test_4x4x4x8.inp` | Wilson fermion correlators |
| `hmc_4x4x4x4.inp` | Dynamical fermion thermalization |
| `spectro_8x8x8x16_b60_k140.inp` | Meson spectroscopy |

## Output

Results are written to auto-created subdirectories under `run/`:
- `obs_plaquette_*.dat` — Average plaquette per configuration
- `obs_polyakov_*.dat` — Polyakov loop measurements
- `config_*.dat` — Gauge field configurations (binary)
- `correlators_*.dat` — Meson 2-point functions
- `action_density_*.txt` — Per-site action density (for visualization)

## Analysis Tools

Post-processing scripts in `analysis/`:
- `plot_plaquette.py` — Plaquette thermalization and beta dependence
- `spectroscopy.py` — Meson masses from correlator fits
- `effective_mass.py` — Exponential fitting for spectroscopy
- `static_potential.py` — String tension from Wilson loops
- `jackknife_analysis.py` — Resampling error estimation
- `viz_action_density.py` — 3D visualization of gauge action density
- `viz_polyakov_phase.py` — Deconfinement phase detection
- `viz_thermalization.py` — Thermalization monitoring

## Project Structure

```
Lattice_QCD_v1/
├── src/                   # C++ source (~50 files)
│   ├── Makefile           # Build configuration (g++ + OpenMP)
│   ├── lqcd_main.cpp      # Main program and simulation dispatcher
│   ├── su3_module.*       # SU(3) matrix operations
│   ├── colorspinor_module.*    # Color-spinor field operations
│   ├── gamma_matrices.*        # Dirac algebra
│   ├── wilson_dirac_operator.cpp  # Wilson Dirac operator
│   ├── cg_solver.cpp             # Conjugate gradient solver
│   ├── bicgstab_solver.cpp       # BiCGstab solver
│   ├── hmc_driver.cpp            # Hybrid Monte Carlo driver
│   ├── interfaces_module.hpp     # Compile-time function signatures (~80 interfaces)
│   └── *.cpp              # Additional source files
├── run/                   # Runtime directory
│   └── lqcd.inp           # Input configuration
├── inputs/                # Pre-configured input files
├── analysis/              # Python post-processing scripts
├── build.sh               # Build script (parallel compilation)
├── run.sh                 # Execution script (OpenMP setup, timing)
└── theory_notes/          # Algorithm derivation notes
    ├── LQCD_algorithm_notes.tex  # Lattice QCD algorithms (LaTeX source)
    └── *.pdf              # Reference materials
```

## Physics

The code discretizes QCD on a hypercubic lattice with the Wilson gauge action:

```
S_G = beta * sum_{x,mu<nu} [1 - (1/N_c) Re Tr U_P(x,mu,nu)]
```

where `U_P` is the plaquette (product of gauge links around an elementary square) and `beta = 2N_c/g^2`.

Wilson fermions are implemented with the Dirac operator:

```
D = 1 - kappa * sum_mu [(1 - gamma_mu) U_mu(x) delta_{y,x+mu} + (1 + gamma_mu) U_mu^dag(y) delta_{y,x-mu}]
```

where `kappa` is the hopping parameter controlling the quark mass.

## Documentation

- **`theory_notes/`** — Algorithm derivation notes covering gauge field updates, Wilson fermion implementation, and HMC

## Contact

Christian Ross — [christian.ross@vanderbilt.edu](mailto:christian.ross@vanderbilt.edu)

## Author

Christian Ross, Vanderbilt University
