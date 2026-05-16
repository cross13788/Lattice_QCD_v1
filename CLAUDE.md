# Lattice-QCD-v1 -- Expert Consultant Reference

## Project Identity

From-scratch C++17 implementation of SU(3) lattice QCD: Wilson gauge action, Wilson fermions (unimproved + clover), heat bath (Kennedy-Pendleton / Cabibbo-Marinari) + overrelaxation, Hybrid Monte Carlo with pseudofermions, CG/BiCGstab solvers with even-odd preconditioning, meson spectroscopy, Wilson flow, and static quark potential. Written by Christian Ross at Vanderbilt University under Prof. A.S. Umar (DOE DE-SC0013847). ~60 CPU source files + ~27 GPU (CUDA) files. No external QCD libraries -- everything from scratch.

**Status**: Production for quenched gauge + Wilson fermion spectroscopy + quenched HMC. Clover fermion force for HMC is partially implemented (structural framework + gradient check tool in place; leaf-specific staple decomposition needs debugging — see `gradient_check.cpp`). Quenched clover spectroscopy works. **GPU port: fermion stack validated, gauge-update/HMC broken** (built+validated on comphys 2026-05-16). The GPU Wilson-Dirac/solver/propagator/correlator path agrees with the CPU to ~1e-10 on a fixed config (RNG-free gate, well inside 1e-8). The GPU gauge-update dynamics (heatbath + HMC) sample the wrong distribution and are NOT usable; use GPU only for fixed-config fermion spectroscopy on CPU-generated configs (`Start Type: file:<path>`). CPU validated on lattices up to 8^3x16.

---

## Theoretical Foundations

### Lattice Gauge Theory

QCD is discretized on a 4D Euclidean spacetime lattice with spacing a (Wilson, Phys. Rev. D 10, 2445, 1974). Gauge fields live on **links** as SU(3) matrices U_mu(x), not at sites. This preserves exact gauge invariance at finite lattice spacing -- the key insight of Wilson's formulation.

### Wilson Gauge Action

The fundamental gauge-invariant object is the **plaquette** (Gattringer & Lang, Ch. 3):
```
P_{mu,nu}(x) = U_mu(x) * U_nu(x+mu) * U_mu^dag(x+nu) * U_nu^dag(x)
```

The Wilson gauge action (Wilson 1974):
```
S_G = beta * sum_{x, mu<nu} (1 - (1/3) Re Tr P_{mu,nu}(x))
```
where beta = 2*Nc/g^2 = 6/g^2 for SU(3). In the continuum limit (a -> 0), this recovers the Yang-Mills action S_YM = (1/4) integral F_{mu,nu}^a F_{mu,nu}^a d^4x.

### Wilson Fermions

The naive lattice Dirac operator produces 2^4 = 16 fermion species (doublers) due to extra zeros at Brillouin zone corners (Nielsen-Ninomiya no-go theorem, Phys. Lett. B 105, 219, 1981). Wilson's fix adds a dimension-5 Laplacian term:

**Wilson-Dirac operator** (Gattringer & Lang, Ch. 5):
```
D_W = (1/2) sum_mu { gamma_mu (nabla*_mu + nabla_mu) - a * nabla*_mu nabla_mu }
```

In hopping parameter form: D = (1/2kappa)*I - D_hop, where kappa = 1/(2(am_0 + 4)) is the hopping parameter. The critical kappa_c corresponds to zero physical quark mass.

**Clover improvement** (Sheikholeslami & Wohlert, Nucl. Phys. B 259, 572, 1985): Adds the clover term i*(kappa*c_SW/4)*sigma_{mu,nu}*F_{mu,nu} to remove O(a) discretization errors. The field strength F_{mu,nu} is computed from the "clover leaf" average of 4 plaquettes.

### Monte Carlo Methods

**Heat bath** (Cabibbo & Marinari, Phys. Lett. B 119, 387, 1982): SU(3) link update via 3 successive SU(2) subgroup updates. Each SU(2) update uses the Kennedy-Pendleton algorithm (Phys. Lett. B 156, 393, 1985) to sample from P(a_0) ~ sqrt(1-a_0^2)*exp(2*alpha*a_0).

**Overrelaxation** (Brown & Woch, Phys. Rev. Lett. 58, 2394, 1987): Microcanonical update that preserves the action while decorrelating configurations. Reduces autocorrelation time by ~3x. Not ergodic alone; combined with heat bath sweeps.

### Hybrid Monte Carlo

For dynamical fermions (Duane, Kennedy, Pendleton, Roweth, Phys. Lett. B 195, 216, 1987):

**Pseudofermion trick**: det(D^dag D) = integral exp(-phi^dag (D^dag D)^{-1} phi). Generate phi = D^dag * xi with Gaussian xi.

**HMC Hamiltonian** (Luscher, arXiv:1002.4232, Sec. 1.4):
```
H = (1/2) sum pi^2 + S_G[U] + phi^dag (D^dag D)^{-1} phi
```

**Leapfrog integrator**: Half-step momenta -> full-step gauge field -> half-step momenta. Metropolis accept/reject on exp(-Delta H) ensures exact sampling despite finite step size. Target acceptance: 70-90%.

**Fermion force**: Requires CG inversion X = (D^dag D)^{-1} phi at each MD step -- the computational bottleneck.

### Observables

**Polyakov loop** (confinement order parameter): Product of temporal links wrapping periodic time direction. <P> = 0 in confined phase, <P> != 0 in deconfined phase.

**Static quark potential**: From Wilson loops W(R,T): V(R) = -ln(W(R,T)/W(R,T-1)). At large R: V(R) = -C/R + sigma*R (Cornell potential). Linear rise = confinement, sigma ~ (440 MeV)^2.

**Meson correlators**: C(t) = sum_x <O_M(x,t) O_M^dag(0)> with O_M = psi_bar*Gamma*psi. Effective mass from plateau: m_eff(t) = ln(C(t)/C(t+1)). Channels: pion (gamma_5), rho (gamma_i), scalar (I), a1 (gamma_5*gamma_i).

### Key References

**Foundational:**
- K.G. Wilson, Phys. Rev. D 10, 2445 (1974) -- lattice gauge theory
- H.B. Nielsen, M. Ninomiya, Phys. Lett. B 105, 219 (1981) -- fermion doubling no-go theorem

**Textbooks:**
- C. Gattringer, C.B. Lang, "Quantum Chromodynamics on the Lattice" (Springer, 2010) -- Ch. 3 (gauge action), Ch. 5 (Wilson fermions), Ch. 8 (HMC)
- H.J. Rothe, "Lattice Gauge Theories," 4th ed. (World Scientific, 2012) -- Monte Carlo, fermion actions
- T. DeGrand, C. DeTar, "Lattice Methods for QCD" (World Scientific, 2006) -- spectroscopy
- M. Creutz, "Quarks, Gluons and Lattices" (Cambridge, 1983) -- the classic reference

**Algorithms:**
- N. Cabibbo, E. Marinari, Phys. Lett. B 119, 387 (1982) -- SU(3) heat bath
- A.D. Kennedy, B.J. Pendleton, Phys. Lett. B 156, 393 (1985) -- improved SU(2) heat bath
- F.R. Brown, T.J. Woch, Phys. Rev. Lett. 58, 2394 (1987) -- overrelaxation
- S. Duane et al., Phys. Lett. B 195, 216 (1987) -- HMC
- B. Sheikholeslami, R. Wohlert, Nucl. Phys. B 259, 572 (1985) -- clover improvement

**Lectures:**
- M. Luscher, arXiv:1002.4232 (2010) -- Sec. 1.4-1.5 (HMC), Sec. 2.2 (solvers, even-odd)

---

## Numerical Methods

### SU(3) Matrix Operations
Plain struct `SU3matrix` with 3x3 complex array. Explicit multiply (unrolled over color). Reunitarization via modified Gram-Schmidt + cross product for det=+1. Random generation via 3 SU(2) subgroup rotations (quaternion parameterization). Thread-local `std::mt19937_64` RNG.

### Heat Bath + Overrelaxation
Checkerboard decomposition (even/odd parity). Each site updates all 4 link directions. Cabibbo-Marinari cycles through 3 SU(2) subgroups per link. Kennedy-Pendleton rejection sampling. Typical: 1 heat bath + 3-5 overrelaxation sweeps per update cycle.

### CG and BiCGstab Solvers
- **CG**: Solves D^dag D x = D^dag b (normal equations). Hermitian positive definite.
- **BiCGstab**: Solves D x = b directly (non-Hermitian). Default solver.
- **Even-odd preconditioning**: Schur complement D_hat = 1 - D_eo*D_oe. Halves iteration count. Clover even-odd uses 6x6 LU decomposition for (1+A)^{-1}.

### HMC Implementation
Matrix exponential via 12-term Taylor/Horner + reunitarization. Leapfrog integrator. Fermion force from CG inversion at each MD step. **Clover fermion force is partially implemented** -- structural framework with gradient check tool; leaf-specific staple decomposition needs debugging. Wilson HMC works correctly. A `gradient_check.cpp` tool validates the force numerically.

### Wilson Flow
Luscher's 3-stage RK3 integrator. Both plaquette and clover energy density definitions. Reversibility test included.

### Known Numerical Pitfalls

1. **Clover HMC force is partially implemented**: The zero-force stub has been replaced with a structural implementation (`clover_fermion_force.cpp`) using non-Hermitian Lambda, left/right staple decomposition per leaf, and TA projection. A numerical gradient check (`gradient_check.cpp`) validates the Wilson force perfectly (ratio = 1.0 ± 1e-5) but shows the clover force has the right magnitude but incorrect leaf-specific staple index structure — the 8 leaf contributions (4 upper + 4 lower per mu-nu pair) need their left/right matrix splitting debugged against the gradient check. Use `gradient_check.cpp` to systematically test each leaf. The clover force requires ~5x smaller MD step sizes than Wilson (known property of clover actions). For now, use quenched clover for spectroscopy (unaffected by force).
2. **Thermalization**: Cold start needs fewer sweeps than hot start. Monitor plaquette for plateau.
3. **Critical slowing down**: Near kappa_c, CG iteration count diverges as condition number ~ (am)^{-2}.
4. **Overrelaxation alone is not ergodic**: Must combine with heat bath sweeps.
5. **Reunitarization**: Accumulation of roundoff in SU(3) multiplication requires periodic reunitarization.
6. **GPU gauge-update dynamics are broken (validated 2026-05-16, comphys)**: GPU quenched heatbath equilibrates to the wrong plaquette (`<P>=0.540` vs CPU `0.427` at beta=5.6, ≫60σ); GPU HMC plaquette runs away with one-sign ΔH. At 5× smaller MD step ΔH→0 with good Creutz but the plaquette still runs away → the GPU MD is self-consistent for the *wrong* action. The GPU fermion stack (Dirac/solver/propagator/correlator) is correct — it agrees with the CPU to ~1e-10 on a fixed loaded config. So the bug is isolated to the GPU **gauge-update dynamics** (suspect `src/gpu/heat_bath_gpu.cu`, `gauge_ops_gpu.cu`, `gauge_force_gpu.cu`, or beta application), NOT measurement, Dirac, or solver. Use GPU only for fixed-config fermion spectroscopy via `Start Type: file:<path>` on CPU-generated configs. The RNG-free gate is the test harness: `inputs/rngfree_fixedcfg_4x4x4x8.inp` + `inputs/gen_fixedcfg_4x4x4x8.inp`.
7. **nvc++ localrc / host-GCC skew**: HPC SDK `localrc` is pinned to the GCC present at SDK-install time. A host GCC upgrade breaks `build.sh gpu` with `limits.h: no directories in search list`. `build.sh gpu` self-heals by regenerating a project-local `.nvhpc/localrc` via `makelocalrc` and passing `-rc=`. If the GPU build dies on header search right after a host upgrade, this is why.

---

## Code Architecture

### Directory Structure
```
Lattice_QCD_v1/
├── src/                # ~60 C++ source files
│   └── gpu/            # ~27 CUDA source files
├── run/                # Executables, output directories
├── inputs/             # Pre-configured .inp files
├── analysis/           # Python scripts (spectroscopy, thermalization, correlators)
├── theory_notes/       # Reference PDFs (Luscher, KEK lectures)
├── build.sh, run.sh, deploy.sh, remote_run.sh
```

### Source File Map

#### Core Data Types
| File | Purpose |
|------|---------|
| `su3_module.hpp/.cpp` | SU3matrix struct, multiply, adjoint, trace, reunitarize, random, heat bath, overrelaxation |
| `colorspinor_module.hpp/.cpp` | ColorSpinor struct (4 Dirac x 3 color), field-level operations |
| `gamma_matrices.hpp/.cpp` | Sparse gamma matrices (DeGrand-Rossi basis) |
| `lattice_module.hpp/.cpp` | Site indexing, neighbor tables, periodic BCs |
| `clover_module.hpp` | CloverBlock (6x6), CloverField structs |

#### Gauge Sector
| File | Purpose |
|------|---------|
| `compute_plaquette.cpp` | Average plaquette and action density |
| `compute_staple.cpp` | Staple sum for link updates |
| `heat_bath_sweep.cpp` | Checkerboard heat bath sweep (OMP parallel) |
| `overrelaxation_sweep.cpp` | Microcanonical overrelaxation sweep |
| `metropolis_sweep.cpp` | Metropolis accept/reject gauge update |
| `gauge_configuration_io.cpp` | Save/load configurations (binary) |

#### Fermion Sector
| File | Purpose |
|------|---------|
| `wilson_dirac_operator.cpp` | Apply Wilson D to ColorSpinor field |
| `clover_dirac_operator.cpp` | Apply clover-improved D |
| `clover_field.cpp` | Compute clover F_{mu,nu} from plaquettes |
| `cg_solver.cpp` | CG for D^dag D |
| `bicgstab_solver.cpp` | BiCGstab for D |
| `even_odd_preconditioning.cpp` | Schur complement preconditioning |
| `clover_even_odd.cpp` | Clover even-odd with 6x6 LU |
| `quark_propagator.cpp` | Full propagator (12 inversions) |
| `meson_correlators.cpp` | 2-point functions for 8 meson channels |
| `effective_mass.cpp` | m_eff extraction + jackknife errors |

#### HMC
| File | Purpose |
|------|---------|
| `hmc_driver.cpp` | Outer loop: thermalization + production |
| `hmc_trajectory.cpp` | Full trajectory: save U, momenta, pseudofermion, MD, Metropolis |
| `leapfrog_integrator.cpp` | Verlet: half-step pi, full-step U, half-step pi |
| `conjugate_momentum.cpp` | Gaussian su(3) momenta, kinetic energy |
| `exp_su3_algebra.cpp` | Matrix exp (Taylor/Horner), TA projection |
| `generate_pseudofermion.cpp` | phi = D^dag * xi |
| `pseudofermion_action.cpp` | S_PF = phi^dag (D^dag D)^{-1} phi |
| `gauge_force.cpp` | F_G = (beta/Nc) * [U*staple]_TA |
| `fermion_force.cpp` | F_F from CG inversion + clover force (partially implemented) |
| `gradient_check.cpp` | Numerical gradient validation for fermion force |

#### Observables
| File | Purpose |
|------|---------|
| `polyakov_loop.cpp` | Confinement order parameter |
| `wilson_loops.cpp` | W(R,T) for static potential |
| `static_potential.cpp` | V(R) extraction from Wilson loops |
| `wilson_flow.cpp` | Gradient flow with RK3 integrator |
| `field_strength_tensor.cpp` | Clover F_{mu,nu} for flow and improvement |

---

## Input/Output

### Input (section-based key:value)
```
[Lattice]          nX, nY, nZ, nT
[Gauge]            beta, startType (cold/hot/file), gaugeUpdateMethod (heatbath/hmc)
[Fermion]          kappa, useClover, cSW, solverType (cg/bicgstab), cgTolerance
[Monte Carlo]      nTherm, nConfigs, nSweepsBetween, nOverrelax
[HMC]              nMDSteps, mdStepSize, hmcTrajectories
[Measurements]     measureWilsonLoops, measureCorrelators, measurePolyakov, measureFlow
```

### Pre-configured Inputs
| File | Description |
|------|-------------|
| `quick_test_4x4x4x4.inp` | Fast validation: 4^4, beta=6.0, heat bath |
| `quenched_8x8x8x16.inp` | Production quenched: 8^3x16, Wilson loops on |
| `fermion_test_4x4x4x8.inp` | Spectroscopy: kappa=0.15, BiCGstab |
| `hmc_4x4x4x4.inp` | HMC test: beta=5.6, kappa=0.12, 10 MD steps |

### Build
```bash
./build.sh cpu    # g++ with OpenMP
./build.sh gpu    # nvc++ with CUDA
```

### Code Conventions
- C++17, `real_t = double`, `complex_t = std::complex<double>`
- Plain structs + free functions (no OOP)
- snake_case functions, physics-conventional short names
- One concept per source file
- OpenMP on outer loops with checkerboard for gauge updates

---

## Validation and Benchmarks

### Quenched Gauge
- Plaquette vs beta validated at multiple beta values (5.6-6.2)
- Polyakov loop deconfinement transition observed
- Static potential V(R) from Wilson loops shows linear confinement + Coulomb

### Spectroscopy
- Pion effective mass extracted at kappa = 0.120, 0.140, 0.150, 0.154
- Jackknife error estimation implemented

### HMC
- Thermalization from hot and cold starts verified (plaquette convergence)
- Acceptance rate tested across MD step sizes

---

## Research Context

### Where This Code Fits

| Code | Scale | Key Feature |
|------|-------|-------------|
| **This code** | Educational/research | From-scratch, everything visible, GPU port |
| **QUDA** (lattice community) | Production | Highly optimized GPU library |
| **openQCD** (Luscher) | Production | Deflated solvers, DD-HMC |
| **CL2QCD** | Research | OpenCL implementation |

### Relationship to Functional-QCD-v1
Lattice QCD and DSE/BSE are **complementary approaches to the same QCD**:
- Lattice: discretize spacetime, Monte Carlo sample path integral. Exact (up to discretization + finite volume). Limited to Euclidean space.
- DSE/BSE: solve continuum equations of motion. Requires truncation. Access to timelike domain.
- Cross-validation: gluon propagator D(k^2), quark mass function M(p^2), meson masses should agree.

### Open Problems
1. **Clover HMC**: Fermion force stub needs implementation for dynamical clover simulations
2. **Staggered/domain-wall fermions**: Alternative fermion discretizations not implemented
3. **Scale setting**: Sommer parameter r_0 or Wilson flow t_0 for physical units
4. **Finite temperature**: Polyakov loop transition needs systematic T scan
5. **B-spline lattice fermions**: Novel research direction being explored

---

## Cross-Project Connections

### Shared with Other QFT Codes
- **Functional-QCD-v1**: Same QCD, different method. Compare gluon propagator, quark mass function, meson masses.
- **Coding conventions**: Same C++17 procedural style, one-file-per-concept, OpenMP

### Physics Connections
- Lattice gluon propagator data can inform the "lattice-informed" effective interaction model in Functional-QCD-v1
- Meson masses from lattice spectroscopy validate DSE/BSE results
- Both probe nonperturbative QCD from different angles

---

## Expert Consultant Directive

When acting as domain expert for this project:
- **Be blunt and direct.** If the user confuses lattice conventions, say so.
- **beta = 6/g^2 for SU(3).** Not 2*Nc/g^2 with Nc left symbolic. beta=6.0 means g^2=1.
- **The clover HMC force is partially implemented.** The structural framework is in place but the leaf-specific staple decomposition needs debugging. Use `gradient_check.cpp` (runs automatically with `Verbose Output: on` and `Clover Coefficient > 0`) to validate. Wilson HMC is verified correct. Quenched clover spectroscopy works regardless of force status.
- **Fermion doubling is fundamental.** Wilson's fix (the r-term) adds O(a) errors that clover removes. You cannot have chiral symmetry + locality + no doubling on the lattice (Nielsen-Ninomiya).
- **kappa_c is NOT 1/8.** It depends on the gauge coupling and is determined non-perturbatively. At beta=6.0, kappa_c ~ 0.157 for unimproved Wilson.
- **Thermalization matters.** Always check plaquette vs sweep number. Discard pre-thermalization configurations.
- **Autocorrelation matters.** Consecutive configurations are correlated. Space measurements by several update cycles.
- **Overrelaxation is not ergodic.** Must combine with heat bath. Pure overrelaxation gets stuck.

---

## Sources

### Foundational Papers
- K.G. Wilson, Phys. Rev. D 10, 2445 (1974) -- lattice gauge theory
- H.B. Nielsen, M. Ninomiya, Phys. Lett. B 105, 219 (1981) -- fermion doubling theorem
- M. Creutz, Phys. Rev. D 21, 2308 (1980) -- first numerical Wilson loop calculations

### Textbooks
- C. Gattringer, C.B. Lang, "Quantum Chromodynamics on the Lattice" (Springer, 2010) -- Ch. 3 (gauge), Ch. 5 (fermions), Ch. 8 (HMC)
- H.J. Rothe, "Lattice Gauge Theories," 4th ed. (World Scientific, 2012) -- Monte Carlo, fermion actions
- T. DeGrand, C. DeTar, "Lattice Methods for QCD" (World Scientific, 2006) -- spectroscopy, connecting to phenomenology
- M. Creutz, "Quarks, Gluons and Lattices" (Cambridge, 1983) -- the classic

### Algorithm Papers
- N. Cabibbo, E. Marinari, Phys. Lett. B 119, 387 (1982) -- SU(3) heat bath via SU(2) subgroups
- A.D. Kennedy, B.J. Pendleton, Phys. Lett. B 156, 393 (1985) -- improved SU(2) heat bath
- F.R. Brown, T.J. Woch, Phys. Rev. Lett. 58, 2394 (1987) -- overrelaxation for SU(3)
- S. Duane, A.D. Kennedy, B.J. Pendleton, D. Roweth, Phys. Lett. B 195, 216 (1987) -- HMC
- B. Sheikholeslami, R. Wohlert, Nucl. Phys. B 259, 572 (1985) -- clover improvement
- H.A. van der Vorst, SIAM J. Sci. Stat. Comput. 13, 631 (1992) -- BiCGstab

### Lectures/Reviews
- M. Luscher, arXiv:1002.4232 (2010) -- Sec. 1.4-1.5: HMC; Sec. 2.2: solvers and preconditioning
- F. Knechtli, arXiv:1706.00282 (2017) -- modern lattice QCD introduction
