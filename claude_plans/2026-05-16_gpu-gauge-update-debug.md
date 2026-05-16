# Plan: Debug GPU Gauge-Update Dynamics

**Date**: 2026-05-16
**Project**: Lattice_QCD_v1
**Branch**: gpu-validation-comphys (or successor)
**Parent commit**: see git log

## Context

GPU port built + validated on comphys 2026-05-16 (HPC SDK 24.11, Quadro
RTX 4000, cc75). Outcome:

- **Build**: fixed (nvc++ localrc self-heal in `build.sh`).
- **GPU fermion stack**: CORRECT. RNG-free CPU-vs-GPU gate (free-field and
  a bit-identical loaded thermalized config, both `<P>=0.5700938166`): all
  8 meson correlators agree to ~1e-10 abs / 4e-11 rel, inside the 1e-8 gate.
- **GPU gauge update + HMC**: BROKEN. GPU quenched heatbath equilibrium
  `<P>=0.540` vs CPU `0.427` at beta=5.6 (≫60σ). GPU HMC: one-sign ΔH,
  plaquette runs away to ~0.79, 100% accept. At 5× smaller MD step ΔH→0
  with good Creutz **but plaquette still runs away** → GPU MD is
  self-consistent for the *wrong* action.

## Localization (already done)

- GPU plaquette *measurement* is bit-exact on a fixed loaded config
  (matched CPU `0.5700938166`). So measurement, Dirac, solver, propagator,
  correlators are NOT the bug.
- The bug is in GPU **gauge-update dynamics**: it equilibrates to the wrong
  Boltzmann distribution. HMC failure is downstream of the same root cause
  (HMC self-consistent but wrong action ⇒ same gauge-action/force/beta
  defect as heatbath, or a shared SU(3)/staple kernel).

## Next Phase: Goals

Find and fix the GPU gauge-action / gauge-update defect so GPU quenched
heatbath equilibrium plaquette matches CPU within statistics, then re-check
GPU HMC (⟨exp−ΔH⟩≈1 AND correct equilibrium plaquette).

## Implementation Steps

1. Branch is already set; build both on comphys (`build.sh cpu` → preserve
   `run/lqcd_cpu_preserved.exe`; `build.sh gpu`). The `.nvhpc/localrc`
   self-heal is automatic.
2. **Isolate the gauge sector RNG-free.** Load the fixed config
   (`Start Type: file:.../fixedcfg.bin`), do exactly ONE gauge update
   step, and compare the resulting plaquette / a few links CPU vs GPU.
   Even one heatbath/overrelaxation sweep diverges by RNG, so instead
   compare the **staple** and **gauge action/force** on the loaded config
   directly (deterministic): instrument or add a debug print of
   `compute_staple`/`gauge_force` for a fixed link on CPU vs GPU. This is
   the RNG-free probe analogous to the fermion gate.
3. Prime suspects in order: `src/gpu/gauge_force_gpu.cu` (beta/Nc
   prefactor, TA projection sign), `src/gpu/gauge_ops_gpu.cu` (staple
   construction, plaquette orientation), `src/gpu/heat_bath_gpu.cu`
   (Cabibbo-Marinari SU(2) subgroup extraction, Kennedy-Pendleton
   acceptance, beta→alpha mapping), `src/gpu/su3_exp_gpu.cu` (HMC link
   update). Compare each against the CPU counterpart numerically on the
   fixed config.
4. A wrong equilibrium that is *higher* than CPU ≈ effectively weaker
   coupling ⇒ check beta normalization (`beta/Nc` vs `beta`, `2*Nc/g^2`),
   and the staple sum count (6 staples per link, correct mu≠nu set).
5. Fix, rebuild, re-run `inputs/quenched_4x4x4x4_b56.inp` GPU vs CPU
   production plaquette until they agree within ~2σ. Then re-run
   `inputs/hmc_4x4x4x4.inp` GPU and require ⟨exp−ΔH⟩≈1 *and* equilibrium
   ⟨P⟩ matching CPU.
6. Commit fix; update `project_status.md` memory and CLAUDE.md pitfall #6.

## Test Harness (in place)

- `inputs/gen_fixedcfg_4x4x4x8.inp` — CPU generates a thermalized config
  (seed 42), saved binary; copy to `run/fixedcfg.bin`.
- `inputs/rngfree_fixedcfg_4x4x4x8.inp` — loads it via `Start Type:
  file:/home/ross/Lattice_QCD_v1/run/fixedcfg.bin`, RNG-free.
- `inputs/rngfree_freefield_4x4x4x8.inp` — cold, 0 sweeps, free-field gate.
- Comparison recipe: run `lqcd_cpu_preserved.exe` then `lqcd_gpu.exe` into
  separate dirs, diff `correlator_*.dat` / `plaquette.dat` with numpy.

## Things to Watch Out For

- CPU/GPU RNG streams differ — never gate on RNG-path observables. Use the
  fixed-config / direct-kernel comparisons.
- Clover HMC force is separately broken (CLAUDE.md pitfall #1) — out of
  scope here; debug Wilson gauge update first.
- `build.sh gpu` cleans both exes on target switch — preserve the CPU exe.
- cluster-ops auto-loads now (`.claude-remote.yml` committed): comphys,
  `/home/ross/Lattice_QCD_v1`, GPU build cycle, full autonomy in-tree.

## References

- Project CLAUDE.md pitfalls #1 (clover), #6 (GPU gauge update), #7 (nvc++)
- Memory: `project_status.md` (May 2026), `reference_compute.md`
- Validated-correct path: GPU fermion stack (gate passed this session)
