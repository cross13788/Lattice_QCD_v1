# Next Session Prompt

**Resume work in**: `/home/ross/Lattice_QCD_v1`
**Plan**: `claude_plans/2026-05-16_gpu-gauge-update-debug.md` — read first.

## TL;DR

GPU port was built + validated on comphys (2026-05-16). Build fixed; GPU
fermion stack proven correct (RNG-free gate, ~1e-10). **GPU gauge-update
dynamics are broken** — heatbath/HMC sample the wrong distribution.
Next session: find and fix the GPU gauge-action/update defect.

## Current State

- **Branch**: gpu-validation-comphys (committed). Tree clean.
- cluster-ops auto-loads: comphys, `/home/ross/Lattice_QCD_v1`, GPU cycle.
- comphys: HPC SDK 24.11, Quadro RTX 4000 cc75. nvc++ on PATH.

## Validated This Session

- **Build FIXED**: `build.sh gpu` self-heals nvc++ localrc vs host GCC
  skew (regenerates project-local `.nvhpc/localrc`, passes `-rc=`).
- **GPU fermion stack CORRECT**: new `Start Type: file:<path>` wiring;
  RNG-free CPU-vs-GPU on free-field + loaded thermalized config — all 8
  correlators agree to ~1e-10 (gate 1e-8). PRIMARY GATE PASSED.

## The Bug to Fix

GPU quenched heatbath `<P>=0.540` vs CPU `0.427` at beta=5.6 (≫60σ). GPU
HMC plaquette runs away (→0.79); at 5× smaller step ΔH→0 with good Creutz
but plaquette still runs away ⇒ GPU MD self-consistent for the *wrong*
action. GPU measurement/Dirac/solver are bit-exact on a fixed config, so
the defect is isolated to GPU **gauge-update dynamics**.

## Immediate Next Steps

1. On comphys: build CPU (preserve `run/lqcd_cpu_preserved.exe`), build GPU.
2. RNG-free probe: instrument `compute_staple` / `gauge_force` on the fixed
   loaded config (`run/fixedcfg.bin`), compare CPU vs GPU for one link.
3. Suspects in order: `gpu/gauge_force_gpu.cu` (beta/Nc prefactor, TA
   sign), `gpu/gauge_ops_gpu.cu` (staple/plaquette), `gpu/heat_bath_gpu.cu`
   (CM SU(2) subgroups, KP acceptance, beta→alpha), `gpu/su3_exp_gpu.cu`.
4. Higher-than-CPU plaquette ≈ weaker effective coupling → check beta
   normalization first.
5. Gate: GPU vs CPU `quenched_4x4x4x4_b56.inp` production `<P>` within ~2σ,
   then GPU `hmc_4x4x4x4.inp` ⟨exp−ΔH⟩≈1 AND correct equilibrium `<P>`.

## Before You Start

- [ ] Read `claude_plans/2026-05-16_gpu-gauge-update-debug.md`
- [ ] Read CLAUDE.md pitfalls #1 (clover, out of scope), #6 (GPU gauge), #7
- [ ] Confirm cluster-ops auto-loaded (`.claude-remote.yml` present)
- [ ] `git log -1` — confirm on the gpu-validation-comphys branch

## Watch Out For

- Never gate on RNG-path observables (CPU/GPU RNG streams differ).
- `build.sh gpu` cleans both exes on target switch — preserve the CPU exe.
- Clover HMC force separately broken (pitfall #1) — Wilson gauge first.
