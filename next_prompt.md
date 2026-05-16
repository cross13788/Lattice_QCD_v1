# Next Session Prompt

**Resume work in**: `/home/ross/Lattice_QCD_v1`
**Plan**: `claude_plans/2026-05-16_cluster-gpu-validation.md` — read this first.

## TL;DR

The CUDA GPU port has never been compiled or run anywhere. Next session:
build it on `comphys` with `./build.sh gpu`, then prove correctness against the
CPU via an RNG-free fixed-configuration solver test. Clover HMC force stays
broken and out of scope.

## Current State

- **Branch**: master @ 86340a1
- **Working tree**: clean (run artifacts now gitignored)
- **Last session**: planning only — reviewed state, wrote the validation plan,
  committed plan + .gitignore.

## Key Findings to Carry Forward

- `build.sh gpu` uses **nvc++ for everything, no nvcc**. The `project_status`
  memory ("nvcc for .cu files, sm_75") is outdated — correct it after this round.
- CPU/GPU RNG streams differ → Monte-Carlo observables are NOT bitwise
  comparable. Correctness gate must be the fixed-config propagator test.
- `run/lqcd.inp` is a clover gradient-check leftover (Clover 1.0, MD Steps 1,
  Trajectories 0). Do NOT deploy it as the GPU validation input.
- comphys is the only cluster with the NVIDIA HPC SDK. CPU cluster (nuclps)
  cannot build GPU.

## Immediate Next Steps

1. Confirm comphys SSH access and the NVIDIA HPC SDK module load command
   (blocking — needed before `./build.sh gpu`).
2. Generate local CPU reference output for `inputs/quick_test_4x4x4x4.inp`,
   `inputs/fermion_test_4x4x4x8.inp`, `inputs/hmc_4x4x4x4.inp`.
3. `./deploy.sh comphys`, then on-node `./build.sh gpu` (capture log + arch),
   fix build errors iteratively.
4. Run the fixed-config CPU-vs-GPU propagator test — gate at CG tol 1e-8.

## Watch Out For

- Likely GPU build breaks: `thrust::complex` ABI, cuRAND link, `--gcc-toolchain`,
  sm-arch mismatch.
- Don't trust any clover HMC dynamics — force is broken (CLAUDE.md pitfall #1).
- Don't compare CPU vs GPU plaquette trajectories for correctness (RNG diverges).

## Before You Start

- [ ] Read `claude_plans/2026-05-16_cluster-gpu-validation.md` in full
- [ ] Verify `./build.sh cpu` still compiles cleanly locally (baseline)
- [ ] Confirm comphys NVIDIA HPC SDK module name
