# Plan: Cluster GPU Validation on comphys

**Date**: 2026-05-16
**Project**: Lattice_QCD_v1
**Branch**: master
**Parent commit**: 86340a1

## Context

The CPU code is production (validated to 8³×16; last CPU cluster run Feb 27,
2026). The CUDA GPU port was added Mar 18, 2026 and **has never been compiled or
run anywhere** — this is the largest untested gap. This phase builds and
correctness-validates the GPU port on `comphys` (the only cluster with the
NVIDIA HPC SDK / `nvc++`). Clover HMC fermion force remains broken and is
explicitly out of scope; quenched clover spectroscopy is unaffected.

## What Was Accomplished This Session

- Reviewed full project state: CLAUDE.md, README.md, git history, memory files,
  deploy/remote_run/build scripts, Feb 27 cluster log.
- Identified the GPU port as never-built — the focus of the next round.
- Confirmed `build.sh gpu` now uses **nvc++ for everything, no nvcc** (the
  `project_status` memory saying "nvcc for .cu files, sm_75" is outdated).
- Wrote `CLUSTER_GPU_VALIDATION_PLAN.md` (project root) — the detailed plan.
- Extended `.gitignore` to cover run/ outputs, caches, LaTeX build artifacts.
- Committed plan + gitignore + the clover-gradient-check `lqcd.inp` (86340a1).

## What Didn't Work / Dead Ends

- N/A this session (planning only). Note from prior session: clover HMC force
  gradient check shows correct magnitude but wrong leaf-specific staple index
  structure — not addressed here.

## Next Phase: Goals

1. Compile the CUDA port on comphys with `./build.sh gpu`.
2. Prove GPU correctness vs CPU via an RNG-free fixed-configuration solver test.
3. Verify HMC reversibility on GPU.

## Implementation Steps

1. **Pre-flight (local).** Confirm comphys SSH and the NVIDIA HPC SDK module
   name/load command. Note: current `run/lqcd.inp` is a clover gradient-check
   config (Clover Coefficient 1.0, MD Steps 1, Trajectories 0, CG tol 1e-12) —
   **do not deploy as-is for GPU validation**; use the inputs below instead.
2. Generate CPU reference output locally for three inputs covering all paths:
   `inputs/quick_test_4x4x4x4.inp` (gauge), `inputs/fermion_test_4x4x4x8.inp`
   (solver/propagator), `inputs/hmc_4x4x4x4.inp` (HMC). Record plaquette,
   correlators, HMC ΔH/acceptance.
3. `./deploy.sh comphys` (rsync only — excludes .o/.last_target/artifacts).
4. On comphys: load HPC SDK, `./build.sh gpu`. build.sh auto-detects GPU arch
   from `nvidia-smi` (`ccXX`) and the GCC toolchain for nvc++. Capture full
   build log + detected arch. Fix iteratively; record fixes for a commit.
5. Also `./build.sh cpu` on comphys for an apples-to-apples comparison host.
6. **Primary correctness gate (RNG-free):** load one fixed gauge configuration,
   invert the same source on CPU and GPU, require propagator agreement to CG
   tolerance (1e-8). This is the decision gate — it does not depend on RNG.
7. Secondary: meson correlators from the fixed config (agree to solver tol);
   plaquette per config only as a statistical sanity check (CPU/GPU RNG streams
   differ — not bitwise comparable).
8. HMC reversibility test on GPU; ΔH distribution + acceptance statistically
   consistent with CPU.
9. Commit GPU build fixes. Update memory (`project_status.md`: nvc++-only build,
   GPU validation outcome, comphys GPU arch). Update CLAUDE.md GPU section if
   the build invocation changed.

## Open Questions

- **(blocking)** NVIDIA HPC SDK module name / load command on comphys —
  required before `./build.sh gpu`.
- **(deferrable)** comphys GPU compute capability — discovered on-node via
  `nvidia-smi`; affects `ccXX` and any `sm_75+` kernel assumptions.
- **(deferrable)** Whether to add `.claude-remote.yml` so the `cluster-ops`
  skill auto-loads for future comphys cycles (per `feedback_cluster_ops_allowed`).

## Things to Watch Out For

- **RNG divergence CPU vs GPU** makes Monte-Carlo-path observables
  non-bitwise-comparable. Anchor correctness on the fixed-config solver test
  (step 6), not on plaquette/Polyakov trajectories.
- Likely GPU build failures: `thrust::complex` vs `std::complex` ABI, cuRAND
  link flags, `--gcc-toolchain` path, sm-arch mismatch.
- `run/lqcd.inp` is a clover gradient-check leftover — overwrite it via
  `remote_run.sh <cluster> <input_file>` rather than deploying it directly.
- CLAUDE.md "Known Numerical Pitfalls": clover HMC force still broken — use
  quenched clover only; do not interpret clover HMC results as correct.
- Never deploy/run with the broken clover force expecting valid dynamics.

## Validation Checklist

- [ ] `./build.sh cpu` compiles cleanly locally (baseline still good)
- [ ] `./build.sh gpu` compiles cleanly on comphys
- [ ] Fixed-config CPU vs GPU propagator agrees to CG tol (1e-8) — **gate**
- [ ] GPU HMC reversibility test passes
- [ ] GPU correlators agree with CPU to solver tolerance
- [ ] Memory `project_status.md` corrected (nvc++-only, GPU outcome)

## References

- Project CLAUDE.md: `/home/ross/Lattice_QCD_v1/CLAUDE.md` (GPU port; Known
  Numerical Pitfalls #1 clover force)
- Detailed plan: `CLUSTER_GPU_VALIDATION_PLAN.md` (project root)
- Scripts: `deploy.sh`, `remote_run.sh <cluster> [input] [poll]`, `build.sh [cpu|gpu]`
- Prior CPU cluster run: `run/cluster_test_log_20260227_190902.txt` (gitignored)
- Memory: `project_status.md` (outdated re: nvcc), `reference_compute.md`
