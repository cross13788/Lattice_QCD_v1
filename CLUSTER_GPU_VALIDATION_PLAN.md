# Cluster GPU Validation — Plan of Action

**Date:** 2026-05-16
**Goal:** Build and correctness-validate the CUDA GPU port on `comphys`.
**Target cluster:** `comphys` (has NVIDIA HPC SDK: `nvc++`).
**Owner:** Christian Ross

---

## Context / Where We Stand

- Last commit `1cf838d`: clover fermion force structure + gradient check.
- CPU build: production, validated to 8³×16. Last CPU cluster run Feb 27, 2026
  (HMC therm hot/cold β=5.6 8⁴, spectroscopy k=0.154) — logs in
  `run/cluster_test_log_20260227_190902.txt`.
- **GPU CUDA port (added Mar 18, 2026): complete in source, NEVER compiled or
  run anywhere.** This is the single largest untested gap and the focus here.
- Clover HMC force still broken (out of scope for this round — quenched clover
  spectroscopy unaffected).

**Build-method correction:** `build.sh gpu` now uses `nvc++` for *all* files
(host C++ + `.cu`), **no nvcc**, to dodge GCC-version incompatibilities.
The `project_status` memory (says "nvcc for .cu files, sm_75") is outdated and
should be corrected after this round.

---

## Phase 1 — Pre-flight (local, before touching the cluster)

1. Commit/stash the dirty tree. Only `run/lqcd.inp` is modified source-side;
   the rest are run artifacts. Decide: commit the run artifacts or `.gitignore`
   them so `deploy.sh` rsync stays clean.
2. Confirm SSH access to `comphys` and that the NVIDIA HPC SDK module name /
   load command is known (build.sh expects `nvc++` on PATH).
3. Pick the validation input: smallest lattice that exercises gauge + fermion +
   HMC paths. Candidates: `inputs/quick_test_4x4x4x4.inp` (gauge),
   `inputs/fermion_test_4x4x4x8.inp` (solver/propagator),
   `inputs/hmc_4x4x4x4.inp` (HMC). Use all three for path coverage.
4. Generate CPU reference output locally for each chosen input (plaquette,
   Polyakov, correlators, HMC ΔH/acceptance) — this is the ground truth the
   GPU run must reproduce.

## Phase 2 — Deploy & GPU Build

5. `./deploy.sh comphys` (rsync source/inputs/analysis only — no compile/run).
6. On comphys: load NVIDIA HPC SDK, then `./build.sh gpu`.
   - build.sh auto-detects GPU arch via `nvidia-smi` (`ccXX`) and the GCC
     toolchain for `nvc++`. Capture the detected arch and full build log.
   - **Expected failure modes:** `thrust::complex` vs `std::complex` ABI,
     missing cuRAND link, `--gcc-toolchain` path, sm arch mismatch. Fix
     iteratively; record every fix for a follow-up commit.
7. Also build CPU on comphys (`./build.sh cpu`) so CPU-vs-GPU comparison runs
   on identical hardware/compiler-independent inputs.

## Phase 3 — Correctness Validation (GPU vs CPU, bitwise-aware)

8. For each of the 3 inputs, run CPU and GPU, same input file, same seed.
9. Compare:
   - **Plaquette** per config — agreement to solver/RNG tolerance. (RNG streams
     differ CPU vs GPU; for gauge-update comparison, prefer a `startType=file`
     fixed configuration so the comparison is deterministic and RNG-independent.)
   - **Wilson/clover Dirac operator + solver**: load a fixed gauge config,
     invert the same source on CPU and GPU — propagator must agree to CG
     tolerance (1e-8). This is RNG-free and the cleanest correctness test.
   - **Meson correlators** from the fixed config — agree to solver tolerance.
   - **HMC**: reversibility test and ΔH distribution; acceptance within
     statistical agreement of CPU.
10. Decision gate: GPU validated iff the fixed-config solver/propagator test
    matches CPU to CG tolerance AND HMC reversibility passes on GPU.

## Phase 4 — Wrap-up

11. Commit GPU build fixes from Phase 2 with a clear message.
12. Update memory: correct the GPU build method in `project_status.md`
    (nvc++-only, no nvcc), record GPU validation outcome + comphys arch.
13. Update `CLAUDE.md` GPU section if build invocation changed.
14. Optionally create `.claude-remote.yml` so the `cluster-ops` skill
    auto-loads for future comphys cycles (per `feedback_cluster_ops_allowed`).

---

## Risks / Open Questions

- **RNG divergence CPU vs GPU** makes Monte-Carlo-path comparison non-bitwise.
  Mitigation: anchor correctness on fixed-configuration solver/propagator tests
  (RNG-free), use stochastic observables only for statistical sanity.
- comphys GPU compute capability unknown until `nvidia-smi` on-node — affects
  `ccXX` and any `sm_75+` assumptions in kernels.
- NVIDIA HPC SDK module name on comphys not yet confirmed.
- Clover HMC force remains broken — quenched clover only this round.

## Not In Scope

- Debugging clover HMC fermion force.
- Large production runs (16³×32+) — gated on Phase 3 passing.
- B-spline lattice fermions, staggered/DW fermions.
