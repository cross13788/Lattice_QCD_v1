# Next Session Prompt

**Resume work in**: `/home/ross/Lattice_QCD_v1`
**Plan**: `claude_plans/2026-05-16_gpu-clover-spectroscopy-validation.md` — read this first.

## TL;DR

All correctness fixes (gauge, Wilson/clover HMC, CPU+GPU clover
Dirac+force) are committed and validated on `gpu-validation-comphys`.
Remaining gap: GPU clover **meson spectroscopy** output was never
numerically compared to CPU. Next: trace the GPU spectroscopy data flow,
diff all 8 correlator channels CPU vs GPU on the fixed config to ~1e-10,
and add the missing GPU `Pion:` summary print.

## Current State

- **Branch**: gpu-validation-comphys @ d9702c5
- **Working tree**: clean (all committed; no handoff commit needed)
- **Last session**: wired up + validated GPU clover (Dirac CG-count
  match RNG-free; GPU vs CPU clover HMC track within RNG noise).

## Key Findings to Carry Forward

- GPU clover Dirac + force are **validated** — this phase is about the
  spectroscopy *output/contraction*, not the operator.
- GPU spectroscopy (`src/gpu/lqcd_main_gpu.cpp:307-319`) calls the
  **host** `compute_propagator`/`meson_correlator`/`write_correlators`,
  yet the GPU run emitted GPU-style `CG converged` / `Propagator
  complete: 4808` lines. **Unresolved: does the propagator solve on GPU
  or CPU in the GPU binary?** Resolve this before claiming validation.
- GPU does NOT print the CPU `Pion: C(0)=...` summary; it does write
  `correlator_*.dat`. Compare the files, not stdout.
- RNG-free clover gate input already exists: `inputs/rngfree_clover_cg.inp`
  (loads `run/fixedcfg.bin`, β=6 4³×8, clover on). Key is `Clover
  Coefficient` not `c_SW`; cold start ⇒ F=0.
- Per-plane clover gradient harness (`CLOVER_DIAG_PLANE`, env, default
  off) exists and is reusable for any future clover regression.

## Immediate Next Steps

1. Trace GPU spectroscopy: how the linked CPU `compute_propagator.o`
   resolves `apply_dirac` in the GPU build (GPU vs host solve). Read
   `src/gpu/lqcd_main_gpu.cpp:290-320` + `src/Makefile` GPU objects.
2. Rebuild CPU from HEAD (`cp` exes aside first), run
   `inputs/rngfree_clover_cg.inp` on CPU and GPU; diff the 8
   `correlator_*.dat` in the `SU3_*` output dir to ~1e-10. Add a
   `bicgstab` variant too.
3. Add `Pion:`-summary print to the GPU correlator block for parity;
   re-run the Wilson (clover-off) RNG-free gate as regression.

## Watch Out For

- `build.sh {cpu,gpu}` make-cleans the other target's exe — `cp
  run/lqcd_cpu_fixed.exe` / `run/lqcd_gpu.exe` aside first.
- comphys SSH master key ~10-min expiry; fire `ssh comphys true` before
  any long local-only stretch (memory: ssh-master-key-keepalive).
- Compare correlator `.dat` files, not stdout (GPU has no `Pion:` line).
- If GPU spectroscopy turns out host-solved, say so plainly and ask the
  user whether wiring it to the GPU solver is in scope.

## Before You Start

- [ ] Read `claude_plans/2026-05-16_gpu-clover-spectroscopy-validation.md`
- [ ] `git log -1` shows d9702c5; tree clean; on gpu-validation-comphys
- [ ] Confirm comphys reachable (`ssh comphys true`) and
      `run/fixedcfg.bin` present
- [ ] Re-read CLAUDE.md pitfall #1 (clover), #6 (RNG-free gate), #7 (nvc++)
