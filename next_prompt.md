# Next Session Prompt

**Resume work in**: `/home/ross/Lattice_QCD_v1`
**Last plan (COMPLETE)**: `claude_plans/2026-05-16_gpu-clover-spectroscopy-validation.md`

## Status: GPU clover spectroscopy validation phase COMPLETE

The full CPU+GPU stack is validated. This session closed the last gap:

- GPU clover **spectroscopy is host-solved** (gauge D2H → host
  `compute_propagator` → host CG/BiCGstab → host `apply_dirac`; no GPU
  solver invoked for spectroscopy). Scope decision (user): accept this;
  do not wire spectroscopy to the GPU solver.
- Clover **CG**: 8 channels CPU vs GPU agree to **8.5e-12** on
  `fixedcfg.bin` — validated.
- Clover **BiCGstab**: non-convergent on *both* binaries (stagnates
  ~1e-2 at the 2000-iter cap) — pre-existing unpreconditioned-BiCGstab/
  clover conditioning limitation, not a GPU regression. **Use CG for
  clover spectroscopy.**
- Wilson (clover off) BiCGstab regression: 2.18e-11 — passes.
- GPU `Pion:` summary print added (parity with CPU).
- Builds clean (cpu + gpu, comphys).

## Open / Possible Next Directions (none blocking)

1. **Wire GPU spectroscopy to the GPU solver** (deferred this session).
   Route `compute_propagator` through `gpu_cg_solver` keeping gauge on
   device, then re-validate. Substantial; only if GPU spectroscopy
   throughput matters.
2. **Clover BiCGstab conditioning**: even-odd-preconditioned clover
   BiCGstab (`clover_even_odd.cpp` / 6×6 LU) may converge where the plain
   solver stagnates — worth a path if BiCGstab clover spectroscopy is
   wanted. CG is the supported path for now.
3. Production dynamical clover runs (autocorrelation, tuned MD step,
   scale setting) — CLAUDE.md Open Problems #1.
4. Scale setting (Sommer r_0 / Wilson-flow t_0) — Open Problems #3.

## Before You Start

- [ ] Read `claude_plans/2026-05-16_gpu-clover-spectroscopy-validation.md`
      (esp. the OUTCOME section).
- [ ] `git log -1`; tree clean; on gpu-validation-comphys.
- [ ] comphys: spectroscopy gate exes are `run/lqcd_cpu_test.exe` /
      `run/lqcd_gpu_test.exe`; inputs `inputs/rngfree_clover_cg.inp`,
      `inputs/rngfree_fixedcfg_4x4x4x8.inp`; `run/fixedcfg.bin` present.
- [ ] comphys tree is a deploy (rsync) target, NOT a git repo — git lives
      only locally; `deploy.sh comphys` syncs.
