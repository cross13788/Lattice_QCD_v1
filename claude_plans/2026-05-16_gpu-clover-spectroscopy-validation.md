# Plan: Validate GPU clover meson spectroscopy end-to-end

**Date**: 2026-05-16
**Project**: Lattice_QCD_v1
**Branch**: gpu-validation-comphys
**Parent commit**: d9702c5 (GPU clover Dirac+force wired & validated)

## Context

All correctness fixes for gauge + Wilson/clover HMC and the GPU clover
Dirac/force are committed and validated. The one remaining gap: the GPU
clover *meson spectroscopy* output path was never compared numerically
to CPU. This session validated the GPU clover **Dirac operator** via
RNG-free CG-iteration-count match (CPU vs GPU 4807 vs 4808 total,
per-solve exact), but the actual meson **correlator numbers** for clover
were never diffed — CPU printed `Pion: C(0)=2.0166e+01`, the GPU run
printed only `<P>` and `Propagator complete: 4808` with no `Pion:` line.
Next phase: pin down what GPU spectroscopy actually exercises and
validate all 8 meson channels CPU vs GPU to ~1e-10 on a fixed config.

## What Was Accomplished This Session

- Fixed GPU HMC cuRAND per-link/per-site momentum aliasing (`aafbcc4`).
- Fixed CPU clover SW fermion force, gradient-check validated to ~1e-5
  per plane (`ca90027`); built a reusable env-gated per-plane isolation
  harness (`CLOVER_DIAG_PLANE`).
- Discovered GPU clover was entirely unwired (`gpu_alloc_clover` never
  called, `d_sigmaMat` never uploaded); wired it up + ported the clover
  force to `src/gpu/clover_force_gpu.cu`; validated GPU clover Dirac
  (RNG-free CG-count match) and GPU vs CPU clover HMC tracking
  (`d9702c5`).

## What Didn't Work / Dead Ends

- For the CPU clover force: a literal leaf-by-leaf patch of the old
  `leaf_force_matrix` decomposition did not converge. The fix was a
  full mechanical rewrite from the exact `wilson_flow.cpp` Q-loops.
- Guessing structural toggles (Q-only vs Q±Q†, with/without TA,
  with/without explicit `i`) without the per-plane harness was slow.
  The per-plane gradient isolation harness is what made it tractable —
  reuse it, don't guess.

## Next Phase: Goals

1. Determine exactly what the GPU binary's clover spectroscopy path
   executes: does `compute_propagator` (called at
   `src/gpu/lqcd_main_gpu.cpp:307-319`, host code) dispatch the Dirac
   solve to the GPU (the run emitted GPU-style `CG converged` lines and
   `Propagator complete: 4808`) or to CPU? Document the real data flow.
2. Numerically validate GPU vs CPU clover meson correlators: all 8
   channels, `C(t)` for all `t`, and effective masses, on the fixed
   config `run/fixedcfg.bin` (RNG-free), to ~1e-10.
3. Fix the missing GPU `Pion:` summary print so GPU/CPU spectroscopy
   output is at parity (CPU prints it from
   `generate_configurations.cpp`; GPU main doesn't replicate it).

## Implementation Steps

1. Trace the GPU spectroscopy path: read `src/gpu/lqcd_main_gpu.cpp`
   ~lines 290-320 (the `measureCorrelators` block) and how the linked
   CPU `compute_propagator.o` / `cg_solver.o` / `bicgstab_solver.o`
   resolve their `apply_dirac` in the GPU build (does the GPU build
   override `apply_dirac` to dispatch to `gpu_apply_dirac`, or is the
   propagator solved purely on host?). Check `src/Makefile` GPU object
   list and which `apply_dirac` symbol wins at link.
2. Run the RNG-free clover gate on `run/fixedcfg.bin` with clover on
   (`inputs/rngfree_clover_cg.inp` already exists; also make a
   `bicgstab` variant). For CPU: `run/lqcd_cpu_fixed.exe` (rebuild from
   HEAD first). For GPU: `run/lqcd_gpu.exe`. Both write
   `correlator_<CHANNEL>.dat` into the `SU3_*` output dir via
   `write_correlators` (`src/write_correlators.cpp:32`).
3. `diff`/numerically compare the 8 `correlator_*.dat` files CPU vs GPU
   (pion, rho_x/y/z, scalar, a1, etc. — see `meson_correlator.cpp`
   channels). Tolerance ~1e-10 (same as the Wilson RNG-free gate in
   CLAUDE.md pitfall #6). Also compare `effective_mass` output.
4. If GPU spectroscopy is actually host-CPU solved (gauge downloaded),
   state that plainly — then the "validation" is trivial and the real
   gap is that GPU spectroscopy doesn't use the GPU solver at all;
   decide whether to wire `compute_propagator` to the GPU solver.
5. Add the `Pion: C(0)=...` quick-summary print to the GPU correlator
   block in `lqcd_main_gpu.cpp` to match CPU
   (`generate_configurations.cpp` ~line 160 prints it).
6. Regression: re-run the Wilson (clover off) RNG-free gate
   (`inputs/rngfree_fixedcfg_4x4x4x8.inp`) — must still match ~1e-10.

## Open Questions

- (Blocking for step 2 interpretation) Does the GPU binary solve the
  spectroscopy propagator on GPU or CPU? The emitted GPU CG lines
  suggest GPU, but `compute_propagator` is host code — resolve before
  claiming "GPU clover spectroscopy validated".
- (Deferrable) If GPU spectroscopy is host-solved, is wiring it to the
  GPU solver in scope, or is fixed-config spectroscopy expected to run
  via the CPU solver in the GPU binary? Ask the user.

## Things to Watch Out For

- GPU correlator output: the GPU main does NOT print the `Pion:`
  summary line CPU prints; it does call `write_correlators` (files).
  Compare the `.dat` files, not stdout.
- Input key is `Clover Coefficient` (NOT `c_SW`); cold start ⇒ F=0 —
  must load `fixedcfg.bin` (β=6, 4³×8). RNG-free needs 0 therm / 0
  sweeps-between.
- `build.sh {cpu,gpu}` make-cleans the OTHER target's exe — `cp
  run/lqcd_cpu_fixed.exe` (and `lqcd_gpu.exe`) aside before switching
  targets. `.nvhpc/localrc` self-heals (CLAUDE.md pitfall #7).
- comphys SSH master key expires ~10 min after last use; the biggest
  trap is a long local-only stretch (memory: ssh-master-key-keepalive).
- The GPU clover *Dirac/force* are already validated — this phase is
  about the spectroscopy *output/contraction*, not the operator.

## Validation Checklist

- [ ] GPU spectroscopy data-flow documented (GPU vs host solve).
- [ ] CPU vs GPU all 8 `correlator_*.dat` agree ~1e-10 on `fixedcfg.bin`
      with clover on (both `cg` and `bicgstab`).
- [ ] Effective masses match.
- [ ] Wilson (clover off) RNG-free gate still ~1e-10 (regression).
- [ ] `build.sh cpu` and `build.sh gpu` compile cleanly.
- [ ] GPU prints a `Pion:` summary at parity with CPU.

## OUTCOME (2026-05-16, completed)

**User scope decision:** accept host-solved GPU spectroscopy, finish plan
(do NOT wire spectroscopy to the GPU solver this session).

- **Data flow resolved:** GPU `measureCorrelators` block downloads gauge
  D2H then calls **host** `compute_propagator` → host `cg_solver`/
  `bicgstab_solver` → host `apply_dirac`. GPU symbols are `gpu_`-prefixed
  (`ColorSpinorDev*`), never collide; `gpu_compute_propagator`/
  `gpu_cg_solver` are never invoked for spectroscopy. CPU vs GPU
  correlators therefore differ only by g++ vs nvc++ FP (same source).
- **Clover CG validated:** all 8 channels CPU vs GPU agree max **8.5e-12**
  on `fixedcfg.bin` (β=6 4³×8, κ=0.150, c_SW=1.0), within ~1e-10 gate.
- **Clover BiCGstab:** does NOT converge on either binary (stagnates
  ~1e-2 ≫ 1e-12 at the 2000-iter cap). Pre-existing unpreconditioned-
  BiCGstab/clover-conditioning limitation, NOT a GPU regression. Use CG.
- **Wilson (clover off) BiCGstab regression:** CPU vs GPU 2.18e-11,
  within ~1e-10 — passes; confirms BiCGstab itself is fine, the clover
  operator at κ=0.150 is the issue.
- **GPU `Pion:` print added** (`lqcd_main_gpu.cpp`), at parity with CPU.
- **Builds:** `build.sh cpu` and `build.sh gpu` both compile cleanly on
  comphys (nvc++ 24.11, cc75).
- **Docs:** CLAUDE.md Status, `.claude-remote.yml` (stale "clover broken"
  note removed), and memory `project_status` updated.

### Validation Checklist (final)
- [x] GPU spectroscopy data-flow documented (host-solved).
- [x] CPU vs GPU 8 channels agree ~1e-12 clover **CG** on `fixedcfg.bin`.
- [~] BiCGstab: non-convergent for clover (documented; not a gate).
- [x] Effective masses match (Pion m_eff(1)=1.506217 CPU≡GPU, CG).
- [x] Wilson (clover off) RNG-free gate 2.18e-11 (regression passes).
- [x] `build.sh cpu` and `build.sh gpu` compile cleanly.
- [x] GPU prints a `Pion:` summary at parity with CPU.

## References

- Project CLAUDE.md: pitfall #1 (clover, now fixed CPU+GPU), #6
  (RNG-free gate convention), #7 (nvc++ localrc).
- Files: `src/gpu/lqcd_main_gpu.cpp:290-320`, `src/write_correlators.cpp`,
  `src/meson_correlator.cpp`, `src/compute_propagator.cpp`,
  `src/gpu/clover_gpu.cu`, `src/gpu/clover_force_gpu.cu`,
  `src/Makefile` (GPU object list / apply_dirac resolution).
- Inputs: `inputs/rngfree_clover_cg.inp` (new),
  `inputs/rngfree_fixedcfg_4x4x4x8.inp`, `run/fixedcfg.bin`.
- Related plans: `claude_plans/2026-05-16_clover-force-debug.md`,
  `claude_plans/2026-05-16_gpu-hmc-debug.md`.
- Memory: `clover-force-derivation`, `project-status-may-2026-...`,
  `ssh-master-key-keepalive`, `gpu-curand-thread-granularity`,
  `reference_compute`.
