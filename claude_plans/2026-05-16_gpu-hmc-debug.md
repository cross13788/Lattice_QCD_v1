# Plan: Debug GPU HMC (sweep-coloring bug already fixed in 3c438e0)

> **RESOLVED 2026-05-16.** Root cause: `src/gpu/random_fields_gpu.cu`
> `kernel_generate_momenta` ran one-thread-per-link but indexed cuRAND
> state per-site (`randStates[idx/4]`) → 4 identical link momenta per
> site → rank-deficient momentum heat-bath → `<P>≈0.78`. Fixed
> (one-thread-per-site, loop μ). Reversibility probe confirmed the
> integrator was correct; the deterministic-momentum test isolated the
> RNG path. All gates pass. See `next_prompt.md` and CLAUDE.md
> pitfall #6. The investigation steps below are kept as the audit trail.

**Date**: 2026-05-16 (updated after localization session)
**Project**: Lattice_QCD_v1
**Branch**: gpu-validation-comphys
**Parent commit**: 8ec9751 (3c438e0 fix + docs handoff)

## Context

The "GPU gauge-update broken" pitfall #6 was a misdiagnosis (fixed 3c438e0:
parallel-sweep coloring destroyed overrelaxation). GPU quenched heat bath
validated correct (`<P>≈0.54`, β=5.6). **GPU HMC is a separate, still-open
bug.** This session reproduced it and localized it sharply.

## Reproduced (comphys, this session)

`inputs/hmc_4x4x4x4.inp` (dynamical β=5.6 κ=0.12 10 MD dt=0.02):
- CPU-fixed (`run/lqcd_cpu_fixed.exe`): `<P>≈0.50`, ΔH two-sign, 99% acc. CORRECT.
- GPU: `<P>≈0.78`, ΔH systematically negative (−0.04…−0.20), 100% acc. BROKEN.

## Localization results (this is the new information)

1. **Pure-gauge discriminator** — `inputs/hmc_quenchedlimit_4x4x4x4.inp`
   (κ=1e-6 ⇒ fermion force ~1e-6, negligible vs gauge ~O(1)):
   - GPU still `<P>≈0.78`, ΔH<0, 100% acc.
   - CPU κ=1e-6 samples correctly (`<P>≈0.50`, small two-sign ΔH).
   ⇒ **Bug is in the GPU PURE-GAUGE MD path. Fermion sector cleared.**
2. **RNG-free gauge-force gate** (`inputs/hmc_fixedcfg_force.inp`, loads
   `run/fixedcfg.bin`, prints `|Fg|²` at U(0)): GPU == CPU to **all 10
   digits** (`|Fg|²=1.1686793112e+05`, `Fg[0](0,0)=(0,7.5262035692e-01)`).
   ⇒ **GPU gauge force kernel is exact. Not a force/sign bug.**
3. **Integrator is symplectic**: per-trajectory `dT ≈ −dS_G` to ~1 part in
   1e3 (e.g. dT=+523.43, dS_G=−525.40, dH=−0.30 ⇒ H=T+S_G+S_PF conserved).
   ΔH ∝ dt² (N=10 dt=0.02 ΔH≈−0.13; N=80 dt=0.0025 ΔH≈−0.002, ratio ≈64).
   Fixed-dt N-scan (N=10/40/160, dt=0.01, fixed cfg): |ΔH| bounded &
   non-monotonic (−0.095 / −0.855 / −0.459) ⇒ **no secular drift**.
4. β, vol, coeff (β/Nc), S_G=6βV(1−P), momentum gen, kinetic (T≈4·nLinks
   ✓), exp-update sign (U=exp(εΠ)U), TA projection, staple — all
   **textually identical CPU↔GPU** (verified by reading both).

## The paradox to resolve next

A symplectic, exact-force, correct-ΔH, momentum-refreshed HMC **must**
reproduce the heat-bath quenched `<P>` (0.54 at β=5.6) in the pure-gauge
limit. GPU gives 0.78. Since force and ΔH are mutually consistent
(dH→0 as dt→0) the GPU is doing *valid HMC for the wrong stationary
distribution*. With β/vol/force/plaquette all proven correct, the defect
must break **detailed balance / area preservation** without breaking
energy conservation. Prime suspects (all pure-gauge, HMC-only, NOT
exercised by the validated heat bath or the BiCGstab spectroscopy gate):

- `src/gpu/su3_exp_gpu.cu` `kernel_gauge_exp_update` / `dev_su3_exp` +
  `dev_su3_reunitarize` (su3_device.cuh:133): a reunitarization that is
  not exactly the same map forward vs reverse breaks reversibility →
  systematic ΔH bias → 100% accept → wrong equilibrium, *while still
  conserving H to O(dt²)*. CPU uses the same 12-term Horner +
  Gram-Schmidt and is correct, so look for a GPU-specific deviation
  (rsqrt vs 1/sqrt, fma/contraction reordering under nvc++, register
  precision) — not a textual difference.
- Metropolis/accept path in `gpu_hmc_trajectory` (hmc_gpu.cu:91-98) and
  the LCG `hmc_random_uniform` — verify reject actually restores U
  (d_gaugeFieldSaved) and that 100% accept is genuine, not a bypass.
- Reversibility test (definitive): integrate forward, negate Π, integrate
  again; correct leapfrog returns to start to ~1e-10. Build a small
  device kernel to negate momentum; instrument `gpu_hmc_trajectory`
  trajectory 1. Non-return ⇒ exp-update/reunitarize is the culprit.

## Next steps

1. comphys: `build.sh cpu`; `cp run/lqcd.exe run/lqcd_cpu_fixed.exe`;
   `build.sh gpu` (target switch make-cleans the other exe). Artifacts
   currently present and clean: `run/lqcd_cpu_fixed.exe`,
   `run/lqcd_gpu.exe` (built 08:05, uninstrumented), `run/fixedcfg.bin`.
2. Implement the GPU reversibility probe (forward → −Π → forward; print
   |U−U₀|, |ΔH_roundtrip|). If non-reversible → fix `dev_su3_exp_update`/
   `dev_su3_reunitarize` (make the GPU map bit-faithful to CPU; consider
   replacing `rsqrt` with `1.0/sqrt`, disabling fma contraction in
   exp/reunit via `#pragma` or `--fma=false` scoped, or matching CPU
   Gram-Schmidt order exactly).
3. Re-gate: GPU `hmc_4x4x4x4.inp` ⟨exp−ΔH⟩≈1 (two-sign ΔH) AND
   equilibrium `<P>`≈0.50 within ~2σ of CPU-fixed; AND pure-gauge
   κ=1e-6 GPU `<P>`→0.54 (matches heat bath).
4. Confirm no regression: GPU `quenched_4x4x4x4_b56.inp` `<P>`≈0.54
   (PASSES as of this session, clean rebuild).
5. Commit; update CLAUDE.md pitfall #6 (HMC portion) + `project_status.md`.

## Test Harness (all created/verified this session)

- `run/lqcd_cpu_fixed.exe` — correct CPU reference.
- `inputs/hmc_4x4x4x4.inp` — failing dynamical gate.
- `inputs/hmc_quenchedlimit_4x4x4x4.inp` — κ=1e-6 pure-gauge discriminator.
- `inputs/hmc_fixedcfg_force.inp` — RNG-free; loads `run/fixedcfg.bin`;
  used with the `[FRCDBG-*]` leapfrog instrumentation (now reverted).
- `inputs/quenched_4x4x4x4_b56.inp` — quenched regression gate (PASSES).
- Instrumentation pattern (reverted, re-add as needed): `[HMCDBG]`
  fprintf in `gpu_hmc_trajectory` after `deltaH=`; `[FRCDBG-*]` |Fg|²
  dumps in CPU/GPU leapfrog after first gauge-force eval.

## Watch Out For

- `build.sh {cpu,gpu}` make-cleans the OTHER target's exe — `cp` first.
- CPU/GPU RNG differ — gate on ⟨exp−ΔH⟩, equilibrium `<P>`, RNG-free
  per-link force/reversibility, never bitwise RNG-path trajectories.
- Clover HMC force separately broken (pitfall #1) — Wilson only here.
- comphys SSH master key ~10-min expiry — keep warm.
- Remote is an rsync of committed source (no .git there); revert any
  remote instrumentation before ending (done this session — verified
  3 files byte-identical to committed).

## References

- CLAUDE.md pitfall #6 (rewritten 3c438e0), #1 (clover), #7 (nvc++)
- Memory: `project_status.md`, `ssh-master-key-keepalive`, `reference_compute`
