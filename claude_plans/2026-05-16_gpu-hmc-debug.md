# Plan: Debug GPU HMC (separate from the now-fixed sweep-coloring bug)

**Date**: 2026-05-16
**Project**: Lattice_QCD_v1
**Branch**: gpu-validation-comphys
**Parent commit**: 3c438e0 (parallel-sweep coloring fix + corrected pitfall #6)

## Context

The previous "GPU gauge-update broken" pitfall #6 was a misdiagnosis with
an inverted baseline. Fixed and committed (3c438e0):

- Physically correct quenched `<P>(beta=5.6, 4^4) ≈ 0.54` (SU(3) Wilson
  literature). CPU `0.427` was correct heat bath **corrupted by a broken
  overrelaxation** (site-parity + all-mu-per-thread coloring destroyed the
  microcanonical property; CPU cooled ovr=0→0.542, 2→0.447, 8→0.360).
- Fix: mu-outer loop, one `(mu,parity)` pass per launch/sweep, in CPU
  `heat_bath_sweep`/`overrelaxation_sweep`/`metropolis_sweep` and GPU
  `heat_bath_gpu`/`overrelaxation_gpu`.
- Post-fix overrelaxation exactly microcanonical: CPU ovr=0/4/8 all
  `<P>=0.53587±0.00179`; GPU `0.53955±0.00132`; agree ~1.7σ. **GPU
  quenched heat bath validated correct.**

## The remaining bug: GPU HMC

`inputs/hmc_4x4x4x4.inp` (dynamical, beta=5.6, kappa=0.12, 10 MD steps,
dt=0.02):

- **CPU-fixed (correct ref)**: `<P>≈0.50`, ΔH two-sign around 0, 99% acc.
- **GPU (broken)**: `<P>≈0.78`, ΔH **systematically negative**
  (−0.05…−0.20), 100% accept. GPU MD is self-consistent but conserves the
  wrong energy → over-orders the gauge field.

HMC path components NOT touched by the coloring fix: GPU gauge force
(read-only, no race; `gpu/gauge_force_gpu.cu` verified line-identical to
CPU), GPU leapfrog, GPU conjugate momentum, GPU pseudofermion action,
GPU fermion force.

## Implementation Steps

1. On comphys: `build.sh cpu` then `cp run/lqcd.exe run/lqcd_cpu_fixed.exe`;
   `build.sh gpu` (target switch make-cleans the other exe — preserve
   first). `.nvhpc/localrc` self-heal is automatic.
2. RNG-free force gate: load the fixed config (`run/fixedcfg.bin` via
   `Start Type: file:`), compute the **total HMC force** (gauge +
   fermion) for a few links CPU vs GPU. Gauge force already matches; this
   isolates the GPU **fermion force** and momentum/leapfrog sign.
   - Suspects in order: `gpu/leapfrog_gpu.cu` (momentum half-step sign,
     `pi -= (dt/2) F` vs `+=`; gauge vs fermion force sign convention),
     `gpu/fermion_force_gpu.cu` (the X = (D†D)⁻¹φ contraction sign / TA),
     conjugate-momentum sign in `gpu/random_fields_gpu.cu` vs the GPU
     dH kinetic term, `gpu/pseudofermion_action_gpu.cu`.
3. Systematically-negative ΔH with 100% accept is the signature of a
   **force/sign or missing-term error** consistent within the integrator
   but inconsistent with the dH used for Metropolis. Check that the GPU
   computes ΔH from the SAME action (S_G + S_pf) the MD force derives
   from, and that the gauge-force sign in the GPU leapfrog matches the
   `dS = -Re Tr(X F)`, `pi -= (eps/2) F` convention in CPU
   `leapfrog_integrator.cpp` / `gauge_force.cpp`.
4. Cross-check: pure-gauge GPU HMC (set kappa so fermions are off, or a
   quenched-HMC input) — if pure-gauge GPU HMC also runs away, the bug is
   in GPU leapfrog/gauge-force-sign/momentum, not the fermion force. If
   pure-gauge GPU HMC is fine, the bug is in the GPU fermion force /
   pseudofermion path.
5. Gate: GPU `hmc_4x4x4x4.inp` must give ⟨exp−ΔH⟩≈1 (two-sign ΔH) AND
   equilibrium `<P>` matching CPU-fixed (~0.50) within ~2σ.
6. Commit; update CLAUDE.md pitfall #6 (HMC portion) and
   `project_status.md` memory.

## Test Harness

- Correct CPU reference exe: rebuild as `run/lqcd_cpu_fixed.exe`.
- `inputs/hmc_4x4x4x4.inp` — the failing gate.
- `inputs/quenched_4x4x4x4_b56.inp` — quenched heat-bath gate (now PASSES
  GPU vs CPU ~0.54; use to confirm no regression).
- RNG-free fixed config: `inputs/gen_fixedcfg_4x4x4x8.inp` →
  `run/fixedcfg.bin`; load via `Start Type: file:`.

## Watch Out For

- `build.sh {cpu,gpu}` make-cleans the OTHER target's exe on switch.
  Always `cp` the exe you need to a distinct name first.
- CPU/GPU RNG streams differ — gate HMC on ⟨exp−ΔH⟩ and equilibrium
  `<P>`, and on RNG-free per-link force comparisons, never on RNG-path
  trajectories matching bit-wise.
- Clover HMC force separately broken (CLAUDE.md pitfall #1) — use Wilson
  (`useClover` off) for this debug.
- comphys SSH master key expires ~10 min after last use — keep warm.

## References

- CLAUDE.md pitfall #6 (rewritten in 3c438e0), #1 (clover), #7 (nvc++)
- Memory: `project_status.md`, `ssh-master-key-keepalive`,
  `reference_compute`
- Prior plan: `claude_plans/2026-05-16_gpu-gauge-update-debug.md`
  (superseded — its premise that GPU heat bath is broken was wrong)
