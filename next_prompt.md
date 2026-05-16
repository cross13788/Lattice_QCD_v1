# Next Session Prompt

**Resume work in**: `/home/ross/Lattice_QCD_v1`
**Plan**: `claude_plans/2026-05-16_gpu-hmc-debug.md` — read first.

## TL;DR

The "GPU gauge-update broken" pitfall #6 was a **misdiagnosis with an
inverted baseline** — FIXED and committed (3c438e0). The real bug was a
parallel-sweep coloring defect (site parity + all-mu-per-thread) that
destroyed overrelaxation's microcanonical property in **both CPU and
GPU**. GPU quenched heat bath is now validated correct (GPU vs CPU
`<P>≈0.54` at beta=5.6, ~1.7σ). **GPU HMC is still genuinely broken** —
that is the next target.

## Current State

- **Branch**: gpu-validation-comphys @ 3c438e0. Tree clean.
- cluster-ops auto-loads: comphys, `/home/ross/Lattice_QCD_v1`, GPU cycle.
- comphys SSH master key expires ~10 min after last use — ping to keep
  warm (memory: `ssh-master-key-keepalive`).

## Fixed & Validated This Session (commit 3c438e0)

- Root cause: `(x,mu)` and `(x+mu-nu,nu)` share the negative-staple
  plaquette with the same site parity; updating all 4 mu per thread
  reflects them simultaneously → kills microcanonical overrelaxation
  (CPU cooled ovr=0→0.542, 2→0.447, 8→0.360).
- Fix: **mu-outer** loop in CPU `heat_bath_sweep`/`overrelaxation_sweep`/
  `metropolis_sweep` and GPU `heat_bath_gpu`/`overrelaxation_gpu`.
- Result: overrelaxation now exactly count-invariant. CPU ovr=0/4/8 all
  `<P>=0.53587±0.00179`; GPU `0.53955±0.00132`. GPU quenched heat bath
  validated correct. CLAUDE.md pitfall #6 rewritten.

## The Bug to Fix (GPU HMC)

`inputs/hmc_4x4x4x4.inp` (dynamical, beta=5.6, kappa=0.12):
- CPU-fixed (correct): `<P>≈0.50`, ΔH two-sign, 99% acc.
- GPU (broken): `<P>≈0.78`, ΔH **systematically negative**, 100% accept.

HMC components untouched by the coloring fix: GPU gauge force (read-only,
verified line-identical to CPU), leapfrog, conjugate momentum,
pseudofermion action, fermion force. Signature (one-sign ΔH + 100%
accept) = force/sign or missing-term error: GPU MD self-consistent but
conserves the wrong energy / inconsistent with the Metropolis dH.

## Immediate Next Steps

1. comphys: `build.sh cpu`; `cp run/lqcd.exe run/lqcd_cpu_fixed.exe`;
   `build.sh gpu` (target switch make-cleans the other exe).
2. RNG-free per-link **total force** (gauge+fermion) CPU vs GPU on
   `run/fixedcfg.bin`. Gauge force matches → isolates GPU fermion force +
   leapfrog/momentum sign.
3. Discriminator: pure-gauge GPU HMC. Runs away too ⇒ bug in GPU
   leapfrog/gauge-force-sign/momentum. Fine ⇒ bug in GPU fermion-force /
   pseudofermion path.
4. Suspects: `gpu/leapfrog_gpu.cu` (half-step sign), `gpu/
   fermion_force_gpu.cu`, conjugate-momentum sign vs GPU dH kinetic
   term, `gpu/pseudofermion_action_gpu.cu`.
5. Gate: GPU `hmc_4x4x4x4.inp` ⟨exp−ΔH⟩≈1 (two-sign ΔH) AND equilibrium
   `<P>`≈0.50 within ~2σ of CPU-fixed.

## Before You Start

- [ ] Read `claude_plans/2026-05-16_gpu-hmc-debug.md`
- [ ] Read CLAUDE.md pitfall #6 (rewritten), #1 (clover, out of scope), #7
- [ ] Confirm cluster-ops auto-loaded; `git log -1` = 3c438e0
- [ ] Re-confirm no regression: GPU `quenched_4x4x4x4_b56.inp` `<P>`≈0.54

## Watch Out For

- `build.sh {cpu,gpu}` make-cleans the OTHER target's exe — `cp` first.
- Never gate HMC on RNG-path trajectory equality (CPU/GPU RNG differ);
  use ⟨exp−ΔH⟩, equilibrium `<P>`, and RNG-free per-link force.
- Clover HMC force separately broken (pitfall #1) — use Wilson here.
- comphys SSH key 10-min expiry — keep the connection warm.
