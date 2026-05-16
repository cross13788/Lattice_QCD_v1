# Plan: Clover (SW) fermion force — derivation + gradient-check fix

**Date**: 2026-05-16
**Project**: Lattice_QCD_v1
**Branch**: gpu-validation-comphys
**Status**: RESOLVED 2026-05-16. Per-plane `Average|ratio-1| ≈ 1–4e-5`,
full ≈ 2.7e-4 (finite-diff floor); `ratio=cl_rat=1.0000` every entry;
dynamical clover HMC stable. Final fix: single-source `dQ−dQ†` (each
link occurrence = `Φ_dQ` + companion `−Φ_{dQ†}` = `A†RB†U†`/etc.), no
separate reversed loops (that double-counted), explicit `i`,
`C=c_sw/32`. CLAUDE.md pitfall #1 rewritten. The notes below are the
audit trail.

**GPU port also DONE 2026-05-16.** GPU clover was entirely unwired:
`gpu_alloc_clover` never called, `d_sigmaMat` never uploaded. Added
`gpu_upload_sigma` + `get_sigma_base`, wired alloc/upload in
`lqcd_main_gpu.cpp` (when `useClover`), `ensure_clover_field` now uses
`gpuState.cloverCsw`. Ported the force to `src/gpu/clover_force_gpu.cu`
(one-thread-per-site scatter + `atomicAdd` into `d_cloverForceAcc`;
finalize `TA(i·acc)·c_sw/32`), hooked into `gpu_compute_fermion_force`,
added to `src/Makefile`. Validated: RNG-free clover Dirac CPU vs GPU CG
counts match (4807 vs 4808 total / per-solve exact); GPU vs CPU clover
HMC track within RNG noise (same thermalization, `dH≈−0.003…−0.006`,
stable). Test inputs: `rngfree_clover_cg.inp`, `hmc_clover_cmp.inp`.

## What this is

CLAUDE.md pitfall #1: the clover HMC fermion force is implemented but
wrong (right magnitude, wrong leaf/index structure). Quenched clover
spectroscopy is fine; dynamical clover HMC is not. `gradient_check.cpp`
(runs with `Update Method: hmc` + `Verbose Output: on` + clover on) is
the validator: `Average |ratio-1|` → 0 when correct; Wilson part is
already exact (ratio 1.0).

## Conventions (verified this session)

- Input key is `Clover Coefficient:` (NOT `c_SW:`); `set_parameter.cpp:90`.
  A correct test input exists: `inputs/hmc_clover_gradcheck.inp`
  (4^4, hot start, `Clover Coefficient: 1.0`). Cold start gives F=0
  (all-identity links) — must use hot/thermalized.
- `D = D_W + A`, `A(z) = c_sw·(i/4)·Σ_{a<b} σ_{ab}⊗F_{ab}(z)`,
  `F_{ab}=(Q_{ab}-Q_{ab}†)/8`, `σ_{ab}=(i/2)[γ_a,γ_b]`. A is Hermitian.
- `Q_{ab}(z)` = the 4 clover loops L1..L4 EXACTLY as in
  `compute_field_strength_tensor` (`wilson_flow.cpp:195-253`).
- Force: `X=(D†D)⁻¹φ`, `Y=DX` (from `fermion_force.cpp`);
  `dS_cl = -2Re[Y†dA X] = -(Y†dA X + X†dA Y)`.

## Derivation (this session, believed correct)

Per site z, ordered plane (a<b), spin-traced 3×3 colour source:
```
R(z)_{dc} = Σ_{s,s'} σ^{(ab)}_{ss'} [ X_{s'}(z)_d Y*_s(z)_c + Y_{s'}(z)_d X*_s(z)_c ]
```
For each clover loop (product V0V1V2V3) and link position i
(P_left=V0..V_{i-1}, P_right=V_{i+1}..V3):
```
M_i = P_right · R(z) · P_left
link appears as U   : Φ += U · M_i
link appears as U†  : Φ -= M_i · U†
```
summed over the 4 loops and (for the −Q†) their reversed-dagger-flipped
partners. Final per link: `F^cl += C · TA( i · Σ Φ )`, `C = c_sw/8`
(sign/magnitude still to pin).

## Implemented (local + comphys, uncommitted)

- `clover_module.hpp`: `clover_diag_plane()/leaf()` decls.
- `clover_term.cpp`: env-gated `CLOVER_DIAG_PLANE/LEAF` (default −1 =
  off, no production effect), and a plane gate in `compute_clover_field`.
- `clover_fermion_force.cpp`: full mechanical rewrite from the exact
  Q-loops (`build_leaf`), with the i factor and TA.
- `inputs/hmc_clover_gradcheck.inp` (on comphys; add to repo).

## Results

`Average |ratio-1|` per plane (was 0.75–2.4 / total 4.11 at start):
- i + Q±Q† + TA: planes ≈ 0.34–0.60 (most uniform).
- i + Q-only + TA: planes 0,1,2,4 ≈ 0.20–0.26 but plane 3 = 1.01,
  plane 5 = 0.47.

## Remaining leads (in priority order)

1. **U†-position sign / convention.** Re-derive d(U†)/dε for
   U→exp(εG)U with G=iTa: `dU†/dε = -U†G`. The `Φ -= M_i·U†` term and
   its interaction with the −Q† reversed loops likely has a sign or an
   ordering error. Use `CLOVER_DIAG_LEAF` + a single-loop manual check.
2. **Q vs Q† vs TA double-count.** `(Q−Q†)/8` antisymmetry can come
   either from explicitly summing −Q† loops OR from the final TA — doing
   both double-counts. Q-only+TA fixed planes 0/1/2/4 to ~0.2; decide
   the consistent single source of antisymmetrization, then the residual
   ~0.2 is likely a clean prefactor (expect C = ±c_sw/(some power of 2)).
3. **Plane 3 = (1,2) anomaly.** Behaves qualitatively differently under
   toggles ⇒ suspected directional-index bug in `build_leaf` for that
   plane (check the z±a / z±b shift composition order for a=1,b=2).
4. Once structure right and uniform: fit C from any single
   (site,μ,gen) where `ratio` is constant, then verify
   `Average|ratio-1| < 1e-5` for all 6 planes AND full (no plane
   restriction), then a short dynamical clover HMC stability run
   (expect ~5× smaller MD step than Wilson — CLAUDE.md pitfall #1).

## Harness usage

```
cp inputs/hmc_clover_gradcheck.inp run/lqcd.inp
for P in 0 1 2 3 4 5; do CLOVER_DIAG_PLANE=$P bash run.sh cpu \
  2>&1 | grep 'Average |ratio'; done
# detail: CLOVER_DIAG_PLANE=5 ... | sed -n '/site mu/,/Average/p'
# in-plane dirs: plane sigma_index 0:(0,1) 1:(0,2) 2:(0,3) 3:(1,2)
#                4:(1,3) 5:(2,3)
```
`ratio` (= dS_num/gradTot) is the clean metric for in-plane μ;
`cl_rat` is contaminated (its gradW baseline uses a clover-OFF
inversion) — use `ratio`, not `cl_rat`.

## No-regression note

All changes are clover-only and the diagnostics are env-gated (default
−1). `compute_clover_force` is called only when `useClover`. Wilson /
quenched / GPU-HMC paths are untouched. Clover HMC was already broken
(pitfall #1), so this is strictly progress, not a regression — but the
force is NOT yet correct; do not present clover HMC as usable.
