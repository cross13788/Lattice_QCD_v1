# Next Session Prompt

**Status: GPU HMC bug RESOLVED 2026-05-16.** No open debugging task.
Plan history: `claude_plans/2026-05-16_gpu-hmc-debug.md`.

## What was fixed

`src/gpu/random_fields_gpu.cu` `kernel_generate_momenta` was launched
one-thread-per-link (`idx` over `nLinks`) but indexed the cuRAND state
per-site (`randStates[idx/4]`). cuRAND states are allocated per site
(`gpu_alloc_rand(latticeVolume)`), so the 4 link directions at each site
were generated from the SAME state → identical conjugate momenta per
site → rank-deficient momentum heat-bath → non-canonical HMC →
`<P>≈0.78` instead of `≈0.50`. Fix: one thread per site, loop μ=0..3
from the single advancing per-site stream (mirrors the correct
`kernel_generate_gaussian_spinor` / heat-bath kernels).

## Validation (all pass, comphys)

- GPU dynamical `inputs/hmc_4x4x4x4.inp`: `<P>≈0.53`, two-sign ΔH,
  **97% accept** (was 0.78 / one-sign / 100%).
- GPU pure-gauge HMC from a thermalized config
  (`inputs/hmc_fromthermal_b60.inp`, κ=1e-6): **stays** `<P>≈0.58`
  (was drifting to 0.78) — detailed balance restored.
- GPU quenched regression `inputs/quenched_4x4x4x4_b56.inp`: `<P>≈0.54`
  unchanged (heat-bath path untouched by the fix).

## State

- Branch `gpu-validation-comphys`. Working tree has the one-line-class
  source fix in `src/gpu/random_fields_gpu.cu` plus doc updates
  (CLAUDE.md pitfall #6 + status line, this file, the plan). **Not yet
  committed** — commit when ready.
- comphys: `run/lqcd_cpu_fixed.exe` and `run/lqcd_gpu.exe` are the
  fixed builds (09:48). `run/fixedcfg.bin` intact. Remote source ==
  local (verified per-file); only stale `.o` files referenced old
  diagnostics and were recompiled.
- Diagnostic-only inputs created on comphys (not in repo, harmless):
  `hmc_quenchedlimit_4x4x4x4.inp`, `hmc_fixedcfg_force.inp`,
  `hmc_dtscale_4x4x4x4.inp`, `hmc_fromthermal_b60.inp`,
  `rngfree_cg_4x4x4x8.inp`. `hmc_fromthermal_b60.inp` is worth keeping
  as a GPU-HMC sampler regression gate (consider adding to repo).

## If continuing GPU work

The GPU port is now fully validated end-to-end (fermion spectroscopy,
quenched heat bath, Wilson HMC). Remaining known-broken: **clover HMC
force** (CLAUDE.md pitfall #1, separate from this) — that is the natural
next GPU/physics target if desired. Audit other GPU RNG kernels with the
[[gpu-curand-thread-granularity]] lens (all others were already correct,
but it's the cheap recurring check).
