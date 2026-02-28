# Lattice QCD Validation Tests

## 1. Systematic Pure Gauge Validation

### 1a. Plaquette vs Beta
Run quenched simulations at multiple beta values on an 8^4 lattice and compare average plaquette to published values.

| Beta | Expected <P> |
|------|-------------|
| 5.6  | ~0.548      |
| 5.8  | ~0.571      |
| 6.0  | ~0.593-0.594|
| 6.2  | ~0.613      |

**Input file template**: Use `Update Method: heatbath`, `Start Type: hot`, `Thermalization Sweeps: 500`, `Number of Configurations: 100`, `Sweeps Between Measurements: 10`. Measure plaquette and Polyakov loop.

### 1b. Wilson Loops and Static Potential
Turn on Wilson loop measurements (`Measure Wilson Loops: on`) with `Wilson Loop R Max: 4`, `Wilson Loop T Max: 4` on an 8^4 or 12^3x24 lattice at beta=6.0. Extract V(R) from:

```
V(R) = -log(W(R,T) / W(R,T+1))
```

at large T. Should see Cornell potential: `V(R) = -alpha/R + sigma*R + c` with string tension sigma ~ (440 MeV)^2 in physical units.

### 1c. Polyakov Loop and Deconfinement
Compare Polyakov loop on asymmetric lattices at beta=6.0:
- Confined: 8^3 x 8 (low temperature) -> <|P|> ~ 0
- Deconfined: 8^3 x 4 (high temperature) -> <|P|> > 0

The deconfinement transition for SU(3) pure gauge is at T_c ~ 270 MeV.

---

## 2. Fermion Validation

### 2a. Free-Field Propagator Test
Set `Start Type: cold` (all links = identity), `Thermalization Sweeps: 0`, `Number of Configurations: 1`, `Measure Correlators: on`. The quark propagator on a free field has an analytic form. Check that the pion correlator matches the free-field result:

```
C_pi(t) = sum_x <pi(x,t) pi(0,0)>
```

With all links = identity and kappa = 0.12, the effective mass should be constant in t and match the analytic free-field value:

```
cosh(m_free) = 1 + (1 - 2*kappa*(4*r + m_0)) / (2*kappa)
```

where m_0 = (1/(2*kappa) - 4*r) for Wilson fermions with r=1.

### 2b. Pion Mass vs Kappa (Quenched)
At fixed beta=6.0 on an 8^3 x 16 lattice, generate quenched configurations with heatbath. Then measure meson correlators at several kappa values:

| Kappa | Expected behavior |
|-------|------------------|
| 0.120 | Heavy pion        |
| 0.140 | Moderate pion     |
| 0.150 | Lighter pion      |
| 0.154 | Near kappa_c, m_pi -> 0 |

The pion mass squared should be approximately linear in 1/kappa (PCAC relation):

```
m_pi^2 ~ (1/kappa - 1/kappa_c)
```

### 2c. CG/BiCGstab Convergence
Verify solver convergence:
- Residual should decrease monotonically (CG) or roughly monotonically (BiCGstab)
- Heavier quarks (smaller kappa) converge faster
- Near kappa_c, iteration count grows significantly (critical slowing down)

---

## 3. HMC Validation

### 3a. Energy Conservation (Already Passed)
Leapfrog integrator conserves energy to O(epsilon^2):
- epsilon=0.001: |deltaH| ~ O(10^-6)
- epsilon=0.01: |deltaH| ~ O(10^-4)
- Ratio should be ~100 (quadratic scaling)

### 3b. Reversibility (Already Passed)
Forward N steps + backward N steps returns to initial configuration at machine precision (~10^-15).

### 3c. Acceptance Rate Tuning
Target 70-80% acceptance. Tune `MD Step Size` for a given `MD Steps`:
- Too large epsilon: low acceptance (large deltaH)
- Too small epsilon: wasted computation (deltaH ~ 0 but many force evaluations)

Run HMC with different step sizes and check:

| MD Steps | Step Size | Expected Acceptance |
|----------|-----------|-------------------|
| 10       | 0.01      | >90% (over-tuned) |
| 10       | 0.02      | ~80-90%           |
| 10       | 0.05      | ~50-70%           |
| 10       | 0.10      | <30% (too large)  |

### 3d. Large Mass Limit
At very heavy quark mass (kappa ~ 0.05-0.10), HMC results should match pure gauge (quenched) results since the fermion determinant becomes nearly constant. Compare plaquette values between:
- Pure gauge heatbath at beta=5.6
- HMC at beta=5.6, kappa=0.05

They should agree within statistics.

### 3e. Plaquette Thermalization (HMC)
From hot start, plaquette should thermalize to a stable value. From cold start, it should thermalize to the same value. Both should agree within errors.

---

## 4. Analysis Scripts to Write

### 4a. `analysis/plot_plaquette.py`
- Read plaquette data from output files
- Plot thermalization history (plaquette vs sweep/trajectory)
- Compute equilibrium average with jackknife errors
- Identify thermalization point (discard early sweeps)

### 4b. `analysis/static_potential.py`
- Read Wilson loop data W(R,T)
- Extract V(R) = -log(W(R,T)/W(R,T+1)) at large T
- Fit Cornell potential: V(R) = -alpha/R + sigma*R + c
- Plot V(R) with fit

### 4c. `analysis/effective_mass.py`
- Read correlator data C(t)
- Compute m_eff(t) = log(C(t)/C(t+1)) or cosh version
- Plot m_eff(t) and identify plateau region
- Extract mass from plateau average with jackknife errors

### 4d. `analysis/jackknife_analysis.py`
- Delete-1 jackknife for error estimation
- Binned jackknife for handling autocorrelations
- Reusable functions for all other analysis scripts

---

## 5. Input Files to Create

For each test above, create a corresponding input file in `inputs/`:

- `inputs/quenched_8x8x8x8_b56.inp` — Pure gauge beta=5.6
- `inputs/quenched_8x8x8x8_b60.inp` — Pure gauge beta=6.0
- `inputs/quenched_8x8x8x8_b62.inp` — Pure gauge beta=6.2
- `inputs/wilson_loops_12x12x12x24_b60.inp` — Wilson loops for static potential
- `inputs/free_field_8x8x8x16.inp` — Free-field fermion test
- `inputs/quenched_fermions_8x8x8x16_b60.inp` — Quenched meson spectroscopy
- `inputs/hmc_8x8x8x8_b56.inp` — HMC larger lattice
- `inputs/hmc_acceptance_scan.inp` — Template for step size scan

---

## Suggested Order of Execution

1. Pure gauge plaquette vs beta (1a) — quickest physics validation
2. Analysis script: plot_plaquette.py (4a) — needed to analyze everything
3. Wilson loops + static potential (1b) + analysis (4b)
4. Free-field fermion test (2a) — validates Dirac operator
5. Quenched meson correlators (2b) + effective mass analysis (4c)
6. HMC acceptance tuning (3c)
7. HMC large mass limit (3d)
8. Polyakov loop / deconfinement (1c)
