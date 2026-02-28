# B-Spline Collocation for Lattice QCD: Research Notes

## Overview

B-spline collocation methods have **not been applied to Lattice QCD** as of early 2026.
This represents a genuine gap in the literature and a potential avenue for original research.

The core idea: keep the standard lattice gauge action (SU(3) link variables) for Monte Carlo
sampling, but use B-spline collocation for the **fermion sector** — specifically, representing
quark fields in a B-spline basis and solving the Dirac equation via collocation rather than
the standard point-wise finite-difference discretization.

---

## Background

### Current Fermion Discretizations in LQCD

| Method | Chiral Symmetry | Discretization Error | Relative Cost |
|--------|----------------|---------------------|---------------|
| Wilson | Broken | O(a) | 1x (baseline) |
| Clover (Sheikholeslami-Wohlert) | Broken | O(a²) | ~1.5x |
| Staggered | Partial (U(1)) | O(a²) | ~0.3x |
| Domain Wall | Approximate | O(a²) + O(e^{-L_s}) | ~10-50x |
| Overlap (Neuberger) | Exact (Ginsparg-Wilson) | O(a²) | ~50-100x |

All of these use **point values of the fermion field on lattice sites** as degrees of freedom.
None use a basis-function expansion like B-splines.

### B-Spline Collocation: Key Properties

- B-splines of order k provide C^{k-2} smooth basis functions
- Compact support → sparse matrices (like lattice discretizations)
- Non-uniform knot sequences allow adaptive resolution
- Well-conditioned even for high-order approximations
- Widely used in computational physics (fluid dynamics, quantum mechanics, etc.)

---

## Promising Research Directions

### Direction 1: B-Spline Dirac Operator (Primary Target)

**Idea**: Replace the Wilson finite-difference Dirac operator with a B-spline collocation
discretization of the covariant derivative.

The Wilson Dirac operator is:
```
D_W(x,y) = δ_{x,y} - κ Σ_μ [(1-γ_μ) U_μ(x) δ_{y,x+μ} + (1+γ_μ) U_μ†(y) δ_{y,x-μ}]
```

A B-spline version would:
1. Expand ψ(x) = Σ_i c_i B_i(x) where B_i are B-spline basis functions
2. Evaluate the covariant derivative ∇_μ ψ = ∂_μ ψ + A_μ ψ using B-spline derivatives
3. Maintain gauge covariance by using link variables U_μ(x) for parallel transport
4. Solve the collocation equations D_B c = b for the B-spline coefficients

**Key advantage**: Higher-order accuracy without introducing fermion doublers (the doubling
problem is specific to the naive finite-difference discretization).

**Challenge**: Must prove that gauge covariance is maintained. The B-spline derivative
operators couple sites beyond nearest neighbors, so the parallel transport path must be
carefully defined (path-ordered exponentials along shortest lattice paths).

### Direction 2: Adaptive Resolution for Propagators

**Idea**: Use non-uniform B-spline knots that concentrate resolution near the quark source.

The quark propagator S(x,y) = D^{-1}(x,y) has a sharp peak at x ≈ y that falls off
exponentially. On a uniform lattice, you waste resolution on the far-field where the
propagator is exponentially small.

With B-splines:
- Dense knots near the source (high resolution where the propagator varies rapidly)
- Sparse knots far from source (lower resolution where propagator is smooth/small)
- Could reduce the effective matrix dimension significantly

**This is particularly relevant near κ_c** where the CG solver takes hundreds of iterations
because the Dirac matrix becomes nearly singular. The B-spline representation might provide
better preconditioning.

From our test data:
- κ = 0.120: ~97 CG iterations per inversion
- κ = 0.140: ~220 iterations
- κ = 0.150: ~425 iterations
- κ = 0.154: expected ~1000+ iterations (near κ_c ≈ 0.1577)

### Direction 3: Spectral Density Reconstruction

**Idea**: Use B-splines to represent spectral functions ρ(ω) extracted from Euclidean
correlators C(τ).

The relation C(τ) = ∫ dω K(τ,ω) ρ(ω) is an ill-posed inverse problem. Current methods
(Maximum Entropy Method, Backus-Gilbert) struggle with resolution and artifacts.

B-spline representations:
- Naturally positive-definite (if coefficients are positive)
- Smooth and well-conditioned
- Knot placement controls resolution adaptively
- Could provide better spectral function extraction than MEM

### Direction 4: Continuum Extrapolation

**Idea**: Use B-spline interpolation across lattice spacings for the a → 0 limit.

Standard approach: compute observables at 3-4 lattice spacings, fit polynomial in a².
B-spline interpolation could:
- Handle non-polynomial dependence on a
- Provide uncertainty bands from the spline fit
- Better identify and subtract lattice artifacts

---

## Implementation Plan

### Phase 1: Proof of Concept (using our existing code)
1. Implement 1D B-spline basis functions and derivatives
2. Construct B-spline Dirac operator for free-field case (U_μ = 1)
3. Verify: free-field pion mass should match analytic result
4. Compare discretization errors vs Wilson at same "effective lattice spacing"

### Phase 2: Full Gauge-Covariant Implementation
1. Define parallel transport for B-spline derivative operators
2. Implement gauge-covariant B-spline Dirac operator
3. Test on our β = 6.0 quenched configurations
4. Compare pion mass extraction: B-spline vs Wilson at same κ values

### Phase 3: Benchmarking & Paper
1. Scaling study: how do errors scale with lattice spacing?
2. Cost comparison: B-spline overhead vs accuracy gain
3. Critical κ behavior: does B-spline improve CG convergence near κ_c?
4. Write paper with benchmark results

---

## Challenges & Risks

1. **Gauge covariance**: The most fundamental requirement. If we can't maintain exact
   gauge invariance, the approach won't work for QCD. Need careful construction of the
   parallel transport paths for the extended stencil.

2. **Fermion doubling**: Nielsen-Ninomiya theorem says any chirally-symmetric, local,
   hermitian lattice Dirac operator must have doublers. B-splines change the "local"
   assumption since they have wider support — need to check if this helps or creates
   new issues.

3. **Computational cost**: B-spline evaluation adds FLOPs per site. The CG solver calls
   the Dirac operator ~100-1000 times, so any per-site overhead is amplified. Must show
   that accuracy gains outweigh the cost.

4. **Integration with Monte Carlo**: The fermion determinant det(D) appears in the
   path integral measure. Changing the Dirac operator changes the determinant and thus
   the physics. For quenched QCD (det(D) = 1) this isn't an issue, but for full QCD
   we need to ensure the B-spline determinant gives the correct continuum limit.

---

## Literature & References

- No direct references for B-spline + LQCD found (as of Feb 2026)
- Related: multigrid methods for Dirac equation (Brannick et al., adaptive algebraic multigrid)
- Related: spectral methods for lattice field theory
- Related: B-spline collocation for Schrödinger equation (de Boor, Höllig)
- Recent: LSDensities package for spectral density reconstruction from lattice data
- Recent: Neural preconditioners for Dirac operator (arXiv:2509.10378)
