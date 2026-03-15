//-----------------------------------------------
// Fermion force for HMC molecular dynamics
//
// The pseudofermion action is:
//   S_PF = φ† (D†D)^{-1} φ
//
// Its derivative with respect to U_mu(x) gives
// the fermion force:
//
//   F^F_mu(x) = -dS_PF/dA_mu(x)
//
// where A_mu(x) is the algebra-valued field.
//
// Using X = (D†D)^{-1} φ  (obtained by CG):
//
//   F^F_mu(x) = -kappa * sum over directions of
//     contributions from the hopping terms of D_W
//     evaluated at X.
//
// Specifically, for the Wilson-Dirac operator:
//   dS_PF/dU_mu(x) involves:
//     Tr_spin,color [ (r - gamma_mu) * X(x+mu) * Xbar(x)†
//                   + (r + gamma_mu) * Xbar(x) * X(x+mu)† ]
//
// where Xbar = D†^{-1} φ = D X, and the result
// is projected to traceless anti-Hermitian.
//
// More precisely:
//   F^F_mu(x) = kappa * [U_mu(x) * M_mu(x)]_{TA}
//
// where M_mu(x) is the outer product of the
// fermion fields contracted over Dirac indices
// with the appropriate gamma matrix structure.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "lattice_module.hpp"
#include "colorspinor_module.hpp"
#include "gamma_matrices.hpp"
#include "interfaces_module.hpp"
#include <cstdio>

void compute_fermion_force(const SU3matrix* gaugeField,
                           const ColorSpinor* phi,
                           SU3matrix* force,
                           int vol)
{
//-----------------------------------------------
//   Solve for X = (D†D)^{-1} φ via CG
//-----------------------------------------------
    ColorSpinor* X   = new ColorSpinor[vol];
    ColorSpinor* r   = new ColorSpinor[vol];
    ColorSpinor* p   = new ColorSpinor[vol];
    ColorSpinor* Ap  = new ColorSpinor[vol];
    ColorSpinor* tmp = new ColorSpinor[vol];

    // CG for D†D X = phi
    field_zero(X, vol);
    field_copy(phi, r, vol);
    field_copy(phi, p, vol);

    real_t rr = field_norm2(r, vol);
    real_t rr0 = rr;

    for (int iter = 0; iter < cgMaxIter; iter++) {
        apply_dirac(gaugeField, p, tmp, vol);
        apply_dirac_dagger(gaugeField, tmp, Ap, vol);

        complex_t pAp = field_dot(p, Ap, vol);
        real_t alpha = rr / pAp.real();

        field_accumulate(complex_t(alpha, 0.0), p, X, vol);
        field_accumulate(complex_t(-alpha, 0.0), Ap, r, vol);

        real_t rr_new = field_norm2(r, vol);
        if (std::sqrt(rr_new / rr0) < cgTolerance) break;

        real_t betaCG = rr_new / rr;
        #pragma omp parallel for
        for (int i = 0; i < vol; i++)
            for (int s = 0; s < ND; s++)
                for (int c = 0; c < NC; c++)
                    p[i].v[s][c] = r[i].v[s][c] + betaCG * p[i].v[s][c];
        rr = rr_new;
    }

//-----------------------------------------------
//   Compute Y = D X  (needed for force)
//-----------------------------------------------
    ColorSpinor* Y = new ColorSpinor[vol];
    apply_dirac(gaugeField, X, Y, vol);

//-----------------------------------------------
//   Compute fermion force
//
//   F^F_mu(x) = kappa * [U_mu(x) * M_mu(x)]_{TA}
//
//   where M_mu(x) is the color matrix:
//     M_mu(x)_{a,b} = sum_s [
//       (r - gamma_mu)_{s,s'} * Y(x+mu)_{s',b} * conj(X(x)_{s,a})
//     + (r + gamma_mu)_{s,s'} * X(x)_{s',b} * conj(Y(x+mu)_{s,a})
//     ]
//
//   But the standard derivation gives:
//   dS/dU_mu(x) = -kappa * sum_s [
//     (r - gamma_mu)_{s,s'} * X(x+mu)_{s'} ⊗ Y†(x)_{s}
//   + (r + gamma_mu)_{s,s'} * Y(x)_{s'} ⊗ X†(x+mu)_{s}  (wrong dir)
//   ] * U_mu(x)
//
//   Actually the correct derivation for the Wilson operator is:
//   The force from S_PF = X† D†D X = Y† Y where Y = DX gives:
//
//   F^F_mu(x) = kappa * { U_mu(x) * [
//     sum_s (r-gamma_mu)_{s,s'} *
//       ( X(x+mu)_{s'} ⊗ conj(Y(x)_{s}) )
//     ] }_{TA}
//   + kappa * { U_mu(x) * [
//     sum_s (r+gamma_mu)_{s,s'} *
//       ( Y(x+mu)_{s'} ⊗ conj(X(x)_{s}) )
//     ] }_{TA}   (from backward hop contribution)
//
//   Wait, let me be more careful. For dS/dU_mu(x):
//   S_PF = phi† (D†D)^{-1} phi = X† D† D X with phi = D†DX
//   dS/dU = -(X† dD†/dU D X + X† D† dD/dU X)
//         = -(Y† dD/dU X + (dD/dU X)† Y)  ... hermitian
//
//   dD_W/dU_mu(x) acting on psi contributes:
//     at site x:   -kappa * (r - gamma_mu) * delta_U * psi(x+mu)
//     at site x+mu: -kappa * (r + gamma_mu) * delta_U† * psi(x)
//
//   So:
//   dS/dU_mu(x) = kappa * [Y†(x) (r-gamma_mu) psi(x+mu)]
//               + kappa * [Y†(x+mu) (r+gamma_mu) psi(x)]†
//   where psi = X (the solution vector).
//
//   In matrix form (color indices only):
//   dS/dU_mu(x) gives F_mu(x):
//
//   F_mu(x) = kappa * U_mu(x) * sum_s {
//     [(r - gamma_mu) X(x+mu)]_s ⊗ [Y(x)]†_s
//   + [Y(x+mu)]_s ⊗ [(r + gamma_mu) X(x)]†_s
//   }_{color matrix, projected TA}
//
//   Hmm, this is getting notationally heavy.
//   Let me implement the standard form directly.
//-----------------------------------------------

    #pragma omp parallel for
    for (int site = 0; site < vol; site++) {
        for (int mu = 0; mu < 4; mu++) {
            int idx = site * 4 + mu;
            int sitePlus = neighborPlus[site * 4 + mu];

            // Build the color matrix M_mu(x)
            SU3matrix M;
            su3_zero(M);

            for (int s = 0; s < ND; s++) {
                int gs = gammaIdx[mu][s];
                complex_t gv = gammaVal[mu][s];

                for (int a = 0; a < NC; a++) {
                    // (r - gamma_mu) applied to X(x+mu)
                    complex_t fwd_s_a = wilsonR * X[sitePlus].v[s][a]
                                       - gv * X[sitePlus].v[gs][a];

                    for (int b = 0; b < NC; b++) {
                        // Outer product: fwd * Y†(x)
                        M.m[a][b] += fwd_s_a * std::conj(Y[site].v[s][b]);
                    }
                }
            }

            // Second term: from the backward hopping
            // This contributes at site x from the perspective
            // of the link U_mu(x), involving Y at x+mu and X at x.
            for (int s = 0; s < ND; s++) {
                int gs = gammaIdx[mu][s];
                complex_t gv = gammaVal[mu][s];

                for (int a = 0; a < NC; a++) {
                    complex_t bwd_s_a = wilsonR * Y[sitePlus].v[s][a]
                                       + gv * Y[sitePlus].v[gs][a];

                    for (int b = 0; b < NC; b++) {
                        M.m[a][b] += std::conj(X[site].v[s][b]) * bwd_s_a;
                    }
                }
            }

            // F = -2 * kappa * [U * M]_TA
            // Convention: dS/deps = -Re Tr(X * F) (matching gauge force)
            SU3matrix UM, forceTA;
            su3_multiply(gaugeField[idx], M, UM);
            su3_traceless_antiherm(UM, forceTA);
            su3_scale(complex_t(-2.0 * kappa, 0.0), forceTA, force[idx]);
        }
    }

//-----------------------------------------------
//   Clover force contribution (when enabled)
//-----------------------------------------------
    if (useClover) {
        compute_clover_force(gaugeField, X, Y, force, vol);
    }

//-----------------------------------------------
//   Cleanup
//-----------------------------------------------
    delete[] X;
    delete[] Y;
    delete[] r;
    delete[] p;
    delete[] Ap;
    delete[] tmp;
}
