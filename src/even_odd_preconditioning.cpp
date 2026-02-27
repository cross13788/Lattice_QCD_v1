//-----------------------------------------------
// Even-odd (red-black) preconditioning for
// the Wilson-Dirac operator.
//
// The Wilson-Dirac operator in even-odd form:
//
//   D_W = ( D_ee   D_eo )   ( 1       D_eo )
//         ( D_oe   D_oo ) = ( D_oe    1    )
//
// where D_ee = D_oo = 1 (identity) for standard
// Wilson fermions, and D_eo, D_oe are the hopping
// terms connecting even-to-odd and odd-to-even.
//
// The preconditioned operator on even sites:
//   Dhat = 1 - kappa^2 * D_eo * D_oe
//
// This halves the effective system size for the
// solver, improving convergence.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "lattice_module.hpp"
#include "colorspinor_module.hpp"
#include "gamma_matrices.hpp"
#include "interfaces_module.hpp"

//-----------------------------------------------
// Apply the hopping term from one sublattice to
// the other: result(x) = sum_mu hop terms from
// neighbors of x.
//
// This is the off-diagonal piece of D_W (without
// the diagonal identity). Input psi lives on the
// source sublattice, result on the target.
//
// sourceParity: parity of the input field (0=even, 1=odd)
//-----------------------------------------------

static void apply_hopping(const SU3matrix* gaugeField,
                           const ColorSpinor* psi,
                           ColorSpinor* result,
                           int sourceParity,
                           int vol)
{
//-----------------------------------------------
//   Loop over sites of the OPPOSITE parity
//   (target sites)
//-----------------------------------------------
    int targetParity = 1 - sourceParity;

    #pragma omp parallel for
    for (int site = 0; site < vol; site++) {
        if (site_parity(site) != targetParity) continue;

        cs_zero(result[site]);

        for (int mu = 0; mu < 4; mu++) {
            int sitePlus  = neighborPlus[site * 4 + mu];
            int siteMinus = neighborMinus[site * 4 + mu];

            const SU3matrix& Ufwd = gaugeField[site * 4 + mu];
            SU3matrix Ubwd;
            su3_dagger(gaugeField[siteMinus * 4 + mu], Ubwd);

            // Color-rotate neighbors
            ColorSpinor Upsi_fwd, Upsi_bwd;

            for (int s = 0; s < ND; s++) {
                for (int a = 0; a < NC; a++) {
                    Upsi_fwd.v[s][a] = complex_t(0.0, 0.0);
                    Upsi_bwd.v[s][a] = complex_t(0.0, 0.0);
                    for (int b = 0; b < NC; b++) {
                        Upsi_fwd.v[s][a] += Ufwd.m[a][b] * psi[sitePlus].v[s][b];
                        Upsi_bwd.v[s][a] += Ubwd.m[a][b] * psi[siteMinus].v[s][b];
                    }
                }
            }

            for (int s = 0; s < ND; s++) {
                int gIdx_s = gammaIdx[mu][s];
                complex_t gVal_s = gammaVal[mu][s];

                for (int a = 0; a < NC; a++) {
                    // Forward: -kappa * (r - gamma_mu) * U * psi(x+mu)
                    result[site].v[s][a] -= kappa * (
                        wilsonR * Upsi_fwd.v[s][a] - gVal_s * Upsi_fwd.v[gIdx_s][a]
                    );
                    // Backward: -kappa * (r + gamma_mu) * U^dag * psi(x-mu)
                    result[site].v[s][a] -= kappa * (
                        wilsonR * Upsi_bwd.v[s][a] + gVal_s * Upsi_bwd.v[gIdx_s][a]
                    );
                }
            }
        }
    }
}

//-----------------------------------------------
// Apply D_eo: even-to-odd hopping
// Input: psi on even sites, output on odd sites
//-----------------------------------------------
void apply_Deo(const SU3matrix* gaugeField,
               const ColorSpinor* psi,
               ColorSpinor* result,
               int vol)
{
    apply_hopping(gaugeField, psi, result, 0, vol);
}

//-----------------------------------------------
// Apply D_oe: odd-to-even hopping
// Input: psi on odd sites, output on even sites
//-----------------------------------------------
void apply_Doe(const SU3matrix* gaugeField,
               const ColorSpinor* psi,
               ColorSpinor* result,
               int vol)
{
    apply_hopping(gaugeField, psi, result, 1, vol);
}

//-----------------------------------------------
// Apply preconditioned operator:
//   Dhat * psi = psi - kappa^2 * D_eo * D_oe * psi
//
// Acts only on even sites. The input and output
// fields are full-volume but only even-site data
// is meaningful.
//-----------------------------------------------
void apply_preconditioned_operator(const SU3matrix* gaugeField,
                                    const ColorSpinor* psi,
                                    ColorSpinor* result,
                                    int vol)
{
//-----------------------------------------------
//   Temporary fields
//-----------------------------------------------
    ColorSpinor* tmp_odd  = new ColorSpinor[vol];
    ColorSpinor* tmp_even = new ColorSpinor[vol];

//-----------------------------------------------
//   Step 1: tmp_odd = D_oe * psi  (odd-to-even hop reads even psi, writes odd)
//   Wait — D_oe takes odd input and produces even output.
//   We need D_eo(D_oe(psi_even)):
//     Step 1: tmp_odd = hop from even psi to odd sites
//     Step 2: tmp_even = hop from odd tmp to even sites
//-----------------------------------------------
    // D_oe acts on even sites (psi) to produce odd sites
    // But our convention: D_oe = hopping from odd→even
    // We want: D_eo * D_oe * psi_even
    //   D_oe * psi_even: hops from even (source=even) to produce on odd
    //   D_eo * result:   hops from odd (source=odd) to produce on even

    // Step 1: apply hopping from even → odd
    apply_hopping(gaugeField, psi, tmp_odd, 0, vol);

    // Step 2: apply hopping from odd → even
    apply_hopping(gaugeField, tmp_odd, tmp_even, 1, vol);

//-----------------------------------------------
//   result(even) = psi(even) + tmp_even(even)
//   (the hopping already includes -kappa factors,
//    so D_eo*D_oe has kappa^2 built in)
//-----------------------------------------------
    #pragma omp parallel for
    for (int site = 0; site < vol; site++) {
        if (site_parity(site) != 0) continue;
        for (int s = 0; s < ND; s++)
            for (int c = 0; c < NC; c++)
                result[site].v[s][c] = psi[site].v[s][c] + tmp_even[site].v[s][c];
    }

//-----------------------------------------------
//   Cleanup
//-----------------------------------------------
    delete[] tmp_odd;
    delete[] tmp_even;
}

//-----------------------------------------------
// Reconstruct odd-site solution from even-site solution.
//
// Given the even-site solution x_e from the
// preconditioned system Dhat * x_e = bhat_e, the
// full solution is obtained by:
//   x_o = b_o - D_oe * x_e
//
// where b is the original right-hand side.
//-----------------------------------------------
void reconstruct_odd_sites(const SU3matrix* gaugeField,
                           const ColorSpinor* x_even,
                           const ColorSpinor* b,
                           ColorSpinor* x,
                           int vol)
{
//-----------------------------------------------
//   Copy even solution
//-----------------------------------------------
    #pragma omp parallel for
    for (int site = 0; site < vol; site++) {
        if (site_parity(site) == 0) {
            cs_copy(x_even[site], x[site]);
        }
    }

//-----------------------------------------------
//   Compute hopping contribution on odd sites
//-----------------------------------------------
    ColorSpinor* tmp = new ColorSpinor[vol];
    apply_hopping(gaugeField, x_even, tmp, 0, vol);

//-----------------------------------------------
//   x_o = b_o + tmp_o
//   (tmp already has the right signs from hopping)
//-----------------------------------------------
    #pragma omp parallel for
    for (int site = 0; site < vol; site++) {
        if (site_parity(site) != 1) continue;
        for (int s = 0; s < ND; s++)
            for (int c = 0; c < NC; c++)
                x[site].v[s][c] = b[site].v[s][c] + tmp[site].v[s][c];
    }

    delete[] tmp;
}
