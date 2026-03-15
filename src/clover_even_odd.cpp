//-----------------------------------------------
// Even-odd preconditioning for clover fermions
//
// With clover improvement, the diagonal blocks are
// no longer identity:
//   D_ee = 1 + A_ee,  D_oo = 1 + A_oo
//
// The preconditioned operator becomes:
//   D_hat = 1 - kappa^2 * D_ee^{-1} * D_eo * D_oo^{-1} * D_oe
//
// where D_ee^{-1} = (1 + A_ee)^{-1} and D_oo^{-1} = (1 + A_oo)^{-1}
// are precomputed 6x6 block inverses.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "lattice_module.hpp"
#include "colorspinor_module.hpp"
#include "gamma_matrices.hpp"
#include "clover_module.hpp"
#include "interfaces_module.hpp"

//-----------------------------------------------
// Apply (1+A)^{-1} at a single site
// Multiplies the inverse clover block times spinor
//-----------------------------------------------
static void apply_clover_inverse_site(const CloverField& Ainv,
                                       const ColorSpinor& psi,
                                       ColorSpinor& result)
{
    // Upper block: spin 0,1
    for (int s1 = 0; s1 < 2; s1++) {
        for (int a = 0; a < NC; a++) {
            int row = s1 * NC + a;
            complex_t sum(0.0, 0.0);
            for (int s2 = 0; s2 < 2; s2++) {
                for (int b = 0; b < NC; b++) {
                    int col = s2 * NC + b;
                    sum += Ainv.upper.m[row][col] * psi.v[s2][b];
                }
            }
            result.v[s1][a] = sum;
        }
    }

    // Lower block: spin 2,3
    for (int s1 = 2; s1 < 4; s1++) {
        for (int a = 0; a < NC; a++) {
            int row = (s1 - 2) * NC + a;
            complex_t sum(0.0, 0.0);
            for (int s2 = 2; s2 < 4; s2++) {
                for (int b = 0; b < NC; b++) {
                    int col = (s2 - 2) * NC + b;
                    sum += Ainv.lower.m[row][col] * psi.v[s2][b];
                }
            }
            result.v[s1][a] = sum;
        }
    }
}

//-----------------------------------------------
// Apply clover-preconditioned operator:
//
//   D_hat * psi = psi - kappa^2 * D_ee^{-1} * D_eo * D_oo^{-1} * D_oe * psi
//
// where D_eo, D_oe are the standard hopping terms.
// psi and result live on even sites only.
//-----------------------------------------------
void apply_clover_preconditioned_operator(
    const SU3matrix* gaugeField,
    const CloverField* cloverFieldInv,
    const ColorSpinor* psi,
    ColorSpinor* result,
    int vol)
{
    ColorSpinor* tmp_odd  = new ColorSpinor[vol];
    ColorSpinor* tmp_even = new ColorSpinor[vol];
    ColorSpinor* tmp2     = new ColorSpinor[vol];

    // Step 1: D_oe * psi (even → odd)
    apply_Deo(gaugeField, psi, tmp_odd, vol);

    // Step 2: D_oo^{-1} * tmp_odd (apply inverse clover on odd sites)
    #pragma omp parallel for
    for (int site = 0; site < vol; site++) {
        if (site_parity(site) != 1) continue;
        apply_clover_inverse_site(cloverFieldInv[site], tmp_odd[site], tmp2[site]);
    }

    // Step 3: D_eo * tmp2 (odd → even)
    apply_Doe(gaugeField, tmp2, tmp_even, vol);

    // Step 4: D_ee^{-1} * tmp_even (apply inverse clover on even sites)
    // Then: result = psi - tmp_result
    #pragma omp parallel for
    for (int site = 0; site < vol; site++) {
        if (site_parity(site) != 0) continue;
        ColorSpinor invResult;
        apply_clover_inverse_site(cloverFieldInv[site], tmp_even[site], invResult);
        for (int s = 0; s < ND; s++)
            for (int c = 0; c < NC; c++)
                result[site].v[s][c] = psi[site].v[s][c] + invResult.v[s][c];
    }

    delete[] tmp_odd;
    delete[] tmp_even;
    delete[] tmp2;
}
