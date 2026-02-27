//-----------------------------------------------
// Wilson-Dirac operator: D_W * psi = result
//
// D_W(x,y) = delta(x,y)
//   - kappa * sum_mu [ (r - gamma_mu) U_mu(x) delta(y, x+mu)
//                    + (r + gamma_mu) U^dag_mu(x-mu) delta(y, x-mu) ]
//
// This is the most performance-critical routine in
// the fermion sector. OpenMP parallelism over sites.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "lattice_module.hpp"
#include "colorspinor_module.hpp"
#include "gamma_matrices.hpp"
#include "interfaces_module.hpp"

void wilson_dirac_operator(const SU3matrix* gaugeField,
                           const ColorSpinor* psi,
                           ColorSpinor* result,
                           int vol)
{
//-----------------------------------------------
//   Parallel loop over all sites
//-----------------------------------------------
    #pragma omp parallel for
    for (int site = 0; site < vol; site++) {

    //-----------------------------------------------
    //   Start with diagonal term: result = psi
    //-----------------------------------------------
        cs_copy(psi[site], result[site]);

    //-----------------------------------------------
    //   Hopping terms: sum over 4 directions
    //-----------------------------------------------
        for (int mu = 0; mu < 4; mu++) {

            int sitePlus  = neighborPlus[site * 4 + mu];
            int siteMinus = neighborMinus[site * 4 + mu];

            // Forward hop: -kappa * (r - gamma_mu) * U_mu(x) * psi(x+mu)
            // Backward hop: -kappa * (r + gamma_mu) * U^dag_mu(x-mu) * psi(x-mu)

            const SU3matrix& Ufwd = gaugeField[site * 4 + mu];
            SU3matrix Ubwd;
            su3_dagger(gaugeField[siteMinus * 4 + mu], Ubwd);

            // Compute U_mu(x) * psi(x+mu) and U^dag_mu(x-mu) * psi(x-mu)
            // These are color-rotated spinors
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

            // Apply (r - gamma_mu) to forward and (r + gamma_mu) to backward
            // Using sparse gamma representation:
            //   (r - gamma_mu)_{s,s'} = r * delta_{s,s'} - gammaVal[mu][s] * delta_{s', gammaIdx[mu][s]}
            //   (r + gamma_mu)_{s,s'} = r * delta_{s,s'} + gammaVal[mu][s] * delta_{s', gammaIdx[mu][s]}

            int gIdx_s;
            complex_t gVal_s;

            for (int s = 0; s < ND; s++) {
                gIdx_s = gammaIdx[mu][s];
                gVal_s = gammaVal[mu][s];

                for (int a = 0; a < NC; a++) {
                    // Forward: -kappa * [(r * Upsi_fwd[s][a]) - (gVal * Upsi_fwd[gIdx][a])]
                    result[site].v[s][a] -= kappa * (
                        wilsonR * Upsi_fwd.v[s][a] - gVal_s * Upsi_fwd.v[gIdx_s][a]
                    );

                    // Backward: -kappa * [(r * Upsi_bwd[s][a]) + (gVal * Upsi_bwd[gIdx][a])]
                    result[site].v[s][a] -= kappa * (
                        wilsonR * Upsi_bwd.v[s][a] + gVal_s * Upsi_bwd.v[gIdx_s][a]
                    );
                }
            }
        }
    }
}
