//-----------------------------------------------
// Compute average plaquette
//
// <P> = (1 / (6*V)) * sum_{x} sum_{mu>nu}
//       (1/NC) * Re Tr[ U(x,mu) U(x+mu,nu)
//                        U^dag(x+nu,mu) U^dag(x,nu) ]
//
// The plaquette ranges from 0 (fully disordered)
// to 1 (fully ordered / cold start).
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "lattice_module.hpp"
#include "interfaces_module.hpp"
#include <cstdio>

void compute_plaquette(const SU3matrix* gaugeField, real_t& avgPlaquette)
{
//-----------------------------------------------
//   Local variables
//-----------------------------------------------
    real_t plaqSum = 0.0;

//-----------------------------------------------
//   Sum over all sites and plane orientations
//-----------------------------------------------
    #pragma omp parallel for reduction(+:plaqSum)
    for (int site = 0; site < latticeVolume; site++) {
        for (int mu = 0; mu < 4; mu++) {
            for (int nu = mu + 1; nu < 4; nu++) {
                // P = U(x,mu) * U(x+mu,nu) * U^dag(x+nu,mu) * U^dag(x,nu)
                int sitePlusMu = neighborPlus[site * 4 + mu];
                int sitePlusNu = neighborPlus[site * 4 + nu];

                SU3matrix tmp1, tmp2, tmp3, plaq;

                // tmp1 = U(x,mu) * U(x+mu,nu)
                su3_multiply(gaugeField[site * 4 + mu],
                             gaugeField[sitePlusMu * 4 + nu],
                             tmp1);

                // tmp2 = U^dag(x+nu,mu)
                su3_dagger(gaugeField[sitePlusNu * 4 + mu], tmp2);

                // tmp3 = tmp1 * tmp2 = U(x,mu) * U(x+mu,nu) * U^dag(x+nu,mu)
                su3_multiply(tmp1, tmp2, tmp3);

                // plaq = tmp3 * U^dag(x,nu)
                SU3matrix UdagNu;
                su3_dagger(gaugeField[site * 4 + nu], UdagNu);
                su3_multiply(tmp3, UdagNu, plaq);

                plaqSum += su3_retrace(plaq);
            }
        }
    }

//-----------------------------------------------
//   Normalize: 6 planes, V sites, NC for trace
//-----------------------------------------------
    avgPlaquette = plaqSum / (6.0 * latticeVolume * NC);
}
