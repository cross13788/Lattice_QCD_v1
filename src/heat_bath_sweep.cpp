//-----------------------------------------------
// Heat bath sweep over all lattice links
//
// Uses checkerboard (even/odd) decomposition:
// update all even sites first (in parallel), then
// all odd sites. This ensures no two simultaneously
// updated links share a staple.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "lattice_module.hpp"
#include "interfaces_module.hpp"

void heat_bath_sweep(SU3matrix* gaugeField)
{
//-----------------------------------------------
//   Loop over even sites, then odd sites
//-----------------------------------------------
    for (int parity = 0; parity < 2; parity++) {
        #pragma omp parallel for
        for (int site = 0; site < latticeVolume; site++) {
            if (site_parity(site) != parity) continue;

            for (int mu = 0; mu < 4; mu++) {
                SU3matrix staple;
                compute_staple(gaugeField, site, mu, staple);
                su3_heat_bath_update(gaugeField[site * 4 + mu], staple, beta);
            }
        }
    }
}
