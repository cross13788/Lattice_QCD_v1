//-----------------------------------------------
// Overrelaxation sweep over all lattice links
//
// Microcanonical (action-preserving) update for
// decorrelation. Uses checkerboard parallelism.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "lattice_module.hpp"
#include "interfaces_module.hpp"

void overrelaxation_sweep(SU3matrix* gaugeField)
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
                su3_overrelaxation_update(gaugeField[site * 4 + mu], staple);
            }
        }
    }
}
