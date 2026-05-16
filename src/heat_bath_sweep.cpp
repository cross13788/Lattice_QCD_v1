//-----------------------------------------------
// Heat bath sweep over all lattice links
//
// mu-outer, then even/odd site parity. Site parity
// alone is NOT a valid coloring when a thread updates
// all 4 directions: (x,mu) and (x+mu-nu,nu) share the
// negative-staple plaquette and have the same site
// parity. Fixing mu per pass makes the concurrently-
// updated link set share no plaquette.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "lattice_module.hpp"
#include "interfaces_module.hpp"

void heat_bath_sweep(SU3matrix* gaugeField)
{
//-----------------------------------------------
//   mu-outer, then even/odd sites (race-free)
//-----------------------------------------------
    for (int mu = 0; mu < 4; mu++) {
        for (int parity = 0; parity < 2; parity++) {
            #pragma omp parallel for
            for (int site = 0; site < latticeVolume; site++) {
                if (site_parity(site) != parity) continue;

                SU3matrix staple;
                compute_staple(gaugeField, site, mu, staple);
                su3_heat_bath_update(gaugeField[site * 4 + mu], staple, beta);
            }
        }
    }
}
