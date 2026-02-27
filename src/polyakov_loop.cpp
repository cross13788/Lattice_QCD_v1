//-----------------------------------------------
// Polyakov loop measurement
//
// L = (1/NC) * (1/V_spatial) * sum_{x_spatial}
//     Re Tr[ prod_{t=0}^{nT-1} U(x,t; mu=3) ]
//
// Order parameter for the deconfinement transition.
// |<L>| ~ 0 in confined phase, nonzero in deconfined.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "lattice_module.hpp"
#include "interfaces_module.hpp"

void polyakov_loop(const SU3matrix* gaugeField, real_t& polyakovAvg)
{
//-----------------------------------------------
//   Local variables
//-----------------------------------------------
    real_t polySum = 0.0;

//-----------------------------------------------
//   Sum over spatial sites
//-----------------------------------------------
    #pragma omp parallel for reduction(+:polySum)
    for (int ix = 0; ix < nX; ix++) {
        for (int iy = 0; iy < nY; iy++) {
            for (int iz = 0; iz < nZ; iz++) {

                // Product of temporal links at this spatial position
                SU3matrix poly;
                su3_identity(poly);

                int site = site_index(ix, iy, iz, 0);
                for (int it = 0; it < nT; it++) {
                    SU3matrix tmp;
                    su3_multiply(poly, gaugeField[site * 4 + 3], tmp);
                    su3_copy(tmp, poly);
                    site = neighborPlus[site * 4 + 3];
                }

                polySum += su3_retrace(poly);
            }
        }
    }

    polyakovAvg = polySum / (NC * spatialVolume);
}
