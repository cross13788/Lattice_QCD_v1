//-----------------------------------------------
// Wilson loop measurement W(R,T)
//
// Computes rectangular R x T Wilson loops in
// the spatial-temporal plane. The loop is:
//
// W(R,T) = (1/3) * (1/V_spatial) * sum_x
//   Re Tr[ P_spatial(x, R) * P_temporal(x+R, T)
//        * P_spatial^dag(x+T, R) * P_temporal^dag(x, T) ]
//
// where P_spatial is a product of R spatial links
// and P_temporal is a product of T temporal links.
//
// We average over all 3 spatial directions.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "lattice_module.hpp"
#include "interfaces_module.hpp"
#include <cstdlib>

void wilson_loop(const SU3matrix* gaugeField, real_t* wilsonLoops, int rMax, int tMax)
{
//-----------------------------------------------
//   Zero output array
//-----------------------------------------------
    for (int i = 0; i < rMax * tMax; i++)
        wilsonLoops[i] = 0.0;

//-----------------------------------------------
//   Loop over 3 spatial directions
//-----------------------------------------------
    for (int spatialDir = 0; spatialDir < 3; spatialDir++) {
        int muS = spatialDir;   // spatial direction
        int muT = 3;            // temporal direction

        //-----------------------------------------------
        // For each R, T value
        //-----------------------------------------------
        for (int R = 1; R <= rMax; R++) {
            for (int T = 1; T <= tMax; T++) {

                real_t wlSum = 0.0;

                #pragma omp parallel for reduction(+:wlSum)
                for (int site = 0; site < latticeVolume; site++) {

                    // Build spatial path of length R starting at site
                    SU3matrix pathSpatial;
                    su3_identity(pathSpatial);
                    int currentSite = site;
                    for (int r = 0; r < R; r++) {
                        SU3matrix tmp;
                        su3_multiply(pathSpatial, gaugeField[currentSite * 4 + muS], tmp);
                        su3_copy(tmp, pathSpatial);
                        currentSite = neighborPlus[currentSite * 4 + muS];
                    }
                    // currentSite = site + R*hat_s

                    // Build temporal path of length T starting at site+R
                    SU3matrix pathTemporal;
                    su3_identity(pathTemporal);
                    int sitePlusR = currentSite;
                    currentSite = sitePlusR;
                    for (int t = 0; t < T; t++) {
                        SU3matrix tmp;
                        su3_multiply(pathTemporal, gaugeField[currentSite * 4 + muT], tmp);
                        su3_copy(tmp, pathTemporal);
                        currentSite = neighborPlus[currentSite * 4 + muT];
                    }

                    // Build spatial path^dag of length R starting at site+T
                    // This is the reverse path: U^dag(x+T+(R-1),s) * ... * U^dag(x+T,s)
                    SU3matrix pathSpatialBack;
                    su3_identity(pathSpatialBack);
                    // Start from site + T*hat_t
                    currentSite = site;
                    for (int t = 0; t < T; t++)
                        currentSite = neighborPlus[currentSite * 4 + muT];
                    // currentSite = site + T*hat_t
                    // Walk R steps in spatial direction, accumulate product
                    int sitePlusT = currentSite;
                    for (int r = 0; r < R; r++) {
                        SU3matrix tmp;
                        su3_multiply(pathSpatialBack, gaugeField[currentSite * 4 + muS], tmp);
                        su3_copy(tmp, pathSpatialBack);
                        currentSite = neighborPlus[currentSite * 4 + muS];
                    }
                    // Take dagger
                    SU3matrix pathSpatialBackDag;
                    su3_dagger(pathSpatialBack, pathSpatialBackDag);

                    // Build temporal path^dag of length T starting at site
                    SU3matrix pathTemporalOrig;
                    su3_identity(pathTemporalOrig);
                    currentSite = site;
                    for (int t = 0; t < T; t++) {
                        SU3matrix tmp;
                        su3_multiply(pathTemporalOrig, gaugeField[currentSite * 4 + muT], tmp);
                        su3_copy(tmp, pathTemporalOrig);
                        currentSite = neighborPlus[currentSite * 4 + muT];
                    }
                    SU3matrix pathTemporalOrigDag;
                    su3_dagger(pathTemporalOrig, pathTemporalOrigDag);

                    // W = pathSpatial * pathTemporal * pathSpatialBackDag * pathTemporalOrigDag
                    SU3matrix tmp1, tmp2, loop;
                    su3_multiply(pathSpatial, pathTemporal, tmp1);
                    su3_multiply(tmp1, pathSpatialBackDag, tmp2);
                    su3_multiply(tmp2, pathTemporalOrigDag, loop);

                    wlSum += su3_retrace(loop);
                }

                // Normalize by NC * volume
                wilsonLoops[(R-1) * tMax + (T-1)] += wlSum / (NC * latticeVolume);
            }
        }
    }

    // Average over 3 spatial directions
    for (int i = 0; i < rMax * tMax; i++)
        wilsonLoops[i] /= 3.0;
}
