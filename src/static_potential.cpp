//-----------------------------------------------
// Extract the static quark-antiquark potential
// V(R) from Wilson loop averages W(R,T).
//
// V(R) = -log( W(R,T) / W(R,T-1) )  for large T
//
// This gives the potential in lattice units.
//-----------------------------------------------
#include "constants_module.hpp"
#include "interfaces_module.hpp"
#include <cmath>
#include <cstdio>

void static_potential(const real_t* wilsonLoops, int rMax, int tMax,
                      real_t* potential)
{
//-----------------------------------------------
//   Extract V(R) using log ratio at T = tMax/2
//-----------------------------------------------
    int tRef = tMax / 2;
    if (tRef < 2) tRef = 2;
    if (tRef >= tMax) tRef = tMax - 1;

    for (int r = 0; r < rMax; r++) {
        real_t wt   = wilsonLoops[r * tMax + (tRef - 1)];  // W(R, T)
        real_t wt1  = wilsonLoops[r * tMax + tRef];         // W(R, T+1)

        if (wt > SMALL_VALUE && wt1 > SMALL_VALUE) {
            potential[r] = -std::log(wt1 / wt);
        } else {
            potential[r] = 0.0;
        }
    }
}
