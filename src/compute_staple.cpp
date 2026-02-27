//-----------------------------------------------
// Compute the staple sum for a given link U(x,mu)
//
// The staple is the sum of the 6 "staples" (the
// remaining 3 links of each plaquette containing
// the link U(x,mu)). There are 3 positive and
// 3 negative orientation staples.
//
// R(x,mu) = sum_{nu != mu} [
//   U(x+mu,nu) * U^dag(x+nu,mu) * U^dag(x,nu)          (positive)
// + U^dag(x+mu-nu,nu) * U^dag(x-nu,mu) * U(x-nu,nu)    (negative)
// ]
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "lattice_module.hpp"
#include "interfaces_module.hpp"

void compute_staple(const SU3matrix* gaugeField, int site, int mu, SU3matrix& staple)
{
//-----------------------------------------------
//   Local variables
//-----------------------------------------------
    SU3matrix tmp1, tmp2, tmp3;
    int sitePlusMu = neighborPlus[site * 4 + mu];

    su3_zero(staple);

    for (int nu = 0; nu < 4; nu++) {
        if (nu == mu) continue;

        int sitePlusNu     = neighborPlus[site * 4 + nu];
        int siteMinusNu    = neighborMinus[site * 4 + nu];
        int sitePlusMuMinusNu = neighborMinus[sitePlusMu * 4 + nu];

        //-----------------------------------------------
        // Positive staple:
        //   U(x+mu,nu) * U^dag(x+nu,mu) * U^dag(x,nu)
        //-----------------------------------------------
        su3_dagger(gaugeField[sitePlusNu * 4 + mu], tmp1);    // U^dag(x+nu,mu)
        su3_multiply(gaugeField[sitePlusMu * 4 + nu], tmp1, tmp2);  // U(x+mu,nu) * U^dag(x+nu,mu)

        su3_dagger(gaugeField[site * 4 + nu], tmp1);          // U^dag(x,nu)
        su3_multiply(tmp2, tmp1, tmp3);                         // full positive staple

        su3_accumulate(tmp3, staple);

        //-----------------------------------------------
        // Negative staple:
        //   U^dag(x+mu-nu,nu) * U^dag(x-nu,mu) * U(x-nu,nu)
        //-----------------------------------------------
        su3_dagger(gaugeField[sitePlusMuMinusNu * 4 + nu], tmp1);  // U^dag(x+mu-nu,nu)
        su3_dagger(gaugeField[siteMinusNu * 4 + mu], tmp2);        // U^dag(x-nu,mu)
        su3_multiply(tmp1, tmp2, tmp3);

        su3_multiply(tmp3, gaugeField[siteMinusNu * 4 + nu], tmp1); // * U(x-nu,nu)

        su3_accumulate(tmp1, staple);
    }
}
