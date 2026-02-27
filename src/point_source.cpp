//-----------------------------------------------
// Create a point source for quark propagator
// computation.
//
// Sets a delta function in position, spin, and
// color: source(x)_{s,c} = delta(x, x0) *
//                           delta(s, s0) *
//                           delta(c, c0)
//
// The source is placed at the origin (0,0,0,0)
// unless a different source site is specified.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "lattice_module.hpp"
#include "colorspinor_module.hpp"
#include "interfaces_module.hpp"

void point_source(ColorSpinor* source, int vol,
                  int spin0, int color0,
                  int srcX, int srcY, int srcZ, int srcT)
{
//-----------------------------------------------
//   Zero the entire field
//-----------------------------------------------
    field_zero(source, vol);

//-----------------------------------------------
//   Set delta function at source location
//-----------------------------------------------
    int srcSite = site_index(srcX, srcY, srcZ, srcT);
    source[srcSite].v[spin0][color0] = complex_t(1.0, 0.0);
}
