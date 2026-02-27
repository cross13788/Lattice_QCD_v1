//-----------------------------------------------
// Compute the full quark propagator S(x, x0)
//
// The propagator has 12 x 12 components:
//   S(x)_{s,c; s0,c0} for s,s0=0..3, c,c0=0..2
//
// We solve D_W * S_col = source for each of the
// 12 spin-color combinations at the source.
//
// Storage: propagator[12 * site + (s0 * NC + c0)]
//   gives a ColorSpinor at site x for source
//   spin s0 and source color c0.
//
// The source is placed at the origin.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "colorspinor_module.hpp"
#include "interfaces_module.hpp"
#include <cstdio>
#include <cstdlib>

void compute_propagator(const SU3matrix* gaugeField,
                        ColorSpinor* propagator,
                        int vol)
{
//-----------------------------------------------
//   Allocate source and solution vectors
//-----------------------------------------------
    ColorSpinor* source   = new ColorSpinor[vol];
    ColorSpinor* solution = new ColorSpinor[vol];

//-----------------------------------------------
//   Loop over 12 spin-color source combinations
//-----------------------------------------------
    printf("  Computing quark propagator (12 inversions)...\n");

    int totalIter = 0;

    for (int s0 = 0; s0 < ND; s0++) {
        for (int c0 = 0; c0 < NC; c0++) {
            int srcIndex = s0 * NC + c0;

            // Create point source at origin
            point_source(source, vol, s0, c0, 0, 0, 0, 0);

            // Solve D_W * solution = source
            int nIter;
            if (solverType == "bicgstab") {
                nIter = bicgstab_solver(gaugeField, source, solution, vol);
            } else {
                nIter = cg_solver(gaugeField, source, solution, vol);
            }

            totalIter += nIter;

            printf("    spin=%d color=%d: %d iterations\n", s0, c0, nIter);

            // Store solution in propagator array
            #pragma omp parallel for
            for (int site = 0; site < vol; site++) {
                cs_copy(solution[site], propagator[12 * site + srcIndex]);
            }
        }
    }

    printf("  Propagator complete: %d total iterations\n", totalIter);

//-----------------------------------------------
//   Cleanup
//-----------------------------------------------
    delete[] source;
    delete[] solution;
}
