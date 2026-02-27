//-----------------------------------------------
// Measure and record the average plaquette and
// action density on a given configuration.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "interfaces_module.hpp"
#include <cstdio>

void measure_plaquette(const SU3matrix* gaugeField, int configNum, FILE* outFile)
{
    real_t avgPlaquette;
    compute_plaquette(gaugeField, avgPlaquette);

    real_t actionDensity = 6.0 * beta * (1.0 - avgPlaquette);

    if (outFile) {
        fprintf(outFile, "%6d  %.12f  %.12f\n", configNum, avgPlaquette, actionDensity);
    }
}
