//-----------------------------------------------
// Write measured observables to ASCII files
//-----------------------------------------------
#include "constants_module.hpp"
#include "interfaces_module.hpp"
#include <cstdio>

void write_observables(int configNum, real_t plaquette, real_t polyakov,
                       const real_t* wilsonLoops, int rMax, int tMax,
                       const char* directory)
{
//-----------------------------------------------
//   Append plaquette to data file
//-----------------------------------------------
    char fileName[512];

    snprintf(fileName, sizeof(fileName), "%s/observables.dat", directory);
    FILE* fp = fopen(fileName, "a");
    if (fp) {
        fprintf(fp, "%6d  %.12f  %.12f\n", configNum, plaquette, polyakov);
        fclose(fp);
    }
}
