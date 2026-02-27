//-----------------------------------------------
// Thermalization: run nTherm combined update
// sweeps to equilibrate the gauge field from
// the initial configuration.
//
// Prints the average plaquette every 10 sweeps
// so the user can monitor thermalization.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "interfaces_module.hpp"
#include <cstdio>

void thermalization(SU3matrix* gaugeField)
{
    printf("===============================================\n");
    printf("  Thermalization: %d sweeps\n", nTherm);
    printf("===============================================\n\n");

    for (int sweep = 1; sweep <= nTherm; sweep++) {
        gauge_update_combined(gaugeField);

        if (sweep % 10 == 0 || sweep == 1) {
            real_t avgPlaquette;
            compute_plaquette(gaugeField, avgPlaquette);
            printf("  Therm sweep %5d / %5d  <P> = %.10f\n",
                   sweep, nTherm, avgPlaquette);
        }
    }

    printf("\nThermalization complete.\n\n");
}
