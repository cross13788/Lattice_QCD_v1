//-----------------------------------------------
// Combined gauge update: 1 heat bath sweep
// followed by nOverrelax overrelaxation sweeps.
//
// This is the standard recipe for quenched QCD
// Monte Carlo. The heat bath thermalizes, and
// overrelaxation reduces autocorrelation.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "interfaces_module.hpp"

void gauge_update_combined(SU3matrix* gaugeField)
{
//-----------------------------------------------
//   One heat bath sweep
//-----------------------------------------------
    if (updateMethod == "heatbath") {
        heat_bath_sweep(gaugeField);
    } else {
        real_t acceptRate;
        metropolis_sweep(gaugeField, acceptRate);
    }

//-----------------------------------------------
//   nOverrelax overrelaxation sweeps
//-----------------------------------------------
    for (int i = 0; i < nOverrelax; i++) {
        overrelaxation_sweep(gaugeField);
    }
}
