//-----------------------------------------------
// GPU gauge update: combined, thermalization,
// and configuration generation
//
// These are orchestrator functions that call
// the GPU heat bath and overrelaxation kernels.
//-----------------------------------------------
#include "gpu_state.cuh"
#include <cstdio>

// Forward declarations
void gpu_heat_bath_sweep(gpu_real_t beta, int vol);
void gpu_overrelaxation_sweep(int vol);
gpu_real_t gpu_compute_plaquette(int vol);

void gpu_gauge_update_combined(gpu_real_t beta, int nOverrelax, int vol)
{
    gpu_heat_bath_sweep(beta, vol);
    for (int i = 0; i < nOverrelax; i++)
        gpu_overrelaxation_sweep(vol);
}

void gpu_thermalization(gpu_real_t beta, int nTherm, int nOverrelax,
                        bool verbose, int vol)
{
    printf("===============================================\n");
    printf("  GPU Thermalization: %d sweeps\n", nTherm);
    printf("===============================================\n\n");

    for (int sweep = 1; sweep <= nTherm; sweep++) {
        gpu_gauge_update_combined(beta, nOverrelax, vol);

        if (sweep % 10 == 0 || sweep == 1) {
            gpu_real_t avgPlaquette = gpu_compute_plaquette(vol);
            printf("  Therm sweep %5d / %5d  <P> = %.10f\n",
                   sweep, nTherm, avgPlaquette);
        }
    }

    printf("\nThermalization complete.\n\n");
}
