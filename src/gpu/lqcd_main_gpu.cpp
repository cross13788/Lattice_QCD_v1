//-----------------------------------------------
// Lattice QCD - GPU Main Program
//
// Entry point for GPU-accelerated simulations.
// Uses the same input system and shared modules
// as the CPU build, but dispatches all compute
// to CUDA kernels.
//
// Compiled with nvc++ -DUSE_GPU
//-----------------------------------------------
#include "../constants_module.hpp"
#include "../input_module.hpp"
#include "../lattice_module.hpp"
#include "../su3_module.hpp"
#include "../colorspinor_module.hpp"
#include "../gamma_matrices.hpp"
#include "../interfaces_module.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>

// GPU function declarations (defined in .cu files, linked via nvcc)
extern "C++" {
    // gpu_state.cuh declarations
    void gpu_init(int vol);
    void gpu_free();
    void gpu_upload_gauge(const void* hostGaugeField, int nLinks);
    void gpu_download_gauge(void* hostGaugeField, int nLinks);
    void gpu_upload_geometry(const int* neighborPlus, const int* neighborMinus,
                             const int* siteParity, int vol);
    void gpu_upload_gamma(const int* gammaIdx, const void* gammaVal);
    void gpu_alloc_solver(int vol);
    void gpu_alloc_hmc(int vol);
    void gpu_alloc_flow(int vol);
    void gpu_alloc_rand(int nStates, int seed);

    // Gauge operations
    double gpu_compute_plaquette(int vol);

    // Gauge updates (quenched)
    void gpu_thermalization(double beta, int nTherm, int nOverrelax,
                            bool verbose, int vol);
    void gpu_gauge_update_combined(double beta, int nOverrelax, int vol);

    // HMC
    void gpu_hmc_driver(double beta, double kappa, double wilsonR,
                        bool useClover, int cgMaxIter, double cgTolerance,
                        int nTherm, int nMDsteps, double mdStepSize,
                        int nHMCtrajectories, bool measurePolyakov,
                        bool measureWilsonFlow, bool measureCorrelators,
                        bool saveBinaryConfigs, int configSaveFrequency,
                        const char* outputDirectory, int vol);

    // Wilson flow
    void gpu_wilson_flow_driver(double beta, double flowStepSize,
                                double maxFlowTime,
                                const char* outputDirectory, int vol);

    // Polyakov loop
    double gpu_polyakov_loop(int nX, int nY, int nZ, int nT, int spatialVolume);

    // Wilson loops on GPU (avoids D2H transfer)
    void gpu_wilson_loop(double* wilsonLoops, int rMax, int tMax, int vol);
}

int main(int argc, char* argv[])
{
//-----------------------------------------------
//   Disable stdout buffering
//-----------------------------------------------
    setbuf(stdout, NULL);

    int numberProcessors, maxThreads;

//-----------------------------------------------
//   Print header
//-----------------------------------------------
    printf("===============================================\n");
    printf("  Lattice QCD Simulation Code (GPU)\n");
    printf("  SU(%d) Wilson Gauge Action\n", NC);
    printf("===============================================\n\n");

//-----------------------------------------------
//   Read input file
//-----------------------------------------------
    read_input_file();

//-----------------------------------------------
//   Print configuration summary
//-----------------------------------------------
    printf("Configuration summary:\n");
    printf("  Lattice:       %d x %d x %d x %d = %d sites\n",
           nX, nY, nZ, nT, latticeVolume);
    printf("  Beta:          %.4f\n", beta);
    printf("  Update method: %s\n", updateMethod.c_str());
    printf("  Start type:    %s\n", startType.c_str());
    printf("  Thermalization sweeps:  %d\n", nTherm);
    printf("  Configurations:         %d\n", nConfig);
    printf("  Sweeps between configs: %d\n", nSweepsBetween);
    printf("  Overrelaxation sweeps:  %d\n", nOverrelax);
    printf("  Measure Wilson loops:   %s\n", measureWilsonLoops ? "yes" : "no");
    printf("  Measure Polyakov loop:  %s\n", measurePolyakov ? "yes" : "no");
    printf("  Measure correlators:    %s\n", measureCorrelators ? "yes" : "no");
    if (measureCorrelators || updateMethod == "hmc") {
        printf("  Kappa:                  %.6f\n", kappa);
        printf("  Wilson r:               %.1f\n", wilsonR);
        printf("  Solver:                 %s\n", solverType.c_str());
        printf("  CG tolerance:           %.1e\n", cgTolerance);
        if (useClover)
            printf("  Clover c_sw:            %.6f\n", c_sw);
    }
    if (updateMethod == "hmc") {
        printf("  MD steps:               %d\n", nMDsteps);
        printf("  MD step size:           %.4f\n", mdStepSize);
        printf("  HMC trajectories:       %d\n", nHMCtrajectories);
    }
    printf("\n");

//-----------------------------------------------
//   OpenMP setup (for shared CPU code)
//-----------------------------------------------
    openmp_parallel_setup(numberProcessors, maxThreads);

//-----------------------------------------------
//   Initialize lattice geometry
//-----------------------------------------------
    initialize_lattice_geometry();

//-----------------------------------------------
//   Initialize gamma matrices
//-----------------------------------------------
    initialize_gamma_matrices();

//-----------------------------------------------
//   Initialize sigma matrices (clover)
//-----------------------------------------------
    if (useClover) {
        initialize_sigma_matrices();
    }

//-----------------------------------------------
//   Create output directory
//-----------------------------------------------
    char dirName[256];
    snprintf(dirName, sizeof(dirName), "SU%d_%s_beta%.2f_%dx%dx%dx%d",
             NC, updateMethod.c_str(), beta, nX, nY, nZ, nT);
    outputDirectory = dirName;

    struct stat st;
    if (stat(dirName, &st) != 0) {
        mkdir(dirName, 0755);
        printf("Created output directory: %s/\n\n", dirName);
    } else {
        printf("Output directory exists: %s/\n\n", dirName);
    }

//-----------------------------------------------
//   GPU initialization
//-----------------------------------------------
    gpu_init(latticeVolume);

//-----------------------------------------------
//   Compute site parities for GPU
//-----------------------------------------------
    int* parityTable = (int*)malloc(latticeVolume * sizeof(int));
    for (int i = 0; i < latticeVolume; i++)
        parityTable[i] = site_parity(i);

//-----------------------------------------------
//   Upload geometry and gamma matrices to GPU
//-----------------------------------------------
    gpu_upload_geometry(neighborPlus, neighborMinus, parityTable, latticeVolume);
    gpu_upload_gamma((const int*)gammaIdx, (const void*)gammaVal);
    free(parityTable);

//-----------------------------------------------
//   Allocate GPU workspace
//-----------------------------------------------
    if (measureCorrelators || updateMethod == "hmc")
        gpu_alloc_solver(latticeVolume);

    if (updateMethod == "hmc")
        gpu_alloc_hmc(latticeVolume);

    if (measureWilsonFlow)
        gpu_alloc_flow(latticeVolume);

    // cuRAND states (one per site)
    if (updateMethod == "heatbath" || updateMethod == "hmc")
        gpu_alloc_rand(latticeVolume, randomSeed);

    printf("\n");

//-----------------------------------------------
//   Allocate gauge field on host
//-----------------------------------------------
    printf("Allocating gauge field: %d links (%ld MB)\n",
           latticeVolume * 4,
           (long)(latticeVolume * 4 * sizeof(SU3matrix)) / (1024 * 1024));

    SU3matrix* gaugeField = (SU3matrix*)malloc(latticeVolume * 4 * sizeof(SU3matrix));
    if (!gaugeField) {
        printf("Error: Failed to allocate gauge field\n");
        return 1;
    }

//-----------------------------------------------
//   Initialize gauge field (on host)
//-----------------------------------------------
    initialize_gauge_field(gaugeField, startType.c_str());

//-----------------------------------------------
//   Upload gauge field to GPU
//-----------------------------------------------
    gpu_upload_gauge(gaugeField, latticeVolume * 4);

//-----------------------------------------------
//   Compute initial plaquette (on GPU)
//-----------------------------------------------
    real_t avgPlaquette = gpu_compute_plaquette(latticeVolume);
    printf("Initial average plaquette: %.10f\n\n", avgPlaquette);

//-----------------------------------------------
//   Run simulation
//-----------------------------------------------
    if (updateMethod == "hmc") {
        gpu_hmc_driver(beta, kappa, wilsonR, useClover,
                       cgMaxIter, cgTolerance,
                       nTherm, nMDsteps, mdStepSize,
                       nHMCtrajectories, measurePolyakov,
                       measureWilsonFlow, measureCorrelators,
                       saveBinaryConfigs, configSaveFrequency,
                       outputDirectory.c_str(), latticeVolume);
    } else {
        // Quenched simulation on GPU
        gpu_thermalization(beta, nTherm, nOverrelax, verboseOutput, latticeVolume);

        // Configuration generation with measurements
        printf("===============================================\n");
        printf("  Generating %d configurations (GPU)\n", nConfig);
        printf("  (%d sweeps between measurements)\n", nSweepsBetween);
        printf("===============================================\n\n");

        // Open output files
        char plaqFileName[512];
        snprintf(plaqFileName, sizeof(plaqFileName), "%s/plaquette.dat",
                 outputDirectory.c_str());
        FILE* plaqFile = fopen(plaqFileName, "w");
        fprintf(plaqFile, "# config  avg_plaquette  action_density\n");

        FILE* polyFile = nullptr;
        if (measurePolyakov) {
            char polyFileName[512];
            snprintf(polyFileName, sizeof(polyFileName), "%s/polyakov.dat",
                     outputDirectory.c_str());
            polyFile = fopen(polyFileName, "w");
            fprintf(polyFile, "# config  polyakov_loop\n");
        }

        for (int config = 1; config <= nConfig; config++) {
            // Decorrelation sweeps
            for (int sweep = 0; sweep < nSweepsBetween; sweep++)
                gpu_gauge_update_combined(beta, nOverrelax, latticeVolume);

            // Measurements
            real_t plaq = gpu_compute_plaquette(latticeVolume);
            real_t actionDensity = 6.0 * beta * (1.0 - plaq);
            fprintf(plaqFile, "%6d  %.12f  %.12f\n", config, plaq, actionDensity);

            real_t polyakovAvg = 0.0;
            if (measurePolyakov) {
                polyakovAvg = gpu_polyakov_loop(nX, nY, nZ, nT, spatialVolume);
                fprintf(polyFile, "%6d  %.12f\n", config, polyakovAvg);
            }

            if (measureWilsonFlow)
                gpu_wilson_flow_driver(beta, flowStepSize, maxFlowTime,
                                       outputDirectory.c_str(), latticeVolume);

            // Wilson loops — run entirely on GPU (no D2H transfer needed)
            if (measureWilsonLoops) {
                real_t* wilsonLoops = (real_t*)calloc(wilsonLoopRmax * wilsonLoopTmax, sizeof(real_t));
                gpu_wilson_loop(wilsonLoops, wilsonLoopRmax, wilsonLoopTmax, latticeVolume);
                free(wilsonLoops);
            }

            // Only download gauge field to host when strictly necessary:
            // correlators (CPU propagator solve), action density, or config save
            if (measureCorrelators || measureActionDensity ||
                (saveBinaryConfigs && config % configSaveFrequency == 0)) {
                gpu_download_gauge(gaugeField, latticeVolume * 4);
            }

            if (measureCorrelators) {
                ColorSpinor* propagator = new ColorSpinor[12 * latticeVolume];
                compute_propagator(gaugeField, propagator, latticeVolume);

                real_t* corr = new real_t[nT];
                real_t* meff = new real_t[nT];
                for (int ch = 0; ch < 8; ch++) {
                    meson_correlator(propagator, ch, corr, latticeVolume);
                    effective_mass(corr, meff, nT);
                    write_correlators(config, ch, corr, meff, outputDirectory.c_str());
                }
                delete[] corr;
                delete[] meff;
                delete[] propagator;
            }

            if (saveBinaryConfigs && config % configSaveFrequency == 0)
                write_gauge_config(gaugeField, config, outputDirectory.c_str());

            // Progress
            printf("  Config %4d / %4d  <P> = %.10f", config, nConfig, plaq);
            if (measurePolyakov) printf("  <L> = %.8f", polyakovAvg);
            printf("\n");

            if (config % 10 == 0) {
                fflush(plaqFile);
                if (polyFile) fflush(polyFile);
            }
        }

        fclose(plaqFile);
        if (polyFile) fclose(polyFile);
        printf("\nConfiguration generation complete.\n");
        printf("Output saved to: %s/\n\n", outputDirectory.c_str());
    }

//-----------------------------------------------
//   Cleanup
//-----------------------------------------------
    gpu_free();
    free(gaugeField);
    free_lattice_geometry();

    printf("===============================================\n");
    printf("  Simulation complete.\n");
    printf("===============================================\n");

    return 0;
}
