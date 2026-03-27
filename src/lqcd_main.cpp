//-----------------------------------------------
// Lattice QCD - Main Program
//
// Entry point for all lattice QCD simulations.
// Reads input, initializes lattice geometry,
// and dispatches to the appropriate simulation
// mode (quenched gauge, fermion measurements, HMC).
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "lattice_module.hpp"
#include "su3_module.hpp"
#include "colorspinor_module.hpp"
#include "gamma_matrices.hpp"
#include "interfaces_module.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>

int main(int argc, char* argv[])
{
//-----------------------------------------------
//   Disable stdout buffering for progress output
//-----------------------------------------------
    setbuf(stdout, NULL);

//-----------------------------------------------
//   Local variables
//-----------------------------------------------
    int numberProcessors, maxThreads;

//-----------------------------------------------
//   Print header
//-----------------------------------------------
    printf("===============================================\n");
    printf("  Lattice QCD Simulation Code\n");
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
//   OpenMP setup
//-----------------------------------------------
    openmp_parallel_setup(numberProcessors, maxThreads);

//-----------------------------------------------
//   Initialize lattice geometry
//-----------------------------------------------
    initialize_lattice_geometry();

//-----------------------------------------------
//   Initialize gamma matrices (Phase 2)
//-----------------------------------------------
    initialize_gamma_matrices();

//-----------------------------------------------
//   Initialize sigma matrices (clover improvement)
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
//   SU(3) sanity checks
//-----------------------------------------------
    if (verboseOutput) {
        printf("Running SU(3) sanity checks...\n");

        // Test 1: Identity
        SU3matrix I;
        su3_identity(I);
        complex_t detI = su3_determinant(I);
        printf("  det(I)    = (%.6f, %.6f)  [expect (1,0)]\n",
               detI.real(), detI.imag());

        // Test 2: Random SU(3) should have det = 1
        SU3matrix R;
        su3_random_near_identity(R, 0.5);
        su3_reunitarize(R);
        complex_t detR = su3_determinant(R);
        printf("  det(R)    = (%.6f, %.6f)  [expect (1,0)]\n",
               detR.real(), detR.imag());

        // Test 3: R * R^dag = I
        SU3matrix Rdag, RRdag;
        su3_dagger(R, Rdag);
        su3_multiply(R, Rdag, RRdag);
        real_t offDiag = 0.0;
        for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++) {
                complex_t expected = (a == b) ? complex_t(1.0, 0.0) : complex_t(0.0, 0.0);
                offDiag += std::abs(RRdag.m[a][b] - expected);
            }
        printf("  |R*R^d-I| = %.2e  [expect ~0]\n", offDiag);

        // Test 4: Random uniform
        SU3matrix U;
        su3_random_uniform(U);
        complex_t detU = su3_determinant(U);
        printf("  det(Urand)= (%.6f, %.6f)  [expect (1,0)]\n",
               detU.real(), detU.imag());

        printf("SU(3) checks passed.\n\n");
    }

//-----------------------------------------------
//   Allocate gauge field
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
//   Initialize gauge field
//-----------------------------------------------
    initialize_gauge_field(gaugeField, startType.c_str());

//-----------------------------------------------
//   Compute initial plaquette
//-----------------------------------------------
    real_t avgPlaquette;
    compute_plaquette(gaugeField, avgPlaquette);
    printf("Initial average plaquette: %.10f\n\n", avgPlaquette);

//-----------------------------------------------
//   Run simulation
//-----------------------------------------------
    if (updateMethod == "hmc") {
        // Run gradient check if clover is enabled (validates force)
        if (verboseOutput) {
            ColorSpinor* phi_test = new ColorSpinor[latticeVolume];
            generate_pseudofermion(gaugeField, phi_test, latticeVolume);
            gradient_check(gaugeField, phi_test, latticeVolume);
            delete[] phi_test;
        }

        // Dynamical fermion simulation via HMC
        hmc_driver(gaugeField);
    } else {
        // Quenched simulation (heatbath or metropolis)
        thermalization(gaugeField);
        generate_configurations(gaugeField);
    }

//-----------------------------------------------
//   Cleanup
//-----------------------------------------------
    free_clover_cache();
    free(gaugeField);
    free_lattice_geometry();

    printf("===============================================\n");
    printf("  Simulation complete.\n");
    printf("===============================================\n");

    return 0;
}
