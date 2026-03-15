//-----------------------------------------------
// HMC driver
//
// Runs the full HMC algorithm:
//   1. Thermalization trajectories (no measurements)
//   2. Production trajectories with measurements
//
// Measurements are the same as the quenched case:
// plaquette, Polyakov loop, Wilson loops, and
// optionally meson correlators.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "lattice_module.hpp"
#include "colorspinor_module.hpp"
#include "gamma_matrices.hpp"
#include "interfaces_module.hpp"
#include <cstdio>
#include <cstdlib>

void hmc_driver(SU3matrix* gaugeField)
{
//-----------------------------------------------
//   Thermalization
//-----------------------------------------------
    printf("===============================================\n");
    printf("  HMC Thermalization: %d trajectories\n", nTherm);
    printf("  MD steps: %d, step size: %.4f\n", nMDsteps, mdStepSize);
    printf("  Kappa: %.6f\n", kappa);
    printf("===============================================\n\n");

    int acceptedTherm = 0;
    for (int traj = 1; traj <= nTherm; traj++) {
        real_t deltaH;
        int accepted = hmc_trajectory(gaugeField, deltaH);
        acceptedTherm += accepted;

        if (traj % 10 == 0 || traj == 1) {
            real_t avgPlaquette;
            compute_plaquette(gaugeField, avgPlaquette);
            real_t acceptRate = (real_t)acceptedTherm / traj * 100.0;
            printf("  HMC therm %4d / %4d  <P>=%.6f  dH=%.4e  %s  (%.0f%% acc)\n",
                   traj, nTherm, avgPlaquette, deltaH,
                   accepted ? "ACC" : "REJ", acceptRate);
        }
    }

    if (nTherm > 0) {
        printf("\nThermalization complete: %.1f%% acceptance rate\n\n",
               (real_t)acceptedTherm / nTherm * 100.0);
    }

//-----------------------------------------------
//   Open output files
//-----------------------------------------------
    char plaqFileName[512];
    snprintf(plaqFileName, sizeof(plaqFileName), "%s/plaquette.dat",
             outputDirectory.c_str());
    FILE* plaqFile = fopen(plaqFileName, "w");
    fprintf(plaqFile, "# traj  avg_plaquette  action_density  deltaH  accepted\n");

    FILE* polyFile = nullptr;
    if (measurePolyakov) {
        char polyFileName[512];
        snprintf(polyFileName, sizeof(polyFileName), "%s/polyakov.dat",
                 outputDirectory.c_str());
        polyFile = fopen(polyFileName, "w");
        fprintf(polyFile, "# traj  polyakov_loop\n");
    }

//-----------------------------------------------
//   Production trajectories
//-----------------------------------------------
    printf("===============================================\n");
    printf("  HMC Production: %d trajectories\n", nHMCtrajectories);
    printf("===============================================\n\n");

    int acceptedProd = 0;

    for (int traj = 1; traj <= nHMCtrajectories; traj++) {
        real_t deltaH;
        int accepted = hmc_trajectory(gaugeField, deltaH);
        acceptedProd += accepted;

        // Measure plaquette
        real_t avgPlaquette;
        compute_plaquette(gaugeField, avgPlaquette);
        real_t actionDensity = 6.0 * beta * (1.0 - avgPlaquette);
        fprintf(plaqFile, "%6d  %.12f  %.12f  %16.8e  %d\n",
                traj, avgPlaquette, actionDensity, deltaH, accepted);

        // Polyakov loop
        real_t polyakovAvg = 0.0;
        if (measurePolyakov) {
            polyakov_loop(gaugeField, polyakovAvg);
            fprintf(polyFile, "%6d  %.12f\n", traj, polyakovAvg);
        }

        // Wilson flow measurement
        if (measureWilsonFlow) {
            wilson_flow_driver(gaugeField, latticeVolume);
        }

        // Meson correlators (if enabled)
        if (measureCorrelators) {
            ColorSpinor* propagator = new ColorSpinor[12 * latticeVolume];
            compute_propagator(gaugeField, propagator, latticeVolume);

            real_t* corr = new real_t[nT];
            real_t* meff = new real_t[nT];

            for (int ch = 0; ch < 8; ch++) {
                meson_correlator(propagator, ch, corr, latticeVolume);
                effective_mass(corr, meff, nT);
                write_correlators(traj, ch, corr, meff, outputDirectory.c_str());
            }

            if (traj == 1 || traj % 10 == 0) {
                // Quick pion summary
                meson_correlator(propagator, 0, corr, latticeVolume);
                effective_mass(corr, meff, nT);
                printf("    Pion: C(0)=%.4e m_eff(1)=%.6f\n", corr[0], meff[1]);
            }

            delete[] corr;
            delete[] meff;
            delete[] propagator;
        }

        // Save gauge configuration
        if (saveBinaryConfigs && (traj % configSaveFrequency == 0)) {
            write_gauge_config(gaugeField, traj, outputDirectory.c_str());
        }

        // Progress output
        real_t acceptRate = (real_t)acceptedProd / traj * 100.0;
        printf("  HMC traj %4d / %4d  <P>=%.6f  dH=%+.4e  %s  (%.0f%% acc)",
               traj, nHMCtrajectories, avgPlaquette, deltaH,
               accepted ? "ACC" : "REJ", acceptRate);
        if (measurePolyakov)
            printf("  <L>=%.6f", polyakovAvg);
        printf("\n");

        // Flush periodically
        if (traj % 10 == 0) {
            fflush(plaqFile);
            if (polyFile) fflush(polyFile);
        }
    }

//-----------------------------------------------
//   Summary
//-----------------------------------------------
    printf("\nHMC production complete: %.1f%% acceptance rate\n",
           (real_t)acceptedProd / nHMCtrajectories * 100.0);
    printf("Output saved to: %s/\n\n", outputDirectory.c_str());

//-----------------------------------------------
//   Cleanup
//-----------------------------------------------
    fclose(plaqFile);
    if (polyFile) fclose(polyFile);
}
