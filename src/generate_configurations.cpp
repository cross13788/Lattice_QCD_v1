//-----------------------------------------------
// Main measurement loop
//
// After thermalization, generate nConfig gauge
// configurations separated by nSweepsBetween
// combined update sweeps. Measure observables
// on each configuration.
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
#include <cstring>
#include <string>

void generate_configurations(SU3matrix* gaugeField)
{
    printf("===============================================\n");
    printf("  Generating %d configurations\n", nConfig);
    printf("  (%d sweeps between measurements)\n", nSweepsBetween);
    printf("===============================================\n\n");

//-----------------------------------------------
//   Open output files
//-----------------------------------------------
    char plaqFileName[512];
    snprintf(plaqFileName, sizeof(plaqFileName), "%s/plaquette.dat",
             outputDirectory.c_str());
    FILE* plaqFile = fopen(plaqFileName, "w");
    if (!plaqFile) {
        printf("Error: Cannot open %s for writing\n", plaqFileName);
        exit(1);
    }
    fprintf(plaqFile, "# config  avg_plaquette  action_density\n");

    FILE* polyFile = nullptr;
    if (measurePolyakov) {
        char polyFileName[512];
        snprintf(polyFileName, sizeof(polyFileName), "%s/polyakov.dat",
                 outputDirectory.c_str());
        polyFile = fopen(polyFileName, "w");
        fprintf(polyFile, "# config  polyakov_loop\n");
    }

    FILE* wloopFile = nullptr;
    if (measureWilsonLoops) {
        char wloopFileName[512];
        snprintf(wloopFileName, sizeof(wloopFileName), "%s/wilson_loops.dat",
                 outputDirectory.c_str());
        wloopFile = fopen(wloopFileName, "w");
        fprintf(wloopFile, "# config  R  T  W(R,T)\n");
    }

//-----------------------------------------------
//   Allocate Wilson loop storage
//-----------------------------------------------
    real_t* wilsonLoops = nullptr;
    if (measureWilsonLoops) {
        wilsonLoops = (real_t*)calloc(wilsonLoopRmax * wilsonLoopTmax, sizeof(real_t));
    }

//-----------------------------------------------
//   Allocate action density storage
//-----------------------------------------------
    real_t* actionDensityField = nullptr;
    if (measureActionDensity) {
        actionDensityField = (real_t*)calloc(latticeVolume, sizeof(real_t));
        printf("Action density measurement enabled");
        if (nCoolingSteps > 0)
            printf(" with %d cooling steps (alpha=%.2f)", nCoolingSteps, coolingAlpha);
        printf("\n\n");
    }

//-----------------------------------------------
//   Configuration generation loop
//-----------------------------------------------
    for (int config = 1; config <= nConfig; config++) {

        // Decorrelation sweeps
        for (int sweep = 0; sweep < nSweepsBetween; sweep++) {
            gauge_update_combined(gaugeField);
        }

        // Measure plaquette
        real_t avgPlaquette;
        compute_plaquette(gaugeField, avgPlaquette);
        real_t actionDensity = 6.0 * beta * (1.0 - avgPlaquette);
        fprintf(plaqFile, "%6d  %.12f  %.12f\n", config, avgPlaquette, actionDensity);

        // Measure Polyakov loop
        real_t polyakovAvg = 0.0;
        if (measurePolyakov) {
            polyakov_loop(gaugeField, polyakovAvg);
            fprintf(polyFile, "%6d  %.12f\n", config, polyakovAvg);
        }

        // Measure Wilson loops
        if (measureWilsonLoops) {
            wilson_loop(gaugeField, wilsonLoops, wilsonLoopRmax, wilsonLoopTmax);
            for (int r = 0; r < wilsonLoopRmax; r++) {
                for (int t = 0; t < wilsonLoopTmax; t++) {
                    fprintf(wloopFile, "%6d  %3d  %3d  %.12e\n",
                            config, r + 1, t + 1,
                            wilsonLoops[r * wilsonLoopTmax + t]);
                }
            }
        }

        // Wilson flow measurement
        if (measureWilsonFlow) {
            wilson_flow_driver(gaugeField, latticeVolume);

            // Reversibility test on first configuration only
            if (config == 1) {
                real_t revDev = wilson_flow_reversibility_test(gaugeField, latticeVolume);
                printf("  Wilson flow reversibility deviation: %.6e\n", revDev);
            }
        }

        // Measure per-site action density (for visualization)
        if (measureActionDensity) {
            if (nCoolingSteps > 0) {
                // Make a copy for cooling (don't modify the original field)
                int nLinks = latticeVolume * 4;
                SU3matrix* coolField = new SU3matrix[nLinks];
                memcpy(coolField, gaugeField, nLinks * sizeof(SU3matrix));
                cooling_sweep(coolField, actionDensityField, config,
                              nCoolingSteps, coolingAlpha,
                              outputDirectory.c_str());
                delete[] coolField;
            } else {
                // Just write uncooled action density
                compute_action_density(gaugeField, actionDensityField);
                write_action_density(actionDensityField, config, -1,
                                     outputDirectory.c_str());
            }
        }

        // Measure meson correlators (Phase 2)
        if (measureCorrelators) {
            // Allocate propagator: 12 ColorSpinors per site
            ColorSpinor* propagator = new ColorSpinor[12 * latticeVolume];

            compute_propagator(gaugeField, propagator, latticeVolume);

            // Measure and write correlators for each channel
            real_t* corr = new real_t[nT];
            real_t* meff = new real_t[nT];

            // Channels: 0=pion, 1-3=rho, 4=scalar, 5-7=a1
            int nChannels = 8;
            for (int ch = 0; ch < nChannels; ch++) {
                meson_correlator(propagator, ch, corr, latticeVolume);
                effective_mass(corr, meff, nT);
                write_correlators(config, ch, corr, meff, outputDirectory.c_str());

                // Print pion correlator summary
                if (ch == 0) {
                    printf("    Pion: C(0)=%.4e  C(T/2)=%.4e  m_eff(1)=%.6f\n",
                           corr[0], corr[nT/2], meff[1]);
                }
            }

            delete[] corr;
            delete[] meff;
            delete[] propagator;
        }

        // Save gauge configuration
        if (saveBinaryConfigs && (config % configSaveFrequency == 0)) {
            write_gauge_config(gaugeField, config, outputDirectory.c_str());
        }

        // Progress output
        printf("  Config %4d / %4d  <P> = %.10f",
               config, nConfig, avgPlaquette);
        if (measurePolyakov)
            printf("  <L> = %.8f", polyakovAvg);
        printf("\n");

        // Flush output files periodically
        if (config % 10 == 0) {
            fflush(plaqFile);
            if (polyFile) fflush(polyFile);
            if (wloopFile) fflush(wloopFile);
        }
    }

//-----------------------------------------------
//   Static potential extraction
//-----------------------------------------------
    if (measureWilsonLoops) {
        // Note: for proper potential extraction, should average
        // Wilson loops over configurations and use jackknife.
        // This is a placeholder for the single-config case.
        char potFileName[512];
        snprintf(potFileName, sizeof(potFileName), "%s/static_potential.dat",
                 outputDirectory.c_str());
        FILE* potFile = fopen(potFileName, "w");
        if (potFile) {
            fprintf(potFile, "# R  V(R)  (from last config only - use analysis scripts for proper average)\n");
            // V(R) ~ -log(W(R,T)/W(R,T-1)) for large T
            for (int r = 0; r < wilsonLoopRmax; r++) {
                int t = wilsonLoopTmax / 2;  // Use T/2 for extraction
                if (t >= 2) {
                    real_t wt   = wilsonLoops[r * wilsonLoopTmax + t - 1];
                    real_t wt1  = wilsonLoops[r * wilsonLoopTmax + t];
                    if (wt > SMALL_VALUE && wt1 > SMALL_VALUE) {
                        real_t vr = -std::log(wt1 / wt);
                        fprintf(potFile, "%3d  %.10e\n", r + 1, vr);
                    }
                }
            }
            fclose(potFile);
        }
    }

//-----------------------------------------------
//   Cleanup
//-----------------------------------------------
    fclose(plaqFile);
    if (polyFile) fclose(polyFile);
    if (wloopFile) fclose(wloopFile);
    if (wilsonLoops) free(wilsonLoops);
    if (actionDensityField) free(actionDensityField);

    printf("\nConfiguration generation complete.\n");
    printf("Output saved to: %s/\n\n", outputDirectory.c_str());
}
