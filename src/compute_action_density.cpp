//-----------------------------------------------
// Per-site action density for 3D visualization
//
// S(x) = sum_{mu>nu} (1 - (1/NC) Re Tr P_{mu,nu}(x))
//
// This gives the local gauge action density at each
// site, ranging from 0 (ordered) to 6 (disordered).
// Used for CSSM Adelaide-style vacuum visualizations.
//
// Also implements APE smearing (cooling) to reveal
// instanton structure by smoothing UV fluctuations.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "lattice_module.hpp"
#include "interfaces_module.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

//-----------------------------------------------
// Compute per-site action density field
//
// actionDensity[site] = sum_{mu>nu} (1 - ReTr P / NC)
// for all lattice sites. Values range [0, 6].
//-----------------------------------------------
void compute_action_density(const SU3matrix* gaugeField, real_t* actionDensity)
{
    #pragma omp parallel for
    for (int site = 0; site < latticeVolume; site++) {
        real_t localAction = 0.0;

        for (int mu = 0; mu < 4; mu++) {
            for (int nu = mu + 1; nu < 4; nu++) {
                int sitePlusMu = neighborPlus[site * 4 + mu];
                int sitePlusNu = neighborPlus[site * 4 + nu];

                SU3matrix tmp1, tmp2, tmp3, plaq;

                // tmp1 = U(x,mu) * U(x+mu,nu)
                su3_multiply(gaugeField[site * 4 + mu],
                             gaugeField[sitePlusMu * 4 + nu],
                             tmp1);

                // tmp2 = U^dag(x+nu,mu)
                su3_dagger(gaugeField[sitePlusNu * 4 + mu], tmp2);

                // tmp3 = tmp1 * tmp2
                su3_multiply(tmp1, tmp2, tmp3);

                // plaq = tmp3 * U^dag(x,nu)
                SU3matrix UdagNu;
                su3_dagger(gaugeField[site * 4 + nu], UdagNu);
                su3_multiply(tmp3, UdagNu, plaq);

                localAction += (1.0 - su3_retrace(plaq) / NC);
            }
        }

        actionDensity[site] = localAction;
    }
}

//-----------------------------------------------
// Write action density to binary file
//
// Format: 4 ints (nX, nY, nZ, nT) followed by
//         latticeVolume doubles in site-order
//         idx = x + nX*(y + nY*(z + nZ*t))
//-----------------------------------------------
void write_action_density(const real_t* actionDensity, int configNum,
                          int coolStep, const char* directory)
{
    char filename[512];
    if (coolStep < 0) {
        snprintf(filename, sizeof(filename),
                 "%s/action_density_cfg%04d.bin", directory, configNum);
    } else {
        snprintf(filename, sizeof(filename),
                 "%s/action_density_cfg%04d_cool%03d.bin",
                 directory, configNum, coolStep);
    }

    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        printf("Error: Cannot open %s for writing\n", filename);
        return;
    }

    // Write header: lattice dimensions
    fwrite(&nX, sizeof(int), 1, fp);
    fwrite(&nY, sizeof(int), 1, fp);
    fwrite(&nZ, sizeof(int), 1, fp);
    fwrite(&nT, sizeof(int), 1, fp);

    // Write action density field
    fwrite(actionDensity, sizeof(real_t), latticeVolume, fp);

    fclose(fp);
}

//-----------------------------------------------
// Cooling step: minimize the local Wilson action
//
// Uses the Cabibbo-Marinari heat bath at very large
// beta (= low temperature = deterministic cooling).
// At large beta, the SU(2) heat bath deterministically
// selects the maximum of Re Tr(U * Staple), which
// minimizes the local action.
//
// Uses checkerboard decomposition like the regular
// heat bath sweep, allowing parallel updates of
// sites with the same parity.
//-----------------------------------------------
void ape_smearing_step(SU3matrix* gaugeField, real_t /*alpha*/)
{
    // Use beta_cool >> physical beta for deterministic cooling
    const real_t beta_cool = 1000.0;

    for (int parity = 0; parity < 2; parity++) {
        for (int site = 0; site < latticeVolume; site++) {
            if (site_parity(site) != parity) continue;

            for (int mu = 0; mu < 4; mu++) {
                SU3matrix staple;
                compute_staple(gaugeField, site, mu, staple);
                su3_heat_bath_update(gaugeField[site * 4 + mu],
                                     staple, beta_cool);
            }
        }
    }
}

//-----------------------------------------------
// Full cooling sequence: apply nCool APE smearing
// steps, writing action density at each step
//-----------------------------------------------
void cooling_sweep(SU3matrix* gaugeField, real_t* actionDensity,
                   int configNum, int nCool, real_t alpha,
                   const char* directory)
{
    printf("  Cooling config %d: %d steps (alpha=%.2f)\n",
           configNum, nCool, alpha);

    // Write uncooled (step 0) action density
    compute_action_density(gaugeField, actionDensity);
    write_action_density(actionDensity, configNum, 0, directory);

    real_t avgPlaq;
    compute_plaquette(gaugeField, avgPlaq);
    printf("    Cool step %3d:  <P> = %.8f  <S> = %.4f\n",
           0, avgPlaq, 6.0 * (1.0 - avgPlaq));

    for (int step = 1; step <= nCool; step++) {
        ape_smearing_step(gaugeField, alpha);

        compute_action_density(gaugeField, actionDensity);
        write_action_density(actionDensity, configNum, step, directory);

        if (step % 5 == 0 || step == nCool) {
            compute_plaquette(gaugeField, avgPlaq);
            printf("    Cool step %3d:  <P> = %.8f  <S> = %.4f\n",
                   step, avgPlaq, 6.0 * (1.0 - avgPlaq));
        }
    }
}
