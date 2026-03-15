//-----------------------------------------------
// Wilson flow (gradient flow) for scale setting
//
// Implements the 3-stage Runge-Kutta integration
// scheme of Lüscher (JHEP 2010):
//   W0 = V;   Z0 = eps * S'(W0)
//   W1 = exp(1/4 Z0) W0;  Z1 = eps * S'(W1)
//   W2 = exp(8/9 Z1 - 17/36 Z0) W1;  Z2 = eps * S'(W2)
//   V' = exp(3/4 Z2 - 8/9 Z1 + 17/36 Z0) W2
//
// Also provides:
//   - Plaquette and clover energy density definitions
//   - Field strength tensor F_{mu,nu} via the 4-leaf clover
//   - Driver that evolves from t=0 to maxFlowTime
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
// Compute the force Z = eps * (beta/Nc) * [U * staple]_TA
// for all links, storing in forceField.
//
// This is the gradient of the Wilson plaquette action
// projected onto the Lie algebra.
//-----------------------------------------------
static void compute_flow_force(const SU3matrix* gaugeField,
                               SU3matrix* forceField,
                               real_t epsilon,
                               int vol)
{
    real_t coeff = epsilon * beta / (real_t)NC;

    #pragma omp parallel for
    for (int site = 0; site < vol; site++) {
        for (int mu = 0; mu < 4; mu++) {
            int idx = site * 4 + mu;

            SU3matrix staple;
            compute_staple(gaugeField, site, mu, staple);

            SU3matrix UV;
            su3_multiply(gaugeField[idx], staple, UV);

            // Project to traceless anti-Hermitian and scale by coeff
            SU3matrix forceTA;
            su3_traceless_antiherm(UV, forceTA);
            su3_scale(complex_t(coeff, 0.0), forceTA, forceField[idx]);
        }
    }
}

//-----------------------------------------------
// Wilson flow: single RK3 step
//
// Lüscher's 3-stage Runge-Kutta scheme.
// The gauge field W is updated in-place.
//-----------------------------------------------
void wilson_flow_step(SU3matrix* gaugeField, real_t epsilon, int vol)
{
    int nLinks = vol * 4;

    SU3matrix* Z0 = new SU3matrix[nLinks];
    SU3matrix* Z1 = new SU3matrix[nLinks];
    SU3matrix* Z2 = new SU3matrix[nLinks];

//-----------------------------------------------
//   Stage 0: Z0 = eps * S'(W0)
//            W1 = exp(1/4 Z0) * W0
//-----------------------------------------------
    compute_flow_force(gaugeField, Z0, epsilon, vol);

    #pragma omp parallel for
    for (int idx = 0; idx < nLinks; idx++) {
        SU3matrix scaled;
        su3_scale(complex_t(0.25, 0.0), Z0[idx], scaled);
        SU3matrix expZ;
        su3_exp(scaled, expZ);
        SU3matrix Wnew;
        su3_multiply(expZ, gaugeField[idx], Wnew);
        su3_copy(Wnew, gaugeField[idx]);
    }

//-----------------------------------------------
//   Stage 1: Z1 = eps * S'(W1)
//            W2 = exp(8/9 Z1 - 17/36 Z0) * W1
//-----------------------------------------------
    compute_flow_force(gaugeField, Z1, epsilon, vol);

    #pragma omp parallel for
    for (int idx = 0; idx < nLinks; idx++) {
        // exponent = 8/9 Z1 - 17/36 Z0
        SU3matrix exponent;
        for (int i = 0; i < NC; i++)
            for (int j = 0; j < NC; j++)
                exponent.m[i][j] = (8.0/9.0) * Z1[idx].m[i][j]
                                 - (17.0/36.0) * Z0[idx].m[i][j];

        SU3matrix expZ;
        su3_exp(exponent, expZ);
        SU3matrix Wnew;
        su3_multiply(expZ, gaugeField[idx], Wnew);
        su3_copy(Wnew, gaugeField[idx]);
    }

//-----------------------------------------------
//   Stage 2: Z2 = eps * S'(W2)
//            V' = exp(3/4 Z2 - 8/9 Z1 + 17/36 Z0) * W2
//-----------------------------------------------
    compute_flow_force(gaugeField, Z2, epsilon, vol);

    #pragma omp parallel for
    for (int idx = 0; idx < nLinks; idx++) {
        // exponent = 3/4 Z2 - 8/9 Z1 + 17/36 Z0
        SU3matrix exponent;
        for (int i = 0; i < NC; i++)
            for (int j = 0; j < NC; j++)
                exponent.m[i][j] = (3.0/4.0) * Z2[idx].m[i][j]
                                 - (8.0/9.0) * Z1[idx].m[i][j]
                                 + (17.0/36.0) * Z0[idx].m[i][j];

        SU3matrix expZ;
        su3_exp(exponent, expZ);
        SU3matrix Wnew;
        su3_multiply(expZ, gaugeField[idx], Wnew);
        su3_copy(Wnew, gaugeField[idx]);
    }

    delete[] Z0;
    delete[] Z1;
    delete[] Z2;
}

//-----------------------------------------------
// Plaquette-based energy density:
//   E_plaq = 2 * sum_{x, mu>nu} (1 - (1/Nc) Re Tr P_{mu,nu}(x))
//
// Returns the energy density per site (normalized by volume).
//-----------------------------------------------
real_t wilson_flow_energy_plaquette(const SU3matrix* gaugeField, int vol)
{
    real_t eSum = 0.0;

    #pragma omp parallel for reduction(+:eSum)
    for (int site = 0; site < vol; site++) {
        for (int mu = 0; mu < 4; mu++) {
            for (int nu = mu + 1; nu < 4; nu++) {
                int sitePlusMu = neighborPlus[site * 4 + mu];
                int sitePlusNu = neighborPlus[site * 4 + nu];

                SU3matrix tmp1, tmp2, tmp3, plaq;

                su3_multiply(gaugeField[site * 4 + mu],
                             gaugeField[sitePlusMu * 4 + nu], tmp1);
                su3_dagger(gaugeField[sitePlusNu * 4 + mu], tmp2);
                su3_multiply(tmp1, tmp2, tmp3);
                SU3matrix UdagNu;
                su3_dagger(gaugeField[site * 4 + nu], UdagNu);
                su3_multiply(tmp3, UdagNu, plaq);

                eSum += 2.0 * (1.0 - su3_retrace(plaq) / (real_t)NC);
            }
        }
    }

    return eSum / (real_t)vol;
}

//-----------------------------------------------
// Field strength tensor F_{mu,nu}(x) from the 4-leaf clover.
//
// Q_{mu,nu}(x) = sum of 4 oriented plaquettes in the (mu,nu) plane:
//   P1: U_mu(x) U_nu(x+mu) U_mu^dag(x+nu) U_nu^dag(x)
//   P2: U_nu(x) U_mu^dag(x-mu+nu) U_nu^dag(x-mu) U_mu(x-mu)
//   P3: U_mu^dag(x-mu) U_nu^dag(x-mu-nu) U_mu(x-mu-nu) U_nu(x-nu)  (wait, wrong)
//
// Standard clover definition:
//   Q_{mu,nu}(x) = P_{mu,nu}(x) + P_{nu,-mu}(x) + P_{-mu,-nu}(x) + P_{-nu,mu}(x)
// where:
//   P_{mu,nu}(x)   = U_mu(x) U_nu(x+mu) U_mu^dag(x+nu) U_nu^dag(x)
//   P_{nu,-mu}(x)  = U_nu(x) U_mu^dag(x-mu+nu) U_nu^dag(x-mu) U_mu(x-mu)
//   P_{-mu,-nu}(x) = U_mu^dag(x-mu) U_nu^dag(x-mu-nu) U_mu(x-mu-nu) U_nu(x-nu)
//   P_{-nu,mu}(x)  = U_nu^dag(x-nu) U_mu(x-nu) U_nu(x-nu+mu) U_mu^dag(x)
//
// F_{mu,nu} = (Q - Q^dag) / 8  (traceless anti-Hermitian)
//
// This function is shared between Wilson flow and the clover term.
//-----------------------------------------------
void compute_field_strength_tensor(const SU3matrix* gaugeField,
                                   int site, int mu, int nu,
                                   SU3matrix& Fmunu)
{
    int sitePlusMu    = neighborPlus[site * 4 + mu];
    int sitePlusNu    = neighborPlus[site * 4 + nu];
    int siteMinusMu   = neighborMinus[site * 4 + mu];
    int siteMinusNu   = neighborMinus[site * 4 + nu];
    int siteMinusMuPlusNu  = neighborPlus[siteMinusMu * 4 + nu];
    int siteMinusMuMinusNu = neighborMinus[siteMinusMu * 4 + nu];
    int siteMinusNuPlusMu  = neighborPlus[siteMinusNu * 4 + mu];

    SU3matrix tmp1, tmp2, tmp3, plaq;
    SU3matrix Qmunu;
    su3_zero(Qmunu);

    //--- Leaf 1: P_{mu,nu}(x) ---
    // U_mu(x) U_nu(x+mu) U_mu^dag(x+nu) U_nu^dag(x)
    su3_multiply(gaugeField[site * 4 + mu],
                 gaugeField[sitePlusMu * 4 + nu], tmp1);
    su3_dagger(gaugeField[sitePlusNu * 4 + mu], tmp2);
    su3_multiply(tmp1, tmp2, tmp3);
    su3_dagger(gaugeField[site * 4 + nu], tmp2);
    su3_multiply(tmp3, tmp2, plaq);
    su3_accumulate(plaq, Qmunu);

    //--- Leaf 2: P_{nu,-mu}(x) ---
    // U_nu(x) U_mu^dag(x-mu+nu) U_nu^dag(x-mu) U_mu(x-mu)
    su3_dagger(gaugeField[siteMinusMuPlusNu * 4 + mu], tmp1);
    su3_multiply(gaugeField[site * 4 + nu], tmp1, tmp2);
    su3_dagger(gaugeField[siteMinusMu * 4 + nu], tmp1);
    su3_multiply(tmp2, tmp1, tmp3);
    su3_multiply(tmp3, gaugeField[siteMinusMu * 4 + mu], plaq);
    su3_accumulate(plaq, Qmunu);

    //--- Leaf 3: P_{-mu,-nu}(x) ---
    // U_mu^dag(x-mu) U_nu^dag(x-mu-nu) U_mu(x-mu-nu) U_nu(x-nu)
    su3_dagger(gaugeField[siteMinusMu * 4 + mu], tmp1);
    su3_dagger(gaugeField[siteMinusMuMinusNu * 4 + nu], tmp2);
    su3_multiply(tmp1, tmp2, tmp3);
    su3_multiply(tmp3, gaugeField[siteMinusMuMinusNu * 4 + mu], tmp1);
    su3_multiply(tmp1, gaugeField[siteMinusNu * 4 + nu], plaq);
    su3_accumulate(plaq, Qmunu);

    //--- Leaf 4: P_{-nu,mu}(x) ---
    // U_nu^dag(x-nu) U_mu(x-nu) U_nu(x-nu+mu) U_mu^dag(x)
    su3_dagger(gaugeField[siteMinusNu * 4 + nu], tmp1);
    su3_multiply(tmp1, gaugeField[siteMinusNu * 4 + mu], tmp2);
    su3_multiply(tmp2, gaugeField[siteMinusNuPlusMu * 4 + nu], tmp3);
    su3_dagger(gaugeField[site * 4 + mu], tmp1);
    su3_multiply(tmp3, tmp1, plaq);
    su3_accumulate(plaq, Qmunu);

    //--- F_{mu,nu} = (Q - Q^dag) / 8 ---
    SU3matrix Qdag;
    su3_dagger(Qmunu, Qdag);
    su3_subtract(Qmunu, Qdag, Fmunu);
    su3_scale(complex_t(1.0 / 8.0, 0.0), Fmunu, Fmunu);
}

//-----------------------------------------------
// Clover-based energy density:
//   E_clover = -sum_{x, mu>nu} Re Tr(F_{mu,nu}^2) / V
//
// Uses the O(a^2)-improved clover definition of F.
//-----------------------------------------------
real_t wilson_flow_energy_clover(const SU3matrix* gaugeField, int vol)
{
    real_t eSum = 0.0;

    #pragma omp parallel for reduction(+:eSum)
    for (int site = 0; site < vol; site++) {
        for (int mu = 0; mu < 4; mu++) {
            for (int nu = mu + 1; nu < 4; nu++) {
                SU3matrix Fmunu;
                compute_field_strength_tensor(gaugeField, site, mu, nu, Fmunu);

                // Tr(F^2)
                SU3matrix F2;
                su3_multiply(Fmunu, Fmunu, F2);
                eSum -= su3_retrace(F2);
            }
        }
    }

    return eSum / (real_t)vol;
}

//-----------------------------------------------
// Wilson flow driver
//
// Copies the gauge field, evolves t=0 -> maxFlowTime
// in steps of flowStepSize, measuring energy density
// at each step. Writes results to wilson_flow.dat.
// The original gauge field is NOT modified.
//-----------------------------------------------
void wilson_flow_driver(SU3matrix* gaugeField, int vol)
{
    int nLinks = vol * 4;
    int nSteps = (int)(maxFlowTime / flowStepSize + 0.5);

    printf("  Wilson flow: eps=%.4f, t_max=%.2f, %d steps\n",
           flowStepSize, maxFlowTime, nSteps);

//-----------------------------------------------
//   Copy gauge field (original is not modified)
//-----------------------------------------------
    SU3matrix* flowField = new SU3matrix[nLinks];
    std::memcpy(flowField, gaugeField, nLinks * sizeof(SU3matrix));

//-----------------------------------------------
//   Open output file
//-----------------------------------------------
    char filename[512];
    snprintf(filename, sizeof(filename), "%s/wilson_flow.dat",
             outputDirectory.c_str());
    FILE* outFile = fopen(filename, "w");
    if (!outFile) {
        printf("Error: Cannot open %s\n", filename);
        delete[] flowField;
        return;
    }
    fprintf(outFile, "# t  t^2*E_plaq  t^2*E_clover  E_plaq  E_clover\n");

//-----------------------------------------------
//   Measure at t=0
//-----------------------------------------------
    real_t t = 0.0;
    real_t Eplaq = wilson_flow_energy_plaquette(flowField, vol);
    real_t Eclov = wilson_flow_energy_clover(flowField, vol);
    fprintf(outFile, "%.6f  %.10e  %.10e  %.10e  %.10e\n",
            t, t*t*Eplaq, t*t*Eclov, Eplaq, Eclov);

//-----------------------------------------------
//   Flow evolution
//-----------------------------------------------
    for (int step = 1; step <= nSteps; step++) {
        wilson_flow_step(flowField, flowStepSize, vol);
        t = step * flowStepSize;

        Eplaq = wilson_flow_energy_plaquette(flowField, vol);
        Eclov = wilson_flow_energy_clover(flowField, vol);

        fprintf(outFile, "%.6f  %.10e  %.10e  %.10e  %.10e\n",
                t, t*t*Eplaq, t*t*Eclov, Eplaq, Eclov);

        if (step % 10 == 0 || step == nSteps) {
            printf("    t=%.3f  t^2<E_plaq>=%.6f  t^2<E_clover>=%.6f\n",
                   t, t*t*Eplaq, t*t*Eclov);
        }
    }

    fclose(outFile);
    printf("    Flow data saved to: %s\n", filename);

    delete[] flowField;
}

//-----------------------------------------------
// Wilson flow RK3 reversibility test
//
// Checks that flowing forward N steps with +eps
// then backward N steps with -eps returns the
// gauge field to its original value.
//
// Returns the maximum deviation over all link
// matrix elements: max |U_ij(after) - U_ij(before)|.
//-----------------------------------------------
real_t wilson_flow_reversibility_test(const SU3matrix* gaugeField, int vol)
{
    int nLinks = vol * 4;
    int nSteps = 20;
    real_t eps = 0.02;

    printf("\n  === Wilson flow RK3 reversibility test ===\n");
    printf("    Forward %d steps with eps = +%.4f\n", nSteps, eps);
    printf("    Backward %d steps with eps = -%.4f\n", nSteps, eps);

    // Save original field
    SU3matrix* original = new SU3matrix[nLinks];
    std::memcpy(original, gaugeField, nLinks * sizeof(SU3matrix));

    // Working copy
    SU3matrix* workField = new SU3matrix[nLinks];
    std::memcpy(workField, gaugeField, nLinks * sizeof(SU3matrix));

    // Flow forward
    for (int step = 0; step < nSteps; step++) {
        wilson_flow_step(workField, eps, vol);
    }

    // Flow backward
    for (int step = 0; step < nSteps; step++) {
        wilson_flow_step(workField, -eps, vol);
    }

    // Compute max deviation
    real_t maxDev = 0.0;
    for (int idx = 0; idx < nLinks; idx++) {
        for (int i = 0; i < NC; i++) {
            for (int j = 0; j < NC; j++) {
                real_t dev = std::abs(workField[idx].m[i][j]
                                    - original[idx].m[i][j]);
                if (dev > maxDev) maxDev = dev;
            }
        }
    }

    printf("    Max |U_ij(after) - U_ij(before)| = %.6e\n", maxDev);
    printf("  === End reversibility test ===\n\n");

    delete[] original;
    delete[] workField;

    return maxDev;
}
