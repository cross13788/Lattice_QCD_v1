//-----------------------------------------------
// Numerical gradient check for the fermion force
//
// Computes dS/dU numerically via finite difference
// and compares with the analytical force.
//
// For each su(3) generator T_a at a chosen link:
//   U' = exp(eps * i*T_a) * U
//   dS_num = (S(U') - S(U)) / eps
//   dS_ana = Tr[i*T_a * U * Force_matrix]  (from TA projection)
//
// A correct force gives dS_num / dS_ana = 1.0.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "lattice_module.hpp"
#include "colorspinor_module.hpp"
#include "interfaces_module.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

// The 8 Gell-Mann matrices (su(3) generators T_a = lambda_a / 2)
// We only need a few for the check
static void gellmann(int a, SU3matrix& T)
{
    su3_zero(T);
    switch (a) {
        case 0: // lambda_1 / 2
            T.m[0][1] = complex_t(0.5, 0.0);
            T.m[1][0] = complex_t(0.5, 0.0);
            break;
        case 1: // lambda_2 / 2
            T.m[0][1] = complex_t(0.0, -0.5);
            T.m[1][0] = complex_t(0.0, 0.5);
            break;
        case 2: // lambda_3 / 2
            T.m[0][0] = complex_t(0.5, 0.0);
            T.m[1][1] = complex_t(-0.5, 0.0);
            break;
        case 3: // lambda_4 / 2
            T.m[0][2] = complex_t(0.5, 0.0);
            T.m[2][0] = complex_t(0.5, 0.0);
            break;
        case 4: // lambda_5 / 2
            T.m[0][2] = complex_t(0.0, -0.5);
            T.m[2][0] = complex_t(0.0, 0.5);
            break;
        case 5: // lambda_6 / 2
            T.m[1][2] = complex_t(0.5, 0.0);
            T.m[2][1] = complex_t(0.5, 0.0);
            break;
        case 6: // lambda_7 / 2
            T.m[1][2] = complex_t(0.0, -0.5);
            T.m[2][1] = complex_t(0.0, 0.5);
            break;
        case 7: // lambda_8 / 2
            T.m[0][0] = complex_t(0.5 / sqrt(3.0), 0.0);
            T.m[1][1] = complex_t(0.5 / sqrt(3.0), 0.0);
            T.m[2][2] = complex_t(-1.0 / sqrt(3.0), 0.0);
            break;
    }
}

void gradient_check(SU3matrix* gaugeField,
                    const ColorSpinor* phi,
                    int vol)
{
    printf("\n===============================================\n");
    printf("  Numerical Gradient Check: Fermion Force\n");
    printf("  (finite difference vs analytical force)\n");
    printf("===============================================\n");

    int nLinks = vol * 4;

    // Step 1: Compute analytical force (total = Wilson + clover)
    SU3matrix* force = new SU3matrix[nLinks];
    for (int i = 0; i < nLinks; i++) su3_zero(force[i]);

    compute_fermion_force(gaugeField, phi, force, vol);

    // Also compute Wilson-only force for comparison
    SU3matrix* forceW = new SU3matrix[nLinks];
    for (int i = 0; i < nLinks; i++) su3_zero(forceW[i]);

    bool savedClover = useClover;
    useClover = false;
    compute_fermion_force(gaugeField, phi, forceW, vol);
    useClover = savedClover;

    // Step 2: Compute S(U) at current gauge field
    real_t S0 = compute_pseudofermion_action(gaugeField, phi, vol);

    // Step 3: For a few links and generators, do finite-difference check
    real_t eps = 1.0e-5;

    // Test at 3 different sites and all 4 directions
    int testSites[] = {0, vol / 4, vol / 2};
    int nTestSites = 3;

    printf("\n  eps = %.1e\n", eps);
    printf("  S_PF(U) = %.10f\n\n", S0);
    printf("  %-4s %-3s %-3s  %-12s  %-12s  %-12s  %-8s  %-8s\n",
           "site", "mu", "Ta", "dS_num", "dS_ana_tot", "dS_ana_cl",
           "ratio", "cl_rat");
    printf("  ---------------------------------------------------------------\n");

    real_t totalRelErr = 0.0;
    int nTests = 0;

    for (int si = 0; si < nTestSites; si++) {
        int site = testSites[si];

        for (int mu = 0; mu < 4; mu++) {
            int idx = site * 4 + mu;

            // Test generators 0, 2, 4 (mix of diagonal and off-diagonal)
            for (int genIdx = 0; genIdx < 8; genIdx += 2) {
                SU3matrix Ta;
                gellmann(genIdx, Ta);

                // Analytical: dS/deps_a = Re Tr[i*T_a * force[idx]]
                // The force array stores the TA-projected result.
                // For the fermion force F = -2*kappa*[U*M]_TA + clover:
                // dS/deps = sum_a eps_a * Tr[i*T_a * F]
                // For a single direction a:
                // dS_ana = Tr[i*T_a * force[idx]]
                //
                // But force is stored as the TOTAL force (gauge + fermion).
                // We only have fermion force here. The analytical derivative
                // of S_PF should match.

                SU3matrix iTa;
                for (int i = 0; i < NC; i++)
                    for (int j = 0; j < NC; j++)
                        iTa.m[i][j] = complex_t(0.0, 1.0) * Ta.m[i][j];

                // dS_ana = Re Tr[iTa * force]
                complex_t tr(0.0, 0.0);
                for (int i = 0; i < NC; i++)
                    for (int k = 0; k < NC; k++)
                        tr += iTa.m[i][k] * force[idx].m[k][i];
                real_t dS_ana = tr.real();

                // Numerical: perturb U_mu(x) -> exp(eps * iTa) * U_mu(x)
                SU3matrix epsT, expEpsT, Uorig, Unew;
                for (int i = 0; i < NC; i++)
                    for (int j = 0; j < NC; j++)
                        epsT.m[i][j] = eps * iTa.m[i][j];

                su3_exp(epsT, expEpsT);

                // Save original link
                Uorig = gaugeField[idx];

                // Perturb
                su3_multiply(expEpsT, Uorig, Unew);
                gaugeField[idx] = Unew;

                // Invalidate clover cache if using clover
                if (useClover) invalidate_clover_cache();

                real_t S1 = compute_pseudofermion_action(gaugeField, phi, vol);

                // Restore
                gaugeField[idx] = Uorig;
                if (useClover) invalidate_clover_cache();

                real_t dS_num = (S1 - S0) / eps;

                // All derivatives use convention: dS/deps (positive = action increases)
                // Force = -dS/dU, so dS/deps = -Tr[iTa * Force]

                // Wilson-only analytical
                complex_t trW(0.0, 0.0);
                for (int ii = 0; ii < NC; ii++)
                    for (int kk = 0; kk < NC; kk++)
                        trW += iTa.m[ii][kk] * forceW[idx].m[kk][ii];
                real_t gradW = -trW.real();     // dS_W/deps

                // Total analytical
                real_t gradTot = -dS_ana;        // dS_total/deps = -Tr[iTa*F_total]

                // Clover-only analytical
                real_t gradCl = gradTot - gradW; // dS_cl/deps

                // Numerical clover = numerical total - analytical Wilson
                real_t gradCl_num = dS_num - gradW;

                // Ratios
                real_t ratio = (std::fabs(gradTot) > 1e-15) ?
                               dS_num / gradTot : 0.0;
                real_t cl_ratio = (std::fabs(gradCl) > 1e-12) ?
                                  gradCl_num / gradCl : 0.0;

                printf("  %-4d %-3d %-3d  %+12.4e  %+12.4e  %+12.4e  %8.4f  %8.4f\n",
                       site, mu, genIdx, dS_num, gradTot, gradCl,
                       ratio, cl_ratio);

                if (std::fabs(gradTot) > 1e-10) {
                    totalRelErr += std::fabs(ratio - 1.0);
                    nTests++;
                }
            }
        }
    }

    if (nTests > 0) {
        printf("\n  Average |ratio - 1| = %.6f  (%d tests)\n",
               totalRelErr / nTests, nTests);
        printf("  (Perfect force gives ratio = 1.000000)\n");
    }

    printf("===============================================\n\n");

    delete[] force;
    delete[] forceW;
}
