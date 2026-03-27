//-----------------------------------------------
// Clover fermion force for HMC
//
// The correct force matrix for each clover leaf
// L = left * U_mu(x) * right at site z is:
//
//   M_{alpha,beta} = sum_{ss'} sigma_{ss'} *
//     [left^T * Y*_s(z)]_alpha * [right * X_{s'}(z)]_beta
//
// This requires splitting the staple into left/right
// parts around U_mu(x) for each of the 4 leaf positions.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "lattice_module.hpp"
#include "colorspinor_module.hpp"
#include "gamma_matrices.hpp"
#include "clover_module.hpp"
#include "interfaces_module.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

//-----------------------------------------------
// Compute force matrix from one clover leaf.
//
// The leaf is L = left * U * right, with U removed.
// Force: M_{alpha,beta} = sum_{ss'} sigma_{ss'} *
//   v_s_alpha * w_{s'}_beta
//
// where v_s = left^T * conj(Y_s(z)) and
//       w_{s'} = right * X_{s'}(z)
//-----------------------------------------------
static void leaf_force_matrix(
    const SU3matrix* left,    // left part (or nullptr for identity)
    const SU3matrix* right,   // right part (or nullptr for identity)
    const ColorSpinor& Xz,
    const ColorSpinor& Yz,
    int mu, int nu,
    SU3matrix& M)
{
    // Get sigma matrix
    int mA = (mu < nu) ? mu : nu;
    int mB = (mu < nu) ? nu : mu;
    // No sign flip: sigma_{nu,mu}*F_{nu,mu} = (-sigma)*(-F) = +sigma*F
    const complex_t* sig = get_sigma_matrix(mA, mB);

    // Compute v_s = left^T * Y*_s and w_{s'} = right * X_{s'}
    // (NC-component color vectors for each spin)
    complex_t v[ND][NC], w[ND][NC];

    for (int s = 0; s < ND; s++) {
        if (left) {
            // v_s = left^T * conj(Y_s) = sum_a left_{a,alpha}^* ...
            // Actually left^T_{alpha,a} = left_{a,alpha}
            for (int alpha = 0; alpha < NC; alpha++) {
                v[s][alpha] = complex_t(0, 0);
                for (int a = 0; a < NC; a++)
                    v[s][alpha] += left->m[a][alpha] * std::conj(Yz.v[s][a]);
            }
        } else {
            // left = identity
            for (int alpha = 0; alpha < NC; alpha++)
                v[s][alpha] = std::conj(Yz.v[s][alpha]);
        }
    }

    for (int sp = 0; sp < ND; sp++) {
        if (right) {
            for (int beta = 0; beta < NC; beta++) {
                w[sp][beta] = complex_t(0, 0);
                for (int b = 0; b < NC; b++)
                    w[sp][beta] += right->m[beta][b] * Xz.v[sp][b];
            }
        } else {
            // right = identity
            for (int beta = 0; beta < NC; beta++)
                w[sp][beta] = Xz.v[sp][beta];
        }
    }

    // M = sum_{ss'} sigma_{ss'} * v_s ⊗ w_{s'}
    for (int s = 0; s < ND; s++) {
        for (int sp = 0; sp < ND; sp++) {
            complex_t sigma_ss = sig[s * ND + sp];
            if (std::abs(sigma_ss) < 1e-15) continue;

            for (int alpha = 0; alpha < NC; alpha++)
                for (int beta = 0; beta < NC; beta++)
                    M.m[alpha][beta] += sigma_ss * v[s][alpha] * w[sp][beta];
        }
    }
}

void compute_clover_force(const SU3matrix* gaugeField,
                          const ColorSpinor* X,
                          const ColorSpinor* Y,
                          SU3matrix* force,
                          int vol)
{
    printf("  Computing clover fermion force (c_sw = %.4f)...\n", c_sw);

    real_t prefactor = -c_sw / 32.0;

    #pragma omp parallel for
    for (int site = 0; site < vol; site++) {
        for (int mu = 0; mu < 4; mu++) {
            int idx = site * 4 + mu;
            int xPmu = neighborPlus[site * 4 + mu];

            SU3matrix Mtot;
            su3_zero(Mtot);

            for (int nu = 0; nu < 4; nu++) {
                if (nu == mu) continue;

                if (nu > mu) {
                    //=== UPPER STAPLE (nu > mu): U_mu is PRIMARY ===
                    // Leaf 1 at z=x:     L = U * [B1] where B1 = staple
                    // Leaf 2 at z=x+mu:  L = [A2] * U where A2 = staple
                    // Leaf 3 at z=x+mu+nu: L = [A3] * U * [B3]
                    // Leaf 4 at z=x+nu:    L = [A4] * U * [B4]

                    int xPnu    = neighborPlus[site * 4 + nu];
                    int xPmuPnu = neighborPlus[xPmu * 4 + nu];

                    // Staple components
                    SU3matrix Unu_xPmu = gaugeField[xPmu * 4 + nu];
                    SU3matrix UmuDag_xPnu; su3_dagger(gaugeField[xPnu * 4 + mu], UmuDag_xPnu);
                    SU3matrix UnuDag_x;   su3_dagger(gaugeField[site * 4 + nu], UnuDag_x);

                    // Full staple S = Unu(x+mu) * Umu†(x+nu) * Unu†(x)
                    SU3matrix S12, S;
                    su3_multiply(Unu_xPmu, UmuDag_xPnu, S12);
                    su3_multiply(S12, UnuDag_x, S);

                    // Partial products for split staples
                    SU3matrix S23;
                    su3_multiply(UmuDag_xPnu, UnuDag_x, S23);

                    // Leaf 1: left=I, right=S
                    leaf_force_matrix(nullptr, &S, X[site], Y[site], mu, nu, Mtot);

                    // Leaf 2: left=S, right=I
                    leaf_force_matrix(&S, nullptr, X[xPmu], Y[xPmu], mu, nu, Mtot);

                    // Leaf 3: left=S23^dag_transposed...
                    // A3 = Umu†(x+nu) * Unu†(x), B3 = Unu(x+mu)
                    leaf_force_matrix(&S23, &Unu_xPmu, X[xPmuPnu], Y[xPmuPnu], mu, nu, Mtot);

                    // Leaf 4: A4 = Unu†(x), B4 = Unu(x+mu) * Umu†(x+nu)
                    leaf_force_matrix(&UnuDag_x, &S12, X[xPnu], Y[xPnu], mu, nu, Mtot);

                } else {
                    //=== LOWER STAPLE (nu < mu): U_mu is SECONDARY ===
                    // In Q_{nu,mu}, U_mu appears at different positions
                    // Staple links: Unu†(x+mu-nu), Umu†(x-nu), Unu(x-nu)

                    int xMnu    = neighborMinus[site * 4 + nu];
                    int xPmuMnu = neighborMinus[xPmu * 4 + nu];

                    SU3matrix UnuDag_xPmuMnu; su3_dagger(gaugeField[xPmuMnu * 4 + nu], UnuDag_xPmuMnu);
                    SU3matrix UmuDag_xMnu;    su3_dagger(gaugeField[xMnu * 4 + mu], UmuDag_xMnu);
                    SU3matrix Unu_xMnu = gaugeField[xMnu * 4 + nu];

                    SU3matrix Sp12, Sp;
                    su3_multiply(UnuDag_xPmuMnu, UmuDag_xMnu, Sp12);
                    su3_multiply(Sp12, Unu_xMnu, Sp);

                    SU3matrix Sp23;
                    su3_multiply(UmuDag_xMnu, Unu_xMnu, Sp23);

                    // Leaf 1': left=I, right=Sp
                    leaf_force_matrix(nullptr, &Sp, X[site], Y[site], mu, nu, Mtot);

                    // Leaf 2': left=Sp, right=I
                    leaf_force_matrix(&Sp, nullptr, X[xPmu], Y[xPmu], mu, nu, Mtot);

                    // Leaf 3': left=Sp23, right=UnuDag_xPmuMnu
                    leaf_force_matrix(&Sp23, &UnuDag_xPmuMnu, X[xPmuMnu], Y[xPmuMnu], mu, nu, Mtot);

                    // Leaf 4': left=Unu_xMnu, right=Sp12
                    leaf_force_matrix(&Unu_xMnu, &Sp12, X[xMnu], Y[xMnu], mu, nu, Mtot);
                }
            } // end nu

            // F^cl = prefactor * [U · M]_{TA}
            SU3matrix UM, forceTA;
            su3_multiply(gaugeField[idx], Mtot, UM);
            su3_traceless_antiherm(UM, forceTA);

            for (int a = 0; a < NC; a++)
                for (int b = 0; b < NC; b++)
                    force[idx].m[a][b] += complex_t(prefactor, 0.0) * forceTA.m[a][b];

        } // end mu
    } // end site

    printf("  Clover force computed.\n");
}
