//-----------------------------------------------
// Clover (Sheikholeslami-Wohlert) fermion force for HMC
//
// Derived directly from the EXACT clover loops used in
// compute_field_strength_tensor (wilson_flow.cpp):
//
//   F_{ab}(z) = (Q_{ab}(z) - Q_{ab}^dag(z)) / 8
//   Q_{ab}(z) = L1 + L2 + L3 + L4   (the 4-leaf clover)
//   A(z)      = c_sw * (i/4) * sum_{a<b} sigma_{ab} (x) F_{ab}(z)
//
// The pseudofermion action S = phi^dag (D^dag D)^{-1} phi with
// X = (D^dag D)^{-1} phi, Y = D X gives, for the clover part,
//
//   dS_cl = -( Y^dag dA X + X^dag dA Y )          (A is Hermitian)
//         = -(c_sw i/4)(1/8) sum_z sum_{a<b}
//             Tr[ (dQ_{ab}(z) - dQ_{ab}^dag(z)) * R(z) ]
//
// where the spin-traced colour source (3x3, index [d][c]) is
//
//   R(z)_{dc} = sum_{s,s'} sigma^{(ab)}_{s s'}
//                 [ X_{s'}(z)_d conj(Y_s(z)_c)
//                 + Y_{s'}(z)_d conj(X_s(z)_c) ]
//
// Each clover loop is a product of 4 links V0 V1 V2 V3. For the
// link at position i (the object U = U_dir(s), appearing as U or
// U^dag), with P_left = V0..V_{i-1}, P_right = V_{i+1}..V3:
//
//   M_i = P_right * R(z) * P_left
//   U  position : Phi += U * M_i
//   U^dag pos.  : Phi -= M_i * U^dag
//
// summed over the 4 loops (Q) and their reversed-daggered
// partners (-Q^dag). The per-link force is C * TA(Phi), with C a
// single real constant (sign/magnitude fixed by the numerical
// gradient check; the i and the -2 Re fold into C because R is
// built from the Hermitian sigma).
//
// The CLOVER_DIAG_PLANE selector (clover_module.hpp) restricts
// this AND the clover field (hence S_PF / the numerical gradient)
// to one (a,b) plane for term-by-term validation.
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
// Site shift by +/- one lattice unit in a direction.
//-----------------------------------------------
static inline int shiftP(int site, int dir) { return neighborPlus[site * 4 + dir]; }
static inline int shiftM(int site, int dir) { return neighborMinus[site * 4 + dir]; }

//-----------------------------------------------
// One link reference inside a clover loop.
//   site, dir   : the gauge link U_dir(site)
//   dag         : 0 -> appears as U, 1 -> appears as U^dag
//-----------------------------------------------
struct LinkRef { int site, dir, dag; };

//-----------------------------------------------
// Build the 4 links of clover leaf k (0..3) of Q_{a,b}(z),
// in path order, matching compute_field_strength_tensor.
//-----------------------------------------------
static void build_leaf(int z, int a, int b, int k, LinkRef L[4])
{
    int za  = shiftP(z, a);
    int zb  = shiftP(z, b);
    int zma = shiftM(z, a);
    int zmb = shiftM(z, b);

    if (k == 0) {
        // L1: U_a(z) U_b(z+a) U_a^dag(z+b) U_b^dag(z)
        L[0] = {z,  a, 0};
        L[1] = {za, b, 0};
        L[2] = {zb, a, 1};
        L[3] = {z,  b, 1};
    } else if (k == 1) {
        // L2: U_b(z) U_a^dag(z-a+b) U_b^dag(z-a) U_a(z-a)
        int zmapb = shiftP(zma, b);
        L[0] = {z,     b, 0};
        L[1] = {zmapb, a, 1};
        L[2] = {zma,   b, 1};
        L[3] = {zma,   a, 0};
    } else if (k == 2) {
        // L3: U_a^dag(z-a) U_b^dag(z-a-b) U_a(z-a-b) U_b(z-b)
        int zmamb = shiftM(zma, b);
        L[0] = {zma,   a, 1};
        L[1] = {zmamb, b, 1};
        L[2] = {zmamb, a, 0};
        L[3] = {zmb,   b, 0};
    } else {
        // L4: U_b^dag(z-b) U_a(z-b) U_b(z-b+a) U_a^dag(z)
        int zmbpa = shiftP(zmb, a);
        L[0] = {zmb,   b, 1};
        L[1] = {zmb,   a, 0};
        L[2] = {zmbpa, b, 0};
        L[3] = {z,     a, 1};
    }
}

// Matrix of a link reference (U or U^dag).
static inline void link_matrix(const SU3matrix* gaugeField,
                               const LinkRef& L, SU3matrix& V)
{
    const SU3matrix& U = gaugeField[L.site * 4 + L.dir];
    if (L.dag) su3_dagger(U, V);
    else       V = U;
}

void compute_clover_force(const SU3matrix* gaugeField,
                          const ColorSpinor* X,
                          const ColorSpinor* Y,
                          SU3matrix* force,
                          int vol)
{
    printf("  Computing clover fermion force (c_sw = %.4f)...\n", c_sw);

    const int diagPlane = clover_diag_plane();

    // Single real constant: -2 Re, (c_sw i/4), (1/8) from F=(Q-Qdag)/8,
    // and the i absorbed via the Hermitian sigma all collapse to a real
    // coefficient.  Sign/magnitude validated by gradient_check.
    const real_t C = c_sw / 32.0;

    // Per-link force accumulator (color matrix, pre-TA).
    int nLinks = vol * 4;
    SU3matrix* acc = new SU3matrix[nLinks];
    for (int i = 0; i < nLinks; i++) su3_zero(acc[i]);

    for (int a = 0; a < 4; a++) {
        for (int bb = a + 1; bb < 4; bb++) {
            int sigIdx = get_sigma_pair_index(a, bb);
            if (diagPlane >= 0 && sigIdx != diagPlane) continue;

            const complex_t* sig = get_sigma_matrix(a, bb);

            for (int z = 0; z < vol; z++) {

                // Spin-traced colour source R(z)_{dc}
                //   = sum_{ss'} sigma_{ss'} [ X_{s'}_d Y*_s_c + Y_{s'}_d X*_s_c ]
                SU3matrix R;
                su3_zero(R);
                for (int s = 0; s < ND; s++) {
                    for (int sp = 0; sp < ND; sp++) {
                        complex_t sg = sig[s * ND + sp];
                        if (std::abs(sg) < 1e-15) continue;
                        for (int d = 0; d < NC; d++)
                            for (int c = 0; c < NC; c++)
                                R.m[d][c] += sg *
                                    ( X[z].v[sp][d] * std::conj(Y[z].v[s][c])
                                    + Y[z].v[sp][d] * std::conj(X[z].v[s][c]) );
                    }
                }

                // Q loops (sign +1) and their reversed-daggered
                // partners -Q^dag (sign -1).
                for (int k = 0; k < 4; k++) {
                    LinkRef Lp[4];
                    build_leaf(z, a, bb, k, Lp);

                    SU3matrix V[4];
                    for (int q = 0; q < 4; q++)
                        link_matrix(gaugeField, Lp[q], V[q]);

                    for (int i = 0; i < 4; i++) {
                        // A = P_left = V0..V_{i-1}, B = P_right = V_{i+1}..V3
                        SU3matrix A, B, tmp;
                        su3_identity(A);
                        for (int q = 0; q < i; q++) {
                            su3_multiply(A, V[q], tmp); A = tmp;
                        }
                        su3_identity(B);
                        for (int q = i + 1; q < 4; q++) {
                            su3_multiply(B, V[q], tmp); B = tmp;
                        }

                        // BRA = B * R * A
                        SU3matrix BRA, t2;
                        su3_multiply(B, R, t2);
                        su3_multiply(t2, A, BRA);

                        // Ad R Bd = A^dag * R * B^dag  (the -dQ^dag companion)
                        SU3matrix Ad, Bd, ARBd, t3;
                        su3_dagger(A, Ad);
                        su3_dagger(B, Bd);
                        su3_multiply(Ad, R, t3);
                        su3_multiply(t3, Bd, ARBd);

                        const LinkRef& Li = Lp[i];
                        const SU3matrix& U = gaugeField[Li.site * 4 + Li.dir];
                        SU3matrix Ud; su3_dagger(U, Ud);

                        // dS_cl ~ Tr[(dQ - dQ^dag) R]; per occurrence the
                        // coefficient of G = i*Ta is Phi = Phi_dQ - Phi_dQdag:
                        //   V=U  : Phi = U*B*R*A  +  A^dag*R*B^dag*U^dag
                        //   V=U^d: Phi = -B*R*A*U^dag  -  U*A^dag*R*B^dag
                        SU3matrix Phi, ta, tb;
                        if (Li.dag == 0) {
                            su3_multiply(U, BRA, ta);    // U*B*R*A
                            su3_multiply(ARBd, Ud, tb);  // A^d*R*B^d*U^d
                            for (int p = 0; p < NC; p++)
                                for (int qq = 0; qq < NC; qq++)
                                    Phi.m[p][qq] = ta.m[p][qq] + tb.m[p][qq];
                        } else {
                            su3_multiply(BRA, Ud, ta);   // B*R*A*U^d
                            su3_multiply(U, ARBd, tb);   // U*A^d*R*B^d
                            for (int p = 0; p < NC; p++)
                                for (int qq = 0; qq < NC; qq++)
                                    Phi.m[p][qq] = -ta.m[p][qq] - tb.m[p][qq];
                        }

                        int li = Li.site * 4 + Li.dir;
                        for (int p = 0; p < NC; p++)
                            for (int qq = 0; qq < NC; qq++)
                                acc[li].m[p][qq] += Phi.m[p][qq];
                    }
                }
            }
        }
    }

    // F^cl[link] += (i*C) * TA(acc[link])
    // The clover coefficient is c_sw*(i/4); the explicit i must be
    // carried (Re Tr[iTa . (real*acc)] would vanish otherwise).
    for (int li = 0; li < nLinks; li++) {
        SU3matrix accI;
        for (int p = 0; p < NC; p++)
            for (int q = 0; q < NC; q++)
                accI.m[p][q] = complex_t(0.0, 1.0) * acc[li].m[p][q];
        SU3matrix ta;
        su3_traceless_antiherm(accI, ta);
        for (int p = 0; p < NC; p++)
            for (int q = 0; q < NC; q++)
                force[li].m[p][q] += complex_t(C, 0.0) * ta.m[p][q];
    }

    delete[] acc;
    printf("  Clover force computed.\n");
}
