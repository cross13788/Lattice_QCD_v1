//-----------------------------------------------
// Meson two-point correlator
//
// C_Gamma(t) = sum_x Tr[ Gamma * S(x,t; 0) *
//                         gamma5 * S^dag(x,t; 0) * gamma5 * Gamma^dag ]
//
// Using gamma5-hermiticity: S^dag = gamma5 * S^{-1 dag} * gamma5
// the correlator simplifies to:
//
// C_Gamma(t) = -sum_x Tr[ Gamma * S(x,t; 0) *
//                          Gamma^dag * gamma5 * S^dag(x,t; 0) * gamma5 ]
//
// For the pion (Gamma = gamma5):
//   C_pi(t) = sum_x sum_{s,c} |S(x,t; 0)_{s,c}|^2
//
// For general Gamma, we use the propagator components
// directly.
//
// Supported channels:
//   0 = pion     (Gamma = gamma5)
//   1 = rho_1    (Gamma = gamma1)
//   2 = rho_2    (Gamma = gamma2)
//   3 = rho_3    (Gamma = gamma3)
//   4 = scalar   (Gamma = I)
//   5 = a1_1     (Gamma = gamma5 * gamma1)
//   6 = a1_2     (Gamma = gamma5 * gamma2)
//   7 = a1_3     (Gamma = gamma5 * gamma3)
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "lattice_module.hpp"
#include "colorspinor_module.hpp"
#include "gamma_matrices.hpp"
#include "interfaces_module.hpp"
#include <cstdio>
#include <cstring>

//-----------------------------------------------
// Apply a gamma matrix to a ColorSpinor:
//   result = Gamma * psi
//
// gammaIndex: 0..4 (gamma1..gamma5)
//-----------------------------------------------
static void apply_gamma(int gammaIndex, const ColorSpinor& psi, ColorSpinor& result)
{
    for (int s = 0; s < ND; s++) {
        int gs = gammaIdx[gammaIndex][s];
        complex_t gv = gammaVal[gammaIndex][s];
        for (int c = 0; c < NC; c++) {
            result.v[s][c] = gv * psi.v[gs][c];
        }
    }
}

//-----------------------------------------------
// Compute meson correlator C(t) for a given
// Gamma structure.
//
// The propagator array has layout:
//   propagator[12 * site + (s0 * NC + c0)]
// where (s0,c0) is the source spin-color index.
//
// C_Gamma(t) = -sum_x Tr[ Gamma * S(x;0) *
//    Gamma^dag * gamma5 * S^dag(x;0) * gamma5 ]
//
// The Gamma^dag factor acts on the source spin
// index, mixing different source spin components.
//-----------------------------------------------
void meson_correlator(const ColorSpinor* propagator,
                      int channel,
                      real_t* corr,
                      int vol)
{
//-----------------------------------------------
//   Zero the correlator array
//-----------------------------------------------
    for (int t = 0; t < nT; t++)
        corr[t] = 0.0;

//-----------------------------------------------
//   Determine Gamma structure for this channel
//   For composite Gammas (gamma5*gamma_i), we
//   apply them sequentially.
//-----------------------------------------------
    bool isComposite = false;
    int gamma1st = -1, gamma2nd = -1;

    switch (channel) {
        case 0: gamma1st = 4; break;                      // pion: gamma5
        case 1: gamma1st = 0; break;                      // rho_1: gamma1
        case 2: gamma1st = 1; break;                      // rho_2: gamma2
        case 3: gamma1st = 2; break;                      // rho_3: gamma3
        case 4: gamma1st = -1; break;                     // scalar: identity
        case 5: gamma1st = 4; gamma2nd = 0; isComposite = true; break;  // a1_1
        case 6: gamma1st = 4; gamma2nd = 1; isComposite = true; break;  // a1_2
        case 7: gamma1st = 4; gamma2nd = 2; isComposite = true; break;  // a1_3
        default: gamma1st = 4; break;
    }

//-----------------------------------------------
//   Pion correlator (special case — simplest)
//   C_pi(t) = sum_x sum_{s0,c0} sum_{s,c}
//             |S(x,t; 0)_{s,c; s0,c0}|^2
//-----------------------------------------------
    if (channel == 0) {
        real_t* corrLocal = new real_t[nT]();

        #pragma omp parallel
        {
            real_t* corrThread = new real_t[nT]();

            #pragma omp for
            for (int site = 0; site < vol; site++) {
                int x, y, z, t;
                site_coords(site, x, y, z, t);

                for (int srcIdx = 0; srcIdx < 12; srcIdx++) {
                    const ColorSpinor& prop = propagator[12 * site + srcIdx];
                    corrThread[t] += cs_norm2(prop);
                }
            }

            #pragma omp critical
            {
                for (int t = 0; t < nT; t++)
                    corrLocal[t] += corrThread[t];
            }

            delete[] corrThread;
        }

        for (int t = 0; t < nT; t++)
            corr[t] = corrLocal[t];

        delete[] corrLocal;
        return;
    }

//-----------------------------------------------
//   General meson correlator
//
//   C_Gamma(t) = -sum_x Tr[ Gamma * S(x;0) *
//                  Gamma^dag * gamma5 * S^dag(x;0) * gamma5 ]
//
//   In components (M = S(x,0), row=(s,c), col=(s0,c0)):
//
//   C(t) = -sum_x sum_{s,c,s0,c0}
//     [Gamma * M]_{(s,c),(s0,c0)}
//     * g5(s) * { sum_{a} (Gamma^dag)_{s0,a} * g5(a) *
//                 conj(S_{s,c; a,c0}) }
//
//   Gamma^dag mixes the source spin index, so we need
//   access to propagators with different source spins
//   for the same source color.
//
//   All gamma matrices in this basis are Hermitian, so
//   Gamma^dag = Gamma. Since each gamma is a sparse
//   permutation-like matrix (one nonzero per row and
//   per column), the permutation is self-inverse, so:
//     gammaIdx[mu][gammaIdx[mu][s]] = s
//-----------------------------------------------
    real_t* corrLocal = new real_t[nT]();

    #pragma omp parallel
    {
        real_t* corrThread = new real_t[nT]();

        #pragma omp for
        for (int site = 0; site < vol; site++) {
            int xc, yc, zc, tc;
            site_coords(site, xc, yc, zc, tc);

            for (int c0 = 0; c0 < NC; c0++) {
                // Preload propagators for all source spins at this
                // source color, since Gamma^dag mixes source spins.
                const ColorSpinor* Scol[ND];
                for (int s0 = 0; s0 < ND; s0++) {
                    Scol[s0] = &propagator[12 * site + s0 * NC + c0];
                }

                for (int s0 = 0; s0 < ND; s0++) {
                    const ColorSpinor& S = *Scol[s0];

                    // Forward propagator: apply Gamma on sink spin
                    // GammaS_{s,c; s0,c0} = Gamma_{s,s'} S_{s',c; s0,c0}
                    ColorSpinor GammaS;
                    if (channel == 4) {
                        cs_copy(S, GammaS);
                    } else if (!isComposite) {
                        apply_gamma(gamma1st, S, GammaS);
                    } else {
                        ColorSpinor tmp;
                        apply_gamma(gamma2nd, S, tmp);
                        apply_gamma(gamma1st, tmp, GammaS);
                    }

                    // Backward propagator factor:
                    // B_{s0; s,c} = sum_a (Gamma^dag)_{s0,a} * g5(a) * conj(S_{s,c;a,c0})
                    //
                    // For sparse Gamma^dag, only one 'a' contributes per s0.
                    // Since gamma matrices are Hermitian and their permutations
                    // are self-inverse: the unique 'a' such that
                    // (Gamma^dag)_{s0,a} != 0 can be found via the same
                    // gammaIdx table.

                    // Determine the source spin index 'a' and value
                    // of (Gamma^dag)_{s0, a} for each channel type.
                    int a;
                    complex_t gdagTimesG5;

                    if (channel == 4) {
                        // Gamma = I: a = s0, (Gamma^dag)_{s0,a} = 1
                        a = s0;
                        gdagTimesG5 = gammaVal[4][a].real();  // 1 * g5(a)
                    } else if (!isComposite) {
                        // Gamma = gamma_mu (Hermitian), Gamma^dag = gamma_mu
                        // (Gamma^dag)_{s0,a} = Gamma_{s0,a}
                        // For sparse gamma: a = gammaIdx[mu][s0],
                        // value = gammaVal[mu][s0]
                        a = gammaIdx[gamma1st][s0];
                        gdagTimesG5 = gammaVal[gamma1st][s0] * gammaVal[4][a].real();
                    } else {
                        // Gamma = gamma5 * gamma_i, Gamma^dag = gamma_i * gamma5
                        // (gamma_i * gamma5)_{s0, a}:
                        //   gamma_i maps s0 -> gammaIdx[i][s0] with val gammaVal[i][s0]
                        //   gamma5 maps that to same index (diagonal) with val g5(a)
                        //   So a = gammaIdx[i][s0], value = gammaVal[i][s0] * g5(a)
                        a = gammaIdx[gamma2nd][s0];
                        gdagTimesG5 = gammaVal[gamma2nd][s0]
                                    * gammaVal[4][a].real()     // gamma5 from Gamma^dag
                                    * gammaVal[4][a].real();    // gamma5 from formula
                    }

                    for (int s = 0; s < ND; s++) {
                        real_t g5_s = gammaVal[4][s].real();

                        for (int c = 0; c < NC; c++) {
                            complex_t backprop = gdagTimesG5
                                               * std::conj(Scol[a]->v[s][c]);
                            corrThread[tc] += (GammaS.v[s][c] * g5_s * backprop).real();
                        }
                    }
                }
            }
        }

        #pragma omp critical
        {
            for (int t = 0; t < nT; t++)
                corrLocal[t] += corrThread[t];
        }

        delete[] corrThread;
    }

    for (int t = 0; t < nT; t++)
        corr[t] = -corrLocal[t];   // Overall minus sign from fermion contraction

    delete[] corrLocal;
}
