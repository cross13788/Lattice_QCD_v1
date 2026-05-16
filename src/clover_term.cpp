//-----------------------------------------------
// Clover term: compute F_{mu,nu} and assemble the
// clover matrix A(x) = c_sw * (i/4) * sum_{mu,nu} sigma_{mu,nu} F_{mu,nu}
//
// The sigma matrices sigma_{mu,nu} = (i/2)[gamma_mu, gamma_nu]
// are precomputed at startup. In the DeGrand-Rossi basis they
// are block-diagonal: sigma_{mu,nu} acts within (spin 0,1) and
// (spin 2,3) separately.
//
// The clover field at each site stores two Hermitian 6x6 blocks
// (upper for spin 0,1 and lower for spin 2,3), each in the
// combined spin-color space.
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
#include <cstring>
#include <cmath>
#include <cstdlib>

//-----------------------------------------------
// Diagnostic isolation selectors (see clover_module.hpp).
// Lazily read from the environment exactly once.
//-----------------------------------------------
static int read_clover_diag_env(const char* key)
{
    const char* v = std::getenv(key);
    return (v != nullptr) ? std::atoi(v) : -1;
}
int clover_diag_plane()
{
    static int p = read_clover_diag_env("CLOVER_DIAG_PLANE");
    return p;
}
int clover_diag_leaf()
{
    static int l = read_clover_diag_env("CLOVER_DIAG_LEAF");
    return l;
}

//-----------------------------------------------
// sigma_{mu,nu} = (i/2) * [gamma_mu, gamma_nu]
//
// For 6 independent (mu,nu) pairs with mu < nu.
// Each sigma is a 4x4 complex matrix in Dirac space.
// We store the full 4x4 matrix (not sparse, since
// sigma is generally not as sparse as gamma).
//
// Index mapping for the 6 pairs:
//   0: (0,1)  1: (0,2)  2: (0,3)
//   3: (1,2)  4: (1,3)  5: (2,3)
//-----------------------------------------------
static complex_t sigmaMat[6][4][4];
static bool sigmaInitialized = false;

// Map (mu,nu) with mu<nu to linear index 0..5
static int sigma_index(int mu, int nu)
{
    // mu < nu assumed
    // (0,1)->0, (0,2)->1, (0,3)->2, (1,2)->3, (1,3)->4, (2,3)->5
    if (mu == 0) return nu - 1;
    if (mu == 1) return nu + 1;
    return 5; // (2,3)
}

void initialize_sigma_matrices()
{
    if (sigmaInitialized) return;

    // Compute full gamma matrices from sparse representation
    complex_t gamma_full[4][4][4];  // gamma_full[mu][s1][s2]
    for (int mu = 0; mu < 4; mu++) {
        for (int s1 = 0; s1 < 4; s1++)
            for (int s2 = 0; s2 < 4; s2++)
                gamma_full[mu][s1][s2] = complex_t(0.0, 0.0);

        for (int s = 0; s < 4; s++)
            gamma_full[mu][s][gammaIdx[mu][s]] = gammaVal[mu][s];
    }

    // sigma_{mu,nu} = (i/2) * [gamma_mu, gamma_nu]
    //               = (i/2) * (gamma_mu * gamma_nu - gamma_nu * gamma_mu)
    for (int mu = 0; mu < 4; mu++) {
        for (int nu = mu + 1; nu < 4; nu++) {
            int idx = sigma_index(mu, nu);

            for (int s1 = 0; s1 < 4; s1++) {
                for (int s2 = 0; s2 < 4; s2++) {
                    complex_t comm(0.0, 0.0);
                    for (int k = 0; k < 4; k++) {
                        comm += gamma_full[mu][s1][k] * gamma_full[nu][k][s2]
                              - gamma_full[nu][s1][k] * gamma_full[mu][k][s2];
                    }
                    sigmaMat[idx][s1][s2] = complex_t(0.0, 0.5) * comm;
                }
            }
        }
    }

    sigmaInitialized = true;
}

//-----------------------------------------------
// Compute the clover field for all sites.
//
// A(x) = c_sw * (i/4) * sum_{mu<nu} sigma_{mu,nu} ⊗ F_{mu,nu}(x)
//
// The result is assembled into two 6x6 Hermitian blocks
// per site (upper for spin 0,1 and lower for spin 2,3).
//
// The 6x6 block index is (s_local, c) where:
//   s_local = s for upper (s=0,1), s-2 for lower (s=2,3)
//   combined index = s_local * NC + c
//-----------------------------------------------
void compute_clover_field(const SU3matrix* gaugeField,
                          CloverField* cloverField,
                          int vol)
{
    if (!sigmaInitialized) initialize_sigma_matrices();

    const int diagPlane = clover_diag_plane();

    #pragma omp parallel for
    for (int site = 0; site < vol; site++) {
        // Zero the blocks
        for (int i = 0; i < CLOVER_BLOCK_SIZE; i++)
            for (int j = 0; j < CLOVER_BLOCK_SIZE; j++) {
                cloverField[site].upper.m[i][j] = complex_t(0.0, 0.0);
                cloverField[site].lower.m[i][j] = complex_t(0.0, 0.0);
            }

        // Sum over the 6 F_{mu,nu} components
        for (int mu = 0; mu < 4; mu++) {
            for (int nu = mu + 1; nu < 4; nu++) {
                int sigIdx = sigma_index(mu, nu);

                // Diagnostic: restrict to a single (mu,nu) plane
                if (diagPlane >= 0 && sigIdx != diagPlane) continue;

                // Get F_{mu,nu}(x) — a 3x3 color matrix
                SU3matrix Fmunu;
                compute_field_strength_tensor(gaugeField, site, mu, nu, Fmunu);

                // Accumulate c_sw * (i/4) * sigma_{mu,nu}_{s1,s2} * F_{mu,nu}_{a,b}
                // into the appropriate block

                complex_t prefactor = complex_t(0.0, 0.25) * c_sw;

                // Upper block: spin 0,1
                for (int s1 = 0; s1 < 2; s1++) {
                    for (int s2 = 0; s2 < 2; s2++) {
                        complex_t sig = prefactor * sigmaMat[sigIdx][s1][s2];
                        if (std::abs(sig) < 1e-15) continue;

                        for (int a = 0; a < NC; a++) {
                            for (int b = 0; b < NC; b++) {
                                int row = s1 * NC + a;
                                int col = s2 * NC + b;
                                cloverField[site].upper.m[row][col] += sig * Fmunu.m[a][b];
                            }
                        }
                    }
                }

                // Lower block: spin 2,3
                for (int s1 = 2; s1 < 4; s1++) {
                    for (int s2 = 2; s2 < 4; s2++) {
                        complex_t sig = prefactor * sigmaMat[sigIdx][s1][s2];
                        if (std::abs(sig) < 1e-15) continue;

                        for (int a = 0; a < NC; a++) {
                            for (int b = 0; b < NC; b++) {
                                int row = (s1 - 2) * NC + a;
                                int col = (s2 - 2) * NC + b;
                                cloverField[site].lower.m[row][col] += sig * Fmunu.m[a][b];
                            }
                        }
                    }
                }
            }
        }
    }
}

//-----------------------------------------------
// Apply the clover term at a single site:
//   result += A(x) * psi(x)
//
// A(x) is block-diagonal: upper acts on (spin 0,1),
// lower acts on (spin 2,3).
//-----------------------------------------------
void apply_clover_term(const CloverField* cloverField,
                       const ColorSpinor* psi,
                       ColorSpinor* result,
                       int site)
{
    const CloverField& A = cloverField[site];
    const ColorSpinor& p = psi[site];

    // Upper block: spin 0,1
    for (int s1 = 0; s1 < 2; s1++) {
        for (int a = 0; a < NC; a++) {
            int row = s1 * NC + a;
            complex_t sum(0.0, 0.0);
            for (int s2 = 0; s2 < 2; s2++) {
                for (int b = 0; b < NC; b++) {
                    int col = s2 * NC + b;
                    sum += A.upper.m[row][col] * p.v[s2][b];
                }
            }
            result[site].v[s1][a] += sum;
        }
    }

    // Lower block: spin 2,3
    for (int s1 = 2; s1 < 4; s1++) {
        for (int a = 0; a < NC; a++) {
            int row = (s1 - 2) * NC + a;
            complex_t sum(0.0, 0.0);
            for (int s2 = 2; s2 < 4; s2++) {
                for (int b = 0; b < NC; b++) {
                    int col = (s2 - 2) * NC + b;
                    sum += A.lower.m[row][col] * p.v[s2][b];
                }
            }
            result[site].v[s1][a] += sum;
        }
    }
}

//-----------------------------------------------
// In-place 6x6 Hermitian matrix inverse via
// pivoted LU decomposition.
//
// For the clover even-odd preconditioner we need
// (1 + A)^{-1}. Since the blocks are only 6x6,
// a dense solver is fine.
//-----------------------------------------------
void invert_clover_block(CloverBlock& block)
{
    constexpr int N = CLOVER_BLOCK_SIZE;
    complex_t A[N][N];
    complex_t inv[N][N];
    int pivot[N];

    // Copy
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = block.m[i][j];

    // Initialize inverse as identity
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            inv[i][j] = complex_t(0.0, 0.0);
        inv[i][i] = complex_t(1.0, 0.0);
        pivot[i] = i;
    }

    // LU decomposition with partial pivoting
    for (int k = 0; k < N; k++) {
        // Find pivot
        real_t maxVal = 0.0;
        int maxRow = k;
        for (int i = k; i < N; i++) {
            real_t val = std::abs(A[i][k]);
            if (val > maxVal) {
                maxVal = val;
                maxRow = i;
            }
        }

        // Swap rows
        if (maxRow != k) {
            for (int j = 0; j < N; j++) {
                std::swap(A[k][j], A[maxRow][j]);
                std::swap(inv[k][j], inv[maxRow][j]);
            }
        }

        // Eliminate below
        complex_t akk_inv = complex_t(1.0, 0.0) / A[k][k];
        for (int i = k + 1; i < N; i++) {
            complex_t factor = A[i][k] * akk_inv;
            for (int j = 0; j < N; j++) {
                A[i][j] -= factor * A[k][j];
                inv[i][j] -= factor * inv[k][j];
            }
        }
    }

    // Back-substitution
    for (int k = N - 1; k >= 0; k--) {
        complex_t akk_inv = complex_t(1.0, 0.0) / A[k][k];
        for (int j = 0; j < N; j++)
            inv[k][j] *= akk_inv;

        for (int i = 0; i < k; i++) {
            for (int j = 0; j < N; j++)
                inv[i][j] -= A[i][k] * inv[k][j];
        }
    }

    // Copy back
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            block.m[i][j] = inv[i][j];
}

//-----------------------------------------------
// Precompute (1 + A(x))^{-1} for all sites.
// Stores result in cloverFieldInv.
//-----------------------------------------------
void compute_clover_inverse(const CloverField* cloverField,
                            CloverField* cloverFieldInv,
                            int vol)
{
    #pragma omp parallel for
    for (int site = 0; site < vol; site++) {
        // Copy and add identity: (1 + A)
        constexpr int N = CLOVER_BLOCK_SIZE;

        // Upper block
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++)
                cloverFieldInv[site].upper.m[i][j] = cloverField[site].upper.m[i][j];
            cloverFieldInv[site].upper.m[i][i] += complex_t(1.0, 0.0);
        }
        invert_clover_block(cloverFieldInv[site].upper);

        // Lower block
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++)
                cloverFieldInv[site].lower.m[i][j] = cloverField[site].lower.m[i][j];
            cloverFieldInv[site].lower.m[i][i] += complex_t(1.0, 0.0);
        }
        invert_clover_block(cloverFieldInv[site].lower);
    }
}

//-----------------------------------------------
// Public accessor for sigma matrices and index map
// (needed by clover_fermion_force.cpp)
//-----------------------------------------------
const complex_t* get_sigma_matrix(int mu, int nu)
{
    if (!sigmaInitialized) initialize_sigma_matrices();
    int mA = (mu < nu) ? mu : nu;
    int mB = (mu < nu) ? nu : mu;
    return &sigmaMat[sigma_index(mA, mB)][0][0];
}

// Base pointer to the full contiguous sigmaMat[6][4][4] block
// (pair-major: idx*16 + s1*4 + s2). Used to upload sigma to the GPU.
const complex_t* get_sigma_base()
{
    if (!sigmaInitialized) initialize_sigma_matrices();
    return &sigmaMat[0][0][0];
}

int get_sigma_pair_index(int mu, int nu)
{
    int mA = (mu < nu) ? mu : nu;
    int mB = (mu < nu) ? nu : mu;
    return sigma_index(mA, mB);
}
