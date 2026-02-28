//-----------------------------------------------
// SU(3) matrix operations
//
// Implements all SU(3) algebra needed for lattice
// gauge theory: multiplication, adjoint, trace,
// reunitarization, random generation, and
// Cabibbo-Marinari heat bath with Kennedy-Pendleton.
//-----------------------------------------------
#include "su3_module.hpp"
#include "input_module.hpp"
#include <cmath>
#include <cstdio>
#include <random>
#ifdef _OPENMP
#include <omp.h>
#endif

// Thread-local random number generator (extern declared in su3_module.hpp)
thread_local std::mt19937_64 rng_local;
static thread_local bool rngSeeded = false;

void ensure_rng_seeded()
{
    if (!rngSeeded) {
        int threadId = 0;
#ifdef _OPENMP
        threadId = omp_get_thread_num();
#endif
        // Use input seed + thread ID for reproducibility
        rng_local.seed(static_cast<uint64_t>(randomSeed) + threadId * 137);
        rngSeeded = true;
    }
}

static real_t random_uniform()
{
    ensure_rng_seeded();
    static thread_local std::uniform_real_distribution<real_t> dist(0.0, 1.0);
    return dist(rng_local);
}

static real_t random_normal()
{
    ensure_rng_seeded();
    static thread_local std::normal_distribution<real_t> dist(0.0, 1.0);
    return dist(rng_local);
}

//-----------------------------------------------
// Basic operations
//-----------------------------------------------

void su3_identity(SU3matrix& result)
{
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++)
            result.m[a][b] = (a == b) ? complex_t(1.0, 0.0) : complex_t(0.0, 0.0);
}

void su3_zero(SU3matrix& result)
{
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++)
            result.m[a][b] = complex_t(0.0, 0.0);
}

void su3_copy(const SU3matrix& A, SU3matrix& result)
{
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++)
            result.m[a][b] = A.m[a][b];
}

void su3_multiply(const SU3matrix& A, const SU3matrix& B, SU3matrix& result)
{
//-----------------------------------------------
//   result = A * B  (3x3 complex matrix multiply)
//-----------------------------------------------
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            result.m[a][b] = A.m[a][0] * B.m[0][b]
                           + A.m[a][1] * B.m[1][b]
                           + A.m[a][2] * B.m[2][b];
        }
    }
}

void su3_dagger(const SU3matrix& A, SU3matrix& result)
{
//-----------------------------------------------
//   result = A^dagger (conjugate transpose)
//-----------------------------------------------
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++)
            result.m[a][b] = std::conj(A.m[b][a]);
}

complex_t su3_trace(const SU3matrix& A)
{
    return A.m[0][0] + A.m[1][1] + A.m[2][2];
}

real_t su3_retrace(const SU3matrix& A)
{
    return A.m[0][0].real() + A.m[1][1].real() + A.m[2][2].real();
}

void su3_add(const SU3matrix& A, const SU3matrix& B, SU3matrix& result)
{
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++)
            result.m[a][b] = A.m[a][b] + B.m[a][b];
}

void su3_subtract(const SU3matrix& A, const SU3matrix& B, SU3matrix& result)
{
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++)
            result.m[a][b] = A.m[a][b] - B.m[a][b];
}

void su3_scale(complex_t c, const SU3matrix& A, SU3matrix& result)
{
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++)
            result.m[a][b] = c * A.m[a][b];
}

void su3_accumulate(const SU3matrix& A, SU3matrix& result)
{
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++)
            result.m[a][b] += A.m[a][b];
}

complex_t su3_determinant(const SU3matrix& A)
{
//-----------------------------------------------
//   3x3 determinant via cofactor expansion
//-----------------------------------------------
    return A.m[0][0] * (A.m[1][1] * A.m[2][2] - A.m[1][2] * A.m[2][1])
         - A.m[0][1] * (A.m[1][0] * A.m[2][2] - A.m[1][2] * A.m[2][0])
         + A.m[0][2] * (A.m[1][0] * A.m[2][1] - A.m[1][1] * A.m[2][0]);
}

//-----------------------------------------------
// Reunitarization: modified Gram-Schmidt
//
// 1. Normalize row 0
// 2. Orthogonalize row 1 against row 0, normalize
// 3. Row 2 = conj(row 0 x row 1) to ensure det = 1
//-----------------------------------------------
void su3_reunitarize(SU3matrix& U)
{
    // Normalize row 0
    real_t norm0 = 0.0;
    for (int b = 0; b < 3; b++)
        norm0 += std::norm(U.m[0][b]);
    norm0 = 1.0 / std::sqrt(norm0);
    for (int b = 0; b < 3; b++)
        U.m[0][b] *= norm0;

    // Orthogonalize row 1 against row 0
    complex_t dot01(0.0, 0.0);
    for (int b = 0; b < 3; b++)
        dot01 += std::conj(U.m[0][b]) * U.m[1][b];
    for (int b = 0; b < 3; b++)
        U.m[1][b] -= dot01 * U.m[0][b];

    // Normalize row 1
    real_t norm1 = 0.0;
    for (int b = 0; b < 3; b++)
        norm1 += std::norm(U.m[1][b]);
    norm1 = 1.0 / std::sqrt(norm1);
    for (int b = 0; b < 3; b++)
        U.m[1][b] *= norm1;

    // Row 2 = conj(row 0 x row 1) to enforce det = +1
    U.m[2][0] = std::conj(U.m[0][1] * U.m[1][2] - U.m[0][2] * U.m[1][1]);
    U.m[2][1] = std::conj(U.m[0][2] * U.m[1][0] - U.m[0][0] * U.m[1][2]);
    U.m[2][2] = std::conj(U.m[0][0] * U.m[1][1] - U.m[0][1] * U.m[1][0]);
}

//-----------------------------------------------
// Random SU(3) near identity for Metropolis
//
// Generate X = exp(i * epsilon * H) where H is a
// random traceless hermitian matrix. For small
// epsilon this is near the identity.
//-----------------------------------------------
void su3_random_near_identity(SU3matrix& result, real_t epsilon)
{
    // Generate random traceless hermitian 3x3
    // The 8 Gell-Mann generators span su(3)
    real_t h[8];
    for (int i = 0; i < 8; i++)
        h[i] = epsilon * random_normal();

    // Build the matrix i*H in the Gell-Mann basis
    // Then approximate exp(i*epsilon*H) ~ 1 + iH - H^2/2
    // For better accuracy, embed random SU(2) in 3 subgroups

    // Simpler approach: compose 3 random SU(2) subgroup rotations
    su3_identity(result);

    for (int sub = 0; sub < 3; sub++) {
        // Random SU(2) near identity
        real_t r[4];
        r[0] = 1.0;
        for (int i = 1; i < 4; i++)
            r[i] = epsilon * random_normal();

        // Normalize to unit quaternion
        real_t norm = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2] + r[3]*r[3]);
        for (int i = 0; i < 4; i++)
            r[i] /= norm;

        // Convert quaternion to 2x2 matrix:
        //   a = ( r0 + i*r3,  r2 + i*r1 )
        //       (-r2 + i*r1,  r0 - i*r3 )
        complex_t a[2][2];
        a[0][0] = complex_t(r[0], r[3]);
        a[0][1] = complex_t(r[2], r[1]);
        a[1][0] = complex_t(-r[2], r[1]);
        a[1][1] = complex_t(r[0], -r[3]);

        // Embed in SU(3)
        SU3matrix R;
        su3_identity(R);

        int idx[3][2] = {{0,1}, {0,2}, {1,2}};
        int i0 = idx[sub][0], i1 = idx[sub][1];
        R.m[i0][i0] = a[0][0];
        R.m[i0][i1] = a[0][1];
        R.m[i1][i0] = a[1][0];
        R.m[i1][i1] = a[1][1];

        // result = R * result
        SU3matrix tmp;
        su3_multiply(R, result, tmp);
        su3_copy(tmp, result);
    }

    // Also multiply by the inverse with 50% probability (for ergodicity)
    if (random_uniform() < 0.5) {
        SU3matrix tmp;
        su3_dagger(result, tmp);
        su3_copy(tmp, result);
    }
}

//-----------------------------------------------
// Random SU(3) uniformly distributed
//
// Generate by composing many random SU(2) subgroup
// elements until uniformly distributed.
//-----------------------------------------------
void su3_random_uniform(SU3matrix& result)
{
    su3_identity(result);
    // 20 random SU(2) subgroup hits is sufficient for uniformity
    for (int iter = 0; iter < 20; iter++) {
        SU3matrix R;
        su3_random_near_identity(R, 2.0);
        SU3matrix tmp;
        su3_multiply(R, result, tmp);
        su3_copy(tmp, result);
    }
    su3_reunitarize(result);
}

//-----------------------------------------------
// SU(2) heat bath: Kennedy-Pendleton algorithm
//
// Generate SU(2) element a distributed as
//   P(a) ~ exp(beta_eff * Re Tr(a * W))
// where W is a 2x2 complex matrix (the submatrix
// of the staple). Reference: KEK_LQCD.pdf Sec 5.4
//-----------------------------------------------
void su2_heat_bath(real_t beta_eff, const complex_t W[2][2], complex_t a[2][2])
{
    // Decompose W into Pauli components:
    //   W = w0*I + i*(w1*sigma1 + w2*sigma2 + w3*sigma3)
    // where the w_i are real numbers extracted from W.
    // Then k = sqrt(w0^2 + w1^2 + w2^2 + w3^2) and
    // V = (w0, w1, w2, w3)/k is the SU(2) projection of W.

    real_t w0 = 0.5 * (W[0][0].real() + W[1][1].real());
    real_t w3 = 0.5 * (W[0][0].imag() - W[1][1].imag());
    real_t w1 = 0.5 * (W[0][1].imag() + W[1][0].imag());
    real_t w2 = 0.5 * (W[0][1].real() - W[1][0].real());

    real_t k = std::sqrt(w0*w0 + w1*w1 + w2*w2 + w3*w3);

    if (k < SMALL_VALUE) {
        // Staple is nearly zero, generate random SU(2)
        real_t r[4];
        for (int i = 0; i < 4; i++)
            r[i] = random_normal();
        real_t norm = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2] + r[3]*r[3]);
        for (int i = 0; i < 4; i++)
            r[i] /= norm;
        a[0][0] = complex_t(r[0], r[3]);
        a[0][1] = complex_t(r[2], r[1]);
        a[1][0] = complex_t(-r[2], r[1]);
        a[1][1] = complex_t(r[0], -r[3]);
        return;
    }

    // V^dagger in quaternion form: (w0, -w1, -w2, -w3)/k
    // In matrix form:
    //   Vdag = (1/k) * [[ w0 - i*w3, -w2 - i*w1],
    //                    [ w2 - i*w1,  w0 + i*w3]]
    complex_t Vdag[2][2];
    Vdag[0][0] = complex_t( w0, -w3) / k;
    Vdag[0][1] = complex_t(-w2, -w1) / k;
    Vdag[1][0] = complex_t( w2, -w1) / k;
    Vdag[1][1] = complex_t( w0,  w3) / k;

    // Kennedy-Pendleton algorithm for SU(2) heat bath
    // Generate a0 in [-1,1] with distribution:
    //   P(a0) ~ sqrt(1 - a0^2) * exp(2 * alpha * a0)
    // where alpha = beta_eff * k
    //
    // The algorithm generates x = 1 - a0 in [0,2] with:
    //   P(x) ~ sqrt(x*(2-x)) * exp(-2*alpha*x)
    // Reference: Kennedy & Pendleton, Phys.Lett.B 156 (1985) 393
    real_t alpha = 2.0 * beta_eff * k;
    real_t a0;

    while (true) {
        real_t r1 = random_uniform();
        real_t r2 = random_uniform();
        real_t r3 = random_uniform();
        real_t r4 = random_uniform();

        // Protect against log(0)
        if (r1 < 1.0e-300) r1 = 1.0e-300;
        if (r3 < 1.0e-300) r3 = 1.0e-300;

        real_t lambda2 = -std::log(r1) / alpha;
        real_t cosAngle = std::cos(2.0 * PI * r2);
        real_t mu2 = -std::log(r3) / alpha;

        real_t x = lambda2 + mu2 * cosAngle * cosAngle;

        // Reject if x > 2 (since a0 = 1-x must be >= -1)
        if (x > 2.0) continue;

        // Accept with probability sqrt(1 - x/2)
        if (r4 * r4 <= 1.0 - 0.5 * x) {
            a0 = 1.0 - x;
            break;
        }
    }

    // Generate random unit vector for the remaining 3 components
    real_t rr = std::sqrt(1.0 - a0 * a0);

    // Random point on unit sphere
    real_t cosTheta = 2.0 * random_uniform() - 1.0;
    real_t sinTheta = std::sqrt(1.0 - cosTheta * cosTheta);
    real_t phi = 2.0 * PI * random_uniform();

    real_t a1 = rr * sinTheta * std::cos(phi);
    real_t a2 = rr * sinTheta * std::sin(phi);
    real_t a3 = rr * cosTheta;

    // The sampled SU(2) element (relative to V) is:
    //   aRel = ( a0 + i*a3,  a2 + i*a1 )
    //          (-a2 + i*a1,  a0 - i*a3 )
    // The actual update is a = aRel * V^dagger
    complex_t aRel[2][2];
    aRel[0][0] = complex_t(a0, a3);
    aRel[0][1] = complex_t(a2, a1);
    aRel[1][0] = complex_t(-a2, a1);
    aRel[1][1] = complex_t(a0, -a3);

    // a = aRel * Vdag
    a[0][0] = aRel[0][0] * Vdag[0][0] + aRel[0][1] * Vdag[1][0];
    a[0][1] = aRel[0][0] * Vdag[0][1] + aRel[0][1] * Vdag[1][1];
    a[1][0] = aRel[1][0] * Vdag[0][0] + aRel[1][1] * Vdag[1][0];
    a[1][1] = aRel[1][0] * Vdag[0][1] + aRel[1][1] * Vdag[1][1];
}

//-----------------------------------------------
// Heat bath update of one SU(2) subgroup
//
// Given link U and staple S, the local action is
//   S_local = -(beta/NC) * Re Tr(U * S)
// We update the SU(2) subgroup of U using heat bath.
//-----------------------------------------------
void su3_heat_bath_subgroup(SU3matrix& U, const SU3matrix& staple,
                            real_t beta, int subgroup)
{
    // Form W = U * staple
    SU3matrix W;
    su3_multiply(U, staple, W);

    // Extract 2x2 submatrix
    int idx[3][2] = {{0,1}, {0,2}, {1,2}};
    int i0 = idx[subgroup][0], i1 = idx[subgroup][1];

    complex_t Wsub[2][2];
    Wsub[0][0] = W.m[i0][i0];
    Wsub[0][1] = W.m[i0][i1];
    Wsub[1][0] = W.m[i1][i0];
    Wsub[1][1] = W.m[i1][i1];

    // Heat bath to get SU(2) element a
    complex_t a[2][2];
    real_t beta_eff = beta / static_cast<real_t>(NC);
    su2_heat_bath(beta_eff, Wsub, a);

    // Embed a into SU(3): R = identity with a in the subgroup block
    SU3matrix R;
    su3_identity(R);
    R.m[i0][i0] = a[0][0];
    R.m[i0][i1] = a[0][1];
    R.m[i1][i0] = a[1][0];
    R.m[i1][i1] = a[1][1];

    // Update: U -> R * U
    SU3matrix tmp;
    su3_multiply(R, U, tmp);
    su3_copy(tmp, U);
}

//-----------------------------------------------
// Full Cabibbo-Marinari heat bath update
// Cycles through all 3 SU(2) subgroups
//-----------------------------------------------
void su3_heat_bath_update(SU3matrix& U, const SU3matrix& staple, real_t beta)
{
    for (int sub = 0; sub < 3; sub++) {
        su3_heat_bath_subgroup(U, staple, beta, sub);
    }
    // Reunitarize to prevent numerical drift
    su3_reunitarize(U);
}

//-----------------------------------------------
// Overrelaxation (microcanonical) update
//
// U -> V^dag * U^dag * V^dag
// where V is constructed from the staple sum.
// This preserves the action but decorrelates.
//-----------------------------------------------
void su3_overrelaxation_update(SU3matrix& U, const SU3matrix& staple)
{
    // For each SU(2) subgroup, do microcanonical reflection.
    // The idea: compute A = U * staple subblock, project to SU(2),
    // then reflect: U_new such that A_new = A_old^{-1} (in the subgroup).
    // This preserves Re Tr(A) and hence the action.
    for (int sub = 0; sub < 3; sub++) {
        // Compute W = U * staple
        SU3matrix W;
        su3_multiply(U, staple, W);

        int idx[3][2] = {{0,1}, {0,2}, {1,2}};
        int i0 = idx[sub][0], i1 = idx[sub][1];

        // Extract 2x2 submatrix of W
        complex_t Wsub[2][2];
        Wsub[0][0] = W.m[i0][i0];
        Wsub[0][1] = W.m[i0][i1];
        Wsub[1][0] = W.m[i1][i0];
        Wsub[1][1] = W.m[i1][i1];

        // Pauli decomposition: W_sub = k * V where V in SU(2)
        real_t w0 = 0.5 * (Wsub[0][0].real() + Wsub[1][1].real());
        real_t w3 = 0.5 * (Wsub[0][0].imag() - Wsub[1][1].imag());
        real_t w1 = 0.5 * (Wsub[0][1].imag() + Wsub[1][0].imag());
        real_t w2 = 0.5 * (Wsub[0][1].real() - Wsub[1][0].real());
        real_t k = std::sqrt(w0*w0 + w1*w1 + w2*w2 + w3*w3);
        if (k < SMALL_VALUE) continue;

        // V in quaternion form is (w0, w1, w2, w3)/k
        // V^dagger = (w0, -w1, -w2, -w3)/k
        // V^{-2} = V^dagger * V^dagger = conjugate(V)^2
        // The overrelaxation update: R_sub = V^{-2} (in the subgroup)
        // since the current subgroup element of U*staple is approximately V*k,
        // and R * (U*staple)_sub = V^{-2} * k*V = k*V^{-1} = k*V^dag
        // so Re Tr stays the same.

        // V^{-2} in quaternion: if V = (a0,a1,a2,a3) then
        // V^{-1} = V^dag = (a0,-a1,-a2,-a3)
        // V^{-2} = V^dag * V^dag: quaternion product
        real_t v0 = w0/k, v1 = -w1/k, v2 = -w2/k, v3 = -w3/k;
        // (v0,v1,v2,v3) * (v0,v1,v2,v3) quaternion multiply:
        real_t r0 = v0*v0 - v1*v1 - v2*v2 - v3*v3;
        real_t r1 = 2.0*v0*v1;
        real_t r2 = 2.0*v0*v2;
        real_t r3 = 2.0*v0*v3;

        // Convert to 2x2 matrix
        complex_t Rsub[2][2];
        Rsub[0][0] = complex_t(r0, r3);
        Rsub[0][1] = complex_t(r2, r1);
        Rsub[1][0] = complex_t(-r2, r1);
        Rsub[1][1] = complex_t(r0, -r3);

        // Embed in SU(3)
        SU3matrix R;
        su3_identity(R);
        R.m[i0][i0] = Rsub[0][0];
        R.m[i0][i1] = Rsub[0][1];
        R.m[i1][i0] = Rsub[1][0];
        R.m[i1][i1] = Rsub[1][1];

        // U -> R * U
        SU3matrix tmp;
        su3_multiply(R, U, tmp);
        su3_copy(tmp, U);
    }
    su3_reunitarize(U);
}

//-----------------------------------------------
// Print matrix for debugging
//-----------------------------------------------
void su3_print(const SU3matrix& A)
{
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            printf("  (%+.6f, %+.6f)", A.m[a][b].real(), A.m[a][b].imag());
        }
        printf("\n");
    }
}
