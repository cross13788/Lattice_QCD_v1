//-----------------------------------------------
// SU(3) Lie algebra operations for HMC
//
// The conjugate momenta in HMC live in the Lie
// algebra su(3): traceless anti-Hermitian 3x3
// matrices. We store them as SU3matrix structs.
//
// Key operations:
//   1. Project any matrix to traceless anti-Hermitian
//   2. Compute exp(A) * U for gauge field updates
//   3. Trace inner product Tr(A * B)
//-----------------------------------------------
#include "constants_module.hpp"
#include "su3_module.hpp"
#include "interfaces_module.hpp"
#include <cmath>

//-----------------------------------------------
// Project a matrix to traceless anti-Hermitian:
//   A_TA = (A - A^dag) / 2 - Tr(A - A^dag) / (2*NC) * I
//
// This is the projection onto the Lie algebra su(3).
//-----------------------------------------------
void su3_traceless_antiherm(const SU3matrix& A, SU3matrix& result)
{
//-----------------------------------------------
//   Anti-Hermitian part: (A - A^dag) / 2
//-----------------------------------------------
    for (int i = 0; i < NC; i++) {
        for (int j = 0; j < NC; j++) {
            result.m[i][j] = 0.5 * (A.m[i][j] - std::conj(A.m[j][i]));
        }
    }

//-----------------------------------------------
//   Subtract trace / NC to make traceless
//-----------------------------------------------
    complex_t tr(0.0, 0.0);
    for (int i = 0; i < NC; i++)
        tr += result.m[i][i];
    tr /= (real_t)NC;

    for (int i = 0; i < NC; i++)
        result.m[i][i] -= tr;
}

//-----------------------------------------------
// Matrix exponential via Taylor series:
//   exp(A) = I + A + A^2/2! + A^3/3! + ...
//
// For traceless anti-Hermitian A with small norm
// (typical MD step), 12 terms gives machine precision.
// The result is automatically in SU(3).
//-----------------------------------------------
void su3_exp(const SU3matrix& A, SU3matrix& result)
{
//-----------------------------------------------
//   Taylor expansion: exp(A) ≈ sum_{n=0}^{N} A^n / n!
//   Computed via Horner's method for efficiency.
//-----------------------------------------------
    int nTerms = 12;

    // Start with result = I + A/N * (I + A/(N-1) * (...))
    // Horner: result = I; for n = N..1: result = I + (A/n) * result
    su3_identity(result);

    for (int n = nTerms; n >= 1; n--) {
        // result = A * result / n
        SU3matrix tmp;
        su3_multiply(A, result, tmp);
        su3_scale(complex_t(1.0 / n, 0.0), tmp, result);

        // result = I + result
        for (int i = 0; i < NC; i++)
            result.m[i][i] += complex_t(1.0, 0.0);
    }

//-----------------------------------------------
//   Reunitarize to ensure exact SU(3)
//-----------------------------------------------
    su3_reunitarize(result);
}

//-----------------------------------------------
// Update gauge link: U_new = exp(epsilon * H) * U
//
// H is a traceless anti-Hermitian momentum.
// epsilon is the MD step size.
//-----------------------------------------------
void su3_exp_update(real_t epsilon, const SU3matrix& H,
                    SU3matrix& U)
{
//-----------------------------------------------
//   Compute epsilon * H
//-----------------------------------------------
    SU3matrix epsH;
    su3_scale(complex_t(epsilon, 0.0), H, epsH);

//-----------------------------------------------
//   Compute exp(epsilon * H)
//-----------------------------------------------
    SU3matrix expH;
    su3_exp(epsH, expH);

//-----------------------------------------------
//   U = exp(epsilon * H) * U
//-----------------------------------------------
    SU3matrix Unew;
    su3_multiply(expH, U, Unew);
    su3_copy(Unew, U);
}

//-----------------------------------------------
// Trace inner product for algebra elements:
//   (A, B) = -2 * Re Tr(A * B)
//
// For anti-Hermitian A, B: this gives a real,
// positive-definite inner product.
//-----------------------------------------------
real_t su3_algebra_norm2(const SU3matrix& A)
{
    real_t sum = 0.0;
    for (int i = 0; i < NC; i++)
        for (int j = 0; j < NC; j++)
            sum += std::norm(A.m[i][j]);
    return sum;
}
