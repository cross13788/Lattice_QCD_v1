#pragma once
//-----------------------------------------------
// Device-inline SU(3) matrix operations
//
// All operations use gpu_complex_t and are marked
// __device__ __forceinline__ for use inside CUDA
// kernels.
//-----------------------------------------------
#include "gpu_common.cuh"

//-----------------------------------------------
// SU(3) matrix struct for device code
// Same memory layout as CPU SU3matrix
//-----------------------------------------------
struct SU3matrixDev {
    gpu_complex_t m[3][3];
};

//-----------------------------------------------
// Basic operations
//-----------------------------------------------

__device__ __forceinline__
void dev_su3_identity(SU3matrixDev& result)
{
    #pragma unroll
    for (int a = 0; a < 3; a++)
        #pragma unroll
        for (int b = 0; b < 3; b++)
            result.m[a][b] = (a == b) ? gpu_complex_t(1.0, 0.0) : gpu_complex_t(0.0, 0.0);
}

__device__ __forceinline__
void dev_su3_zero(SU3matrixDev& result)
{
    #pragma unroll
    for (int a = 0; a < 3; a++)
        #pragma unroll
        for (int b = 0; b < 3; b++)
            result.m[a][b] = gpu_complex_t(0.0, 0.0);
}

__device__ __forceinline__
void dev_su3_copy(const SU3matrixDev& A, SU3matrixDev& result)
{
    #pragma unroll
    for (int a = 0; a < 3; a++)
        #pragma unroll
        for (int b = 0; b < 3; b++)
            result.m[a][b] = A.m[a][b];
}

__device__ __forceinline__
void dev_su3_multiply(const SU3matrixDev& A, const SU3matrixDev& B, SU3matrixDev& result)
{
    #pragma unroll
    for (int a = 0; a < 3; a++) {
        #pragma unroll
        for (int b = 0; b < 3; b++) {
            result.m[a][b] = A.m[a][0] * B.m[0][b]
                           + A.m[a][1] * B.m[1][b]
                           + A.m[a][2] * B.m[2][b];
        }
    }
}

__device__ __forceinline__
void dev_su3_dagger(const SU3matrixDev& A, SU3matrixDev& result)
{
    #pragma unroll
    for (int a = 0; a < 3; a++)
        #pragma unroll
        for (int b = 0; b < 3; b++)
            result.m[a][b] = gpu_conj(A.m[b][a]);
}

__device__ __forceinline__
gpu_complex_t dev_su3_trace(const SU3matrixDev& A)
{
    return A.m[0][0] + A.m[1][1] + A.m[2][2];
}

__device__ __forceinline__
gpu_real_t dev_su3_retrace(const SU3matrixDev& A)
{
    return A.m[0][0].real() + A.m[1][1].real() + A.m[2][2].real();
}

__device__ __forceinline__
void dev_su3_add(const SU3matrixDev& A, const SU3matrixDev& B, SU3matrixDev& result)
{
    #pragma unroll
    for (int a = 0; a < 3; a++)
        #pragma unroll
        for (int b = 0; b < 3; b++)
            result.m[a][b] = A.m[a][b] + B.m[a][b];
}

__device__ __forceinline__
void dev_su3_subtract(const SU3matrixDev& A, const SU3matrixDev& B, SU3matrixDev& result)
{
    #pragma unroll
    for (int a = 0; a < 3; a++)
        #pragma unroll
        for (int b = 0; b < 3; b++)
            result.m[a][b] = A.m[a][b] - B.m[a][b];
}

__device__ __forceinline__
void dev_su3_scale(gpu_complex_t c, const SU3matrixDev& A, SU3matrixDev& result)
{
    #pragma unroll
    for (int a = 0; a < 3; a++)
        #pragma unroll
        for (int b = 0; b < 3; b++)
            result.m[a][b] = c * A.m[a][b];
}

__device__ __forceinline__
void dev_su3_accumulate(const SU3matrixDev& A, SU3matrixDev& result)
{
    #pragma unroll
    for (int a = 0; a < 3; a++)
        #pragma unroll
        for (int b = 0; b < 3; b++)
            result.m[a][b] += A.m[a][b];
}

//-----------------------------------------------
// Reunitarization: modified Gram-Schmidt
//-----------------------------------------------
__device__ __forceinline__
void dev_su3_reunitarize(SU3matrixDev& U)
{
    // Normalize row 0
    gpu_real_t norm0 = 0.0;
    #pragma unroll
    for (int b = 0; b < 3; b++)
        norm0 += gpu_norm(U.m[0][b]);
    norm0 = rsqrt(norm0);
    #pragma unroll
    for (int b = 0; b < 3; b++)
        U.m[0][b] *= norm0;

    // Orthogonalize row 1 against row 0
    gpu_complex_t dot01(0.0, 0.0);
    #pragma unroll
    for (int b = 0; b < 3; b++)
        dot01 += gpu_conj(U.m[0][b]) * U.m[1][b];
    #pragma unroll
    for (int b = 0; b < 3; b++)
        U.m[1][b] -= dot01 * U.m[0][b];

    // Normalize row 1
    gpu_real_t norm1 = 0.0;
    #pragma unroll
    for (int b = 0; b < 3; b++)
        norm1 += gpu_norm(U.m[1][b]);
    norm1 = rsqrt(norm1);
    #pragma unroll
    for (int b = 0; b < 3; b++)
        U.m[1][b] *= norm1;

    // Row 2 = conj(row 0 x row 1)
    U.m[2][0] = gpu_conj(U.m[0][1] * U.m[1][2] - U.m[0][2] * U.m[1][1]);
    U.m[2][1] = gpu_conj(U.m[0][2] * U.m[1][0] - U.m[0][0] * U.m[1][2]);
    U.m[2][2] = gpu_conj(U.m[0][0] * U.m[1][1] - U.m[0][1] * U.m[1][0]);
}

//-----------------------------------------------
// Traceless anti-Hermitian projection
//-----------------------------------------------
__device__ __forceinline__
void dev_su3_traceless_antiherm(const SU3matrixDev& A, SU3matrixDev& result)
{
    #pragma unroll
    for (int i = 0; i < GPU_NC; i++)
        #pragma unroll
        for (int j = 0; j < GPU_NC; j++)
            result.m[i][j] = gpu_complex_t(0.5) * (A.m[i][j] - gpu_conj(A.m[j][i]));

    gpu_complex_t tr(0.0, 0.0);
    #pragma unroll
    for (int i = 0; i < GPU_NC; i++)
        tr += result.m[i][i];
    tr /= (gpu_real_t)GPU_NC;

    #pragma unroll
    for (int i = 0; i < GPU_NC; i++)
        result.m[i][i] -= tr;
}

//-----------------------------------------------
// Matrix exponential via Taylor series (Horner)
// 12 terms gives machine precision for typical
// MD step sizes.
//-----------------------------------------------
__device__ __forceinline__
void dev_su3_exp(const SU3matrixDev& A, SU3matrixDev& result)
{
    const int nTerms = 12;
    dev_su3_identity(result);

    for (int n = nTerms; n >= 1; n--) {
        SU3matrixDev tmp;
        dev_su3_multiply(A, result, tmp);
        dev_su3_scale(gpu_complex_t(1.0 / n, 0.0), tmp, result);
        #pragma unroll
        for (int i = 0; i < GPU_NC; i++)
            result.m[i][i] += gpu_complex_t(1.0, 0.0);
    }
    dev_su3_reunitarize(result);
}

//-----------------------------------------------
// Gauge link update: U = exp(eps * H) * U
//-----------------------------------------------
__device__ __forceinline__
void dev_su3_exp_update(gpu_real_t epsilon, const SU3matrixDev& H, SU3matrixDev& U)
{
    SU3matrixDev epsH;
    dev_su3_scale(gpu_complex_t(epsilon, 0.0), H, epsH);

    SU3matrixDev expH;
    dev_su3_exp(epsH, expH);

    SU3matrixDev Unew;
    dev_su3_multiply(expH, U, Unew);
    dev_su3_copy(Unew, U);
}

//-----------------------------------------------
// Algebra norm: sum |A_ij|^2
//-----------------------------------------------
__device__ __forceinline__
gpu_real_t dev_su3_algebra_norm2(const SU3matrixDev& A)
{
    gpu_real_t sum = 0.0;
    #pragma unroll
    for (int i = 0; i < GPU_NC; i++)
        #pragma unroll
        for (int j = 0; j < GPU_NC; j++)
            sum += gpu_norm(A.m[i][j]);
    return sum;
}

//-----------------------------------------------
// Staple computation (device-inline)
// R(x,mu) = sum_{nu!=mu} [positive + negative staple]
//-----------------------------------------------
__device__ __forceinline__
void dev_compute_staple(const SU3matrixDev* gaugeField,
                        const int* neighborPlus,
                        const int* neighborMinus,
                        int site, int mu,
                        SU3matrixDev& staple)
{
    SU3matrixDev tmp1, tmp2, tmp3;
    int sitePlusMu = neighborPlus[site * 4 + mu];

    dev_su3_zero(staple);

    #pragma unroll
    for (int nu = 0; nu < 4; nu++) {
        if (nu == mu) continue;

        int sitePlusNu     = neighborPlus[site * 4 + nu];
        int siteMinusNu    = neighborMinus[site * 4 + nu];
        int sitePlusMuMinusNu = neighborMinus[sitePlusMu * 4 + nu];

        // Positive staple: U(x+mu,nu) * U^dag(x+nu,mu) * U^dag(x,nu)
        dev_su3_dagger(gaugeField[sitePlusNu * 4 + mu], tmp1);
        dev_su3_multiply(gaugeField[sitePlusMu * 4 + nu], tmp1, tmp2);
        dev_su3_dagger(gaugeField[site * 4 + nu], tmp1);
        dev_su3_multiply(tmp2, tmp1, tmp3);
        dev_su3_accumulate(tmp3, staple);

        // Negative staple: U^dag(x+mu-nu,nu) * U^dag(x-nu,mu) * U(x-nu,nu)
        dev_su3_dagger(gaugeField[sitePlusMuMinusNu * 4 + nu], tmp1);
        dev_su3_dagger(gaugeField[siteMinusNu * 4 + mu], tmp2);
        dev_su3_multiply(tmp1, tmp2, tmp3);
        dev_su3_multiply(tmp3, gaugeField[siteMinusNu * 4 + nu], tmp1);
        dev_su3_accumulate(tmp1, staple);
    }
}
