//-----------------------------------------------
// BiCGstab solver on GPU
//
// Solves D_W * x = b directly (not normal equations).
// All vectors stay on device.
//-----------------------------------------------
#include "gpu_state.cuh"
#include "colorspinor_device.cuh"
#include <cstdio>
#include <cmath>

// Forward declarations
void gpu_apply_dirac(const ColorSpinorDev* psi, ColorSpinorDev* result,
                     gpu_real_t kappa, gpu_real_t wilsonR,
                     bool useClover, int vol);
void gpu_field_zero(ColorSpinorDev* field, int vol);
void gpu_field_copy(const ColorSpinorDev* src, ColorSpinorDev* result, int vol);
void gpu_field_accumulate(gpu_complex_t alpha, const ColorSpinorDev* x,
                          ColorSpinorDev* result, int vol);
gpu_real_t gpu_field_norm2(const ColorSpinorDev* x, int vol);
gpu_complex_t gpu_field_dot(const ColorSpinorDev* x, const ColorSpinorDev* y, int vol);

// BiCGstab p-update kernel: p = r + beta * (p - omega * v)
__global__
void kernel_bicgstab_update_p(const ColorSpinorDev* r,
                              ColorSpinorDev* p,
                              const ColorSpinorDev* v,
                              gpu_complex_t beta,
                              gpu_complex_t omega,
                              int vol)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vol) return;

    #pragma unroll
    for (int s = 0; s < GPU_ND; s++)
        #pragma unroll
        for (int c = 0; c < GPU_NC; c++)
            p[idx].v[s][c] = r[idx].v[s][c] +
                beta * (p[idx].v[s][c] - omega * v[idx].v[s][c]);
}

int gpu_bicgstab_solver(const ColorSpinorDev* b, ColorSpinorDev* x,
                        gpu_real_t kappa, gpu_real_t wilsonR,
                        bool useClover, int cgMaxIter, gpu_real_t cgTolerance,
                        bool verbose, int vol)
{
    ColorSpinorDev* r  = gpuState.d_r;
    ColorSpinorDev* r0 = gpuState.d_r0;
    ColorSpinorDev* p  = gpuState.d_p;
    ColorSpinorDev* v  = gpuState.d_Ap;    // reuse Ap as v
    ColorSpinorDev* s  = gpuState.d_s;
    ColorSpinorDev* t  = gpuState.d_t;

    // Initial: x = 0, r = b
    gpu_field_zero(x, vol);
    gpu_field_copy(b, r, vol);

    // Random shadow residual: just copy b (simpler than CPU version,
    // but works for most cases)
    gpu_field_copy(b, r0, vol);
    gpu_field_copy(r, p, vol);

    gpu_real_t bnorm2 = gpu_field_norm2(b, vol);
    if (bnorm2 < 1.0e-35) return 0;

    gpu_complex_t rho = gpu_field_dot(r0, r, vol);
    gpu_complex_t alpha_bic(1.0, 0.0);
    gpu_complex_t omega(1.0, 0.0);

    if (verbose)
        printf("  BiCGstab: initial |r|^2 = %.6e, |b|^2 = %.6e\n",
               gpu_field_norm2(r, vol), bnorm2);

    int iter;
    for (iter = 0; iter < cgMaxIter; iter++) {
        // v = A * p
        gpu_apply_dirac(p, v, kappa, wilsonR, useClover, vol);

        // alpha = rho / (r0, v)
        gpu_complex_t r0v = gpu_field_dot(r0, v, vol);
        if (gpu_abs(r0v) < 1.0e-35) {
            printf("  WARNING: BiCGstab breakdown: (r0, v) ~ 0\n");
            break;
        }
        alpha_bic = rho / r0v;

        // s = r - alpha * v
        gpu_field_copy(r, s, vol);
        gpu_field_accumulate(-alpha_bic, v, s, vol);

        // Check if s is small enough
        gpu_real_t snorm2 = gpu_field_norm2(s, vol);
        if (sqrt(snorm2 / bnorm2) < cgTolerance) {
            gpu_field_accumulate(alpha_bic, p, x, vol);
            iter++;
            if (verbose)
                printf("  BiCGstab converged (early) in %d iterations\n", iter);
            break;
        }

        // t = A * s
        gpu_apply_dirac(s, t, kappa, wilsonR, useClover, vol);

        // omega = (t, s) / (t, t)
        gpu_complex_t ts = gpu_field_dot(t, s, vol);
        gpu_real_t tt = gpu_field_norm2(t, vol);
        if (tt < 1.0e-35) {
            printf("  WARNING: BiCGstab breakdown: |t|^2 ~ 0\n");
            break;
        }
        omega = ts / gpu_complex_t(tt, 0.0);

        // x += alpha * p + omega * s
        gpu_field_accumulate(alpha_bic, p, x, vol);
        gpu_field_accumulate(omega, s, x, vol);

        // r = s - omega * t
        gpu_field_copy(s, r, vol);
        gpu_field_accumulate(-omega, t, r, vol);

        // Check convergence
        gpu_real_t rnorm2 = gpu_field_norm2(r, vol);
        gpu_real_t relResidual = sqrt(rnorm2 / bnorm2);

        if (verbose && (iter + 1) % 100 == 0)
            printf("  BiCGstab iter %5d: |r|/|b| = %.6e\n", iter + 1, relResidual);

        if (relResidual < cgTolerance) {
            iter++;
            if (verbose)
                printf("  BiCGstab converged in %d iterations: |r|/|b| = %.6e\n",
                       iter, relResidual);
            break;
        }

        // rho_new = (r0, r)
        gpu_complex_t rho_new = gpu_field_dot(r0, r, vol);
        if (gpu_abs(rho_new) < 1.0e-35) {
            printf("  WARNING: BiCGstab breakdown: rho ~ 0\n");
            break;
        }

        // beta = (rho_new / rho) * (alpha / omega)
        gpu_complex_t betaBic = (rho_new / rho) * (alpha_bic / omega);

        // p = r + beta * (p - omega * v)
        kernel_bicgstab_update_p<<<gpu_grid_size(vol), BLOCK_SIZE>>>(
            r, p, v, betaBic, omega, vol);
        CUDA_CHECK_LAST();

        rho = rho_new;
    }

    if (iter >= cgMaxIter) {
        gpu_real_t finalRes = sqrt(gpu_field_norm2(r, vol) / bnorm2);
        printf("  WARNING: BiCGstab did not converge in %d iterations. |r|/|b| = %.6e\n",
               cgMaxIter, finalRes);
    }

    return iter;
}
