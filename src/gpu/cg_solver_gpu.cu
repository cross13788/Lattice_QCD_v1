//-----------------------------------------------
// Conjugate Gradient solver on GPU (optimized)
//
// Optimizations:
// 1. Fused x-update + r-update + norm computation
//    eliminates 2 kernel launches per CG iteration
// 2. All vectors stay on device; only 2 scalars
//    (rr, pAp) transfer D2H per iteration
// 3. Uses fused D† kernel (no temporary fields
//    for the gamma5 sandwiching)
//-----------------------------------------------
#include "gpu_state.cuh"
#include "colorspinor_device.cuh"
#include <cstdio>
#include <cmath>

// Forward declarations
void gpu_apply_dirac(const ColorSpinorDev* psi, ColorSpinorDev* result,
                     gpu_real_t kappa, gpu_real_t wilsonR,
                     bool useClover, int vol);
void gpu_apply_dirac_dagger(const ColorSpinorDev* psi, ColorSpinorDev* result,
                            ColorSpinorDev* tmp1, ColorSpinorDev* tmp2,
                            gpu_real_t kappa, gpu_real_t wilsonR,
                            bool useClover, int vol);
void gpu_field_zero(ColorSpinorDev* field, int vol);
void gpu_field_copy(const ColorSpinorDev* src, ColorSpinorDev* result, int vol);
void gpu_field_accumulate(gpu_complex_t alpha, const ColorSpinorDev* x,
                          ColorSpinorDev* result, int vol);
gpu_real_t gpu_field_norm2(const ColorSpinorDev* x, int vol);
gpu_complex_t gpu_field_dot(const ColorSpinorDev* x, const ColorSpinorDev* y, int vol);
void gpu_cg_update_p(const ColorSpinorDev* r, ColorSpinorDev* p,
                     gpu_real_t beta, int vol);
gpu_real_t gpu_cg_fused_update(ColorSpinorDev* x, ColorSpinorDev* r,
                               const ColorSpinorDev* p, const ColorSpinorDev* Ap,
                               gpu_real_t alpha, int vol);

int gpu_cg_solver(const ColorSpinorDev* b, ColorSpinorDev* x,
                  gpu_real_t kappa, gpu_real_t wilsonR,
                  bool useClover, int cgMaxIter, gpu_real_t cgTolerance,
                  bool verbose, int vol)
{
    // Workspace: use pre-allocated solver vectors from gpuState
    ColorSpinorDev* r     = gpuState.d_r;
    ColorSpinorDev* p     = gpuState.d_p;
    ColorSpinorDev* Ap    = gpuState.d_Ap;
    ColorSpinorDev* tmp   = gpuState.d_tmp;
    ColorSpinorDev* tmp2  = gpuState.d_tmp2;
    ColorSpinorDev* Ddagb = gpuState.d_psi;  // reuse psi as temp

    // Compute RHS: Ddagb = D^dag * b
    gpu_apply_dirac_dagger(b, Ddagb, tmp, tmp2, kappa, wilsonR, useClover, vol);

    // Initial: x = 0, r = Ddagb, p = Ddagb
    gpu_field_zero(x, vol);
    gpu_field_copy(Ddagb, r, vol);
    gpu_field_copy(r, p, vol);

    gpu_real_t rr = gpu_field_norm2(r, vol);
    gpu_real_t bnorm2 = gpu_field_norm2(Ddagb, vol);

    if (bnorm2 < 1.0e-35) return 0;

    if (verbose)
        printf("  CG: initial |r|^2 = %.6e, |b|^2 = %.6e\n", rr, bnorm2);

    int iter;
    for (iter = 0; iter < cgMaxIter; iter++) {
        // Ap = D^dag D * p
        gpu_apply_dirac(p, tmp, kappa, wilsonR, useClover, vol);
        gpu_apply_dirac_dagger(tmp, Ap, tmp2, Ddagb, kappa, wilsonR, useClover, vol);

        // alpha = (r, r) / (p, Ap)
        gpu_complex_t pAp = gpu_field_dot(p, Ap, vol);
        gpu_real_t alpha = rr / pAp.real();

        // Fused: x += alpha*p, r -= alpha*Ap, compute |r_new|^2
        gpu_real_t rr_new = gpu_cg_fused_update(x, r, p, Ap, alpha, vol);

        gpu_real_t relResidual = sqrt(rr_new / bnorm2);

        if (verbose && (iter + 1) % 100 == 0)
            printf("  CG iter %5d: |r|/|b| = %.6e\n", iter + 1, relResidual);

        if (relResidual < cgTolerance) {
            iter++;
            if (verbose)
                printf("  CG converged in %d iterations: |r|/|b| = %.6e\n", iter, relResidual);
            break;
        }

        // beta = rr_new / rr_old
        gpu_real_t betaCG = rr_new / rr;

        // p = r + beta * p
        gpu_cg_update_p(r, p, betaCG, vol);

        rr = rr_new;
    }

    if (iter >= cgMaxIter) {
        gpu_real_t finalRes = sqrt(gpu_field_norm2(r, vol) / bnorm2);
        printf("  WARNING: CG did not converge in %d iterations. |r|/|b| = %.6e\n",
               cgMaxIter, finalRes);
    }

    return iter;
}
