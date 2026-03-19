//-----------------------------------------------
// Dirac operator dispatch (Wilson vs clover)
//
// GPU equivalents of apply_dirac / apply_dirac_dagger
//-----------------------------------------------
#include "gpu_state.cuh"
#include "su3_device.cuh"
#include "colorspinor_device.cuh"

// Forward declarations from other GPU files
void gpu_wilson_dirac(const ColorSpinorDev* psi, ColorSpinorDev* result,
                      gpu_real_t kappa, gpu_real_t wilsonR, int vol);
void gpu_wilson_dirac_dagger(const ColorSpinorDev* psi, ColorSpinorDev* result,
                             ColorSpinorDev* tmp1, ColorSpinorDev* tmp2,
                             gpu_real_t kappa, gpu_real_t wilsonR, int vol);
void gpu_clover_dirac(const ColorSpinorDev* psi, ColorSpinorDev* result,
                      gpu_real_t kappa, gpu_real_t wilsonR, int vol);
void gpu_clover_dirac_dagger(const ColorSpinorDev* psi, ColorSpinorDev* result,
                             ColorSpinorDev* tmp1, ColorSpinorDev* tmp2,
                             gpu_real_t kappa, gpu_real_t wilsonR, int vol);

//-----------------------------------------------
// Dispatch: apply Dirac operator
//-----------------------------------------------
void gpu_apply_dirac(const ColorSpinorDev* psi, ColorSpinorDev* result,
                     gpu_real_t kappa, gpu_real_t wilsonR,
                     bool useClover, int vol)
{
    if (useClover) {
        gpu_clover_dirac(psi, result, kappa, wilsonR, vol);
    } else {
        gpu_wilson_dirac(psi, result, kappa, wilsonR, vol);
    }
}

//-----------------------------------------------
// Dispatch: apply Dirac dagger operator
//-----------------------------------------------
void gpu_apply_dirac_dagger(const ColorSpinorDev* psi, ColorSpinorDev* result,
                            ColorSpinorDev* tmp1, ColorSpinorDev* tmp2,
                            gpu_real_t kappa, gpu_real_t wilsonR,
                            bool useClover, int vol)
{
    if (useClover) {
        gpu_clover_dirac_dagger(psi, result, tmp1, tmp2, kappa, wilsonR, vol);
    } else {
        gpu_wilson_dirac_dagger(psi, result, tmp1, tmp2, kappa, wilsonR, vol);
    }
}

//-----------------------------------------------
// Cache invalidation wrapper (called from leapfrog)
//-----------------------------------------------
void gpu_invalidate_clover_cache()
{
    gpu_invalidate_clover_field();
}
