//-----------------------------------------------
// Pseudofermion action on GPU
//
// S_PF = phi† (D†D)^{-1} phi
//      = Re(phi, X) where D†D X = phi
//-----------------------------------------------
#include "gpu_state.cuh"
#include "colorspinor_device.cuh"

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

gpu_real_t gpu_compute_pseudofermion_action(gpu_real_t kappa, gpu_real_t wilsonR,
                                            bool useClover, int cgMaxIter,
                                            gpu_real_t cgTolerance, int vol)
{
    ColorSpinorDev* phi = gpuState.d_phi;
    ColorSpinorDev* X   = gpuState.d_X;
    ColorSpinorDev* r   = gpuState.d_r;
    ColorSpinorDev* p   = gpuState.d_p;
    ColorSpinorDev* Ap  = gpuState.d_Ap;
    ColorSpinorDev* tmp = gpuState.d_tmp;
    ColorSpinorDev* tmp2 = gpuState.d_tmp2;

    // CG for D†D X = phi
    gpu_field_zero(X, vol);
    gpu_field_copy(phi, r, vol);
    gpu_field_copy(phi, p, vol);

    gpu_real_t rr = gpu_field_norm2(r, vol);
    gpu_real_t rr0 = rr;

    for (int iter = 0; iter < cgMaxIter; iter++) {
        gpu_apply_dirac(p, tmp, kappa, wilsonR, useClover, vol);
        gpu_apply_dirac_dagger(tmp, Ap, tmp2, gpuState.d_psi, kappa, wilsonR, useClover, vol);

        gpu_complex_t pAp = gpu_field_dot(p, Ap, vol);
        gpu_real_t alpha = rr / pAp.real();

        gpu_field_accumulate(gpu_complex_t(alpha, 0.0), p, X, vol);
        gpu_field_accumulate(gpu_complex_t(-alpha, 0.0), Ap, r, vol);

        gpu_real_t rr_new = gpu_field_norm2(r, vol);
        if (sqrt(rr_new / rr0) < cgTolerance) break;

        gpu_real_t betaCG = rr_new / rr;
        gpu_cg_update_p(r, p, betaCG, vol);
        rr = rr_new;
    }

    // S_PF = Re(phi† X)
    gpu_complex_t action = gpu_field_dot(phi, X, vol);
    return action.real();
}
