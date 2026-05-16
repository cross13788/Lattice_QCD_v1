//-----------------------------------------------
// Fermion force on GPU
//
// Solves (D†D) X = phi via CG on device, then
// computes the fermion force from X and Y = D*X.
//-----------------------------------------------
#include "gpu_state.cuh"
#include "su3_device.cuh"
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
void gpu_compute_clover_force(gpu_real_t c_sw, int vol);

//-----------------------------------------------
// Fermion force outer-product kernel
//
// For each link U_mu(x):
//   M_mu(x) = sum_s [ (r-gamma_mu) X(x+mu)]_s outer Y†(x)]_s
//           + sum_s [ Y(x+mu)]_s outer [(r+gamma_mu) X(x)]†_s
//   F = -2*kappa * [U * M]_TA
//-----------------------------------------------
__global__
void kernel_fermion_force(const SU3matrixDev* gaugeField,
                          const ColorSpinorDev* X,
                          const ColorSpinorDev* Y,
                          SU3matrixDev* force,
                          const int* neighborPlus,
                          const int* gammaIdx,
                          const gpu_complex_t* gammaVal,
                          gpu_real_t kappa, gpu_real_t wilsonR,
                          int vol)
{
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= vol) return;

    for (int mu = 0; mu < 4; mu++) {
        int idx = site * 4 + mu;
        int sitePlus = neighborPlus[site * 4 + mu];

        SU3matrixDev M;
        dev_su3_zero(M);

        // Forward contribution
        for (int s = 0; s < GPU_ND; s++) {
            int gs = gammaIdx[mu * 4 + s];
            gpu_complex_t gv = gammaVal[mu * 4 + s];

            for (int a = 0; a < GPU_NC; a++) {
                gpu_complex_t fwd_s_a = wilsonR * X[sitePlus].v[s][a]
                                       - gv * X[sitePlus].v[gs][a];
                for (int b = 0; b < GPU_NC; b++)
                    M.m[a][b] += fwd_s_a * gpu_conj(Y[site].v[s][b]);
            }
        }

        // Backward contribution
        for (int s = 0; s < GPU_ND; s++) {
            int gs = gammaIdx[mu * 4 + s];
            gpu_complex_t gv = gammaVal[mu * 4 + s];

            for (int a = 0; a < GPU_NC; a++) {
                gpu_complex_t bwd_s_a = wilsonR * Y[sitePlus].v[s][a]
                                       + gv * Y[sitePlus].v[gs][a];
                for (int b = 0; b < GPU_NC; b++)
                    M.m[a][b] += gpu_conj(X[site].v[s][b]) * bwd_s_a;
            }
        }

        SU3matrixDev UM, forceTA;
        dev_su3_multiply(gaugeField[idx], M, UM);
        dev_su3_traceless_antiherm(UM, forceTA);
        dev_su3_scale(gpu_complex_t(-2.0 * kappa, 0.0), forceTA, force[idx]);
    }
}

void gpu_compute_fermion_force(gpu_real_t kappa, gpu_real_t wilsonR,
                               bool useClover, int cgMaxIter,
                               gpu_real_t cgTolerance, int vol)
{
    ColorSpinorDev* X   = gpuState.d_X;
    ColorSpinorDev* Y   = gpuState.d_Y;
    ColorSpinorDev* phi = gpuState.d_phi;

    // Internal CG workspace: reuse solver vectors
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

    // Y = D * X
    gpu_apply_dirac(X, Y, kappa, wilsonR, useClover, vol);

    // Compute force kernel (Wilson part)
    kernel_fermion_force<<<gpu_grid_size(vol), BLOCK_SIZE>>>(
        gpuState.d_gaugeField, X, Y,
        gpuState.d_forceFermion,
        gpuState.d_neighborPlus,
        gpuState.d_gammaIdx, gpuState.d_gammaVal,
        kappa, wilsonR, vol);
    CUDA_CHECK_LAST();

    // Clover part (validated port of CPU compute_clover_force)
    if (useClover)
        gpu_compute_clover_force(gpuState.cloverCsw, vol);
}
