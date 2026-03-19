//-----------------------------------------------
// Gauge force kernel for HMC
//
// F_G = (beta/NC) * [U * staple]_TA
//-----------------------------------------------
#include "gpu_state.cuh"
#include "su3_device.cuh"

__global__
void kernel_gauge_force(const SU3matrixDev* gaugeField,
                        SU3matrixDev* force,
                        const int* neighborPlus,
                        const int* neighborMinus,
                        gpu_real_t coeff,
                        int vol)
{
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= vol) return;

    for (int mu = 0; mu < 4; mu++) {
        int idx = site * 4 + mu;

        SU3matrixDev staple;
        dev_compute_staple(gaugeField, neighborPlus, neighborMinus, site, mu, staple);

        SU3matrixDev UV;
        dev_su3_multiply(gaugeField[idx], staple, UV);

        SU3matrixDev forceTA;
        dev_su3_traceless_antiherm(UV, forceTA);
        dev_su3_scale(gpu_complex_t(coeff, 0.0), forceTA, force[idx]);
    }
}

void gpu_compute_gauge_force(gpu_real_t beta, int vol)
{
    gpu_real_t coeff = beta / (gpu_real_t)GPU_NC;
    kernel_gauge_force<<<gpu_grid_size(vol), BLOCK_SIZE>>>(
        gpuState.d_gaugeField, gpuState.d_forceGauge,
        gpuState.d_neighborPlus, gpuState.d_neighborMinus,
        coeff, vol);
    CUDA_CHECK_LAST();
}
