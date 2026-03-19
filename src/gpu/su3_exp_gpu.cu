//-----------------------------------------------
// SU(3) exponential update and momentum kernels
//-----------------------------------------------
#include "gpu_state.cuh"
#include "su3_device.cuh"

//-----------------------------------------------
// Momentum update kernel:
//   momentum -= stepSize * (forceGauge + forceFermion)
//-----------------------------------------------
__global__
void kernel_momentum_update(SU3matrixDev* momentum,
                            const SU3matrixDev* forceGauge,
                            const SU3matrixDev* forceFermion,
                            gpu_real_t stepSize,
                            int nLinks)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nLinks) return;

    #pragma unroll
    for (int i = 0; i < GPU_NC; i++)
        #pragma unroll
        for (int j = 0; j < GPU_NC; j++)
            momentum[idx].m[i][j] -= stepSize * (
                forceGauge[idx].m[i][j] + forceFermion[idx].m[i][j]
            );
}

void gpu_momentum_update(gpu_real_t stepSize, int nLinks)
{
    kernel_momentum_update<<<gpu_grid_size(nLinks), BLOCK_SIZE>>>(
        gpuState.d_momentum, gpuState.d_forceGauge, gpuState.d_forceFermion,
        stepSize, nLinks);
    CUDA_CHECK_LAST();
}

//-----------------------------------------------
// Gauge field update kernel:
//   U = exp(stepSize * H) * U
//-----------------------------------------------
__global__
void kernel_gauge_exp_update(SU3matrixDev* gaugeField,
                             const SU3matrixDev* momentum,
                             gpu_real_t stepSize,
                             int nLinks)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nLinks) return;

    dev_su3_exp_update(stepSize, momentum[idx], gaugeField[idx]);
}

void gpu_gauge_exp_update(gpu_real_t stepSize, int nLinks)
{
    // Use BLOCK_SIZE_EXP (128) for exp kernel to reduce register pressure
    // The 12-term Taylor series + reunitarize uses many registers
    kernel_gauge_exp_update<<<gpu_grid_size(nLinks, BLOCK_SIZE_EXP), BLOCK_SIZE_EXP>>>(
        gpuState.d_gaugeField, gpuState.d_momentum, stepSize, nLinks);
    CUDA_CHECK_LAST();
}
