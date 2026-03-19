//-----------------------------------------------
// Gauge field operations on GPU
//
// - Plaquette computation (reduction kernel)
// - Staple is a device-inline function in su3_device.cuh
//-----------------------------------------------
#include "gpu_state.cuh"
#include "su3_device.cuh"
#include "reduction_gpu.cuh"

//-----------------------------------------------
// Plaquette kernel: each thread computes 6 plaquettes
// at one site, reduces to partial sums per block
//-----------------------------------------------
__global__
void kernel_plaquette(const SU3matrixDev* gaugeField,
                      const int* neighborPlus,
                      gpu_real_t* partials,
                      int vol)
{
    gpu_real_t localSum = 0.0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int site = idx; site < vol; site += stride) {
        for (int mu = 0; mu < 4; mu++) {
            for (int nu = mu + 1; nu < 4; nu++) {
                int sitePlusMu = neighborPlus[site * 4 + mu];
                int sitePlusNu = neighborPlus[site * 4 + nu];

                SU3matrixDev tmp1, tmp2, tmp3, plaq;

                dev_su3_multiply(gaugeField[site * 4 + mu],
                                 gaugeField[sitePlusMu * 4 + nu], tmp1);
                dev_su3_dagger(gaugeField[sitePlusNu * 4 + mu], tmp2);
                dev_su3_multiply(tmp1, tmp2, tmp3);

                SU3matrixDev UdagNu;
                dev_su3_dagger(gaugeField[site * 4 + nu], UdagNu);
                dev_su3_multiply(tmp3, UdagNu, plaq);

                localSum += dev_su3_retrace(plaq);
            }
        }
    }

    localSum = block_reduce_sum(localSum);
    if (threadIdx.x == 0)
        partials[blockIdx.x] = localSum;
}

gpu_real_t gpu_compute_plaquette(int vol)
{
    int nBlocks = gpu_grid_size(vol);
    if (nBlocks > gpuState.reductionBufSize) nBlocks = gpuState.reductionBufSize;

    kernel_plaquette<<<nBlocks, BLOCK_SIZE>>>(
        gpuState.d_gaugeField, gpuState.d_neighborPlus,
        gpuState.d_reductionBuf, vol);
    CUDA_CHECK_LAST();

    gpu_real_t plaqSum = gpu_reduce_partial(nBlocks);
    return plaqSum / (6.0 * vol * GPU_NC);
}

//-----------------------------------------------
// Action density kernel: per-site action density
//-----------------------------------------------
__global__
void kernel_action_density(const SU3matrixDev* gaugeField,
                           const int* neighborPlus,
                           gpu_real_t* actionDensity,
                           int vol)
{
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= vol) return;

    gpu_real_t localAction = 0.0;

    for (int mu = 0; mu < 4; mu++) {
        for (int nu = mu + 1; nu < 4; nu++) {
            int sitePlusMu = neighborPlus[site * 4 + mu];
            int sitePlusNu = neighborPlus[site * 4 + nu];

            SU3matrixDev tmp1, tmp2, tmp3, plaq;

            dev_su3_multiply(gaugeField[site * 4 + mu],
                             gaugeField[sitePlusMu * 4 + nu], tmp1);
            dev_su3_dagger(gaugeField[sitePlusNu * 4 + mu], tmp2);
            dev_su3_multiply(tmp1, tmp2, tmp3);

            SU3matrixDev UdagNu;
            dev_su3_dagger(gaugeField[site * 4 + nu], UdagNu);
            dev_su3_multiply(tmp3, UdagNu, plaq);

            localAction += (1.0 - dev_su3_retrace(plaq) / GPU_NC);
        }
    }

    actionDensity[site] = localAction;
}

void gpu_compute_action_density(gpu_real_t* d_actionDensity, int vol)
{
    kernel_action_density<<<gpu_grid_size(vol), BLOCK_SIZE>>>(
        gpuState.d_gaugeField, gpuState.d_neighborPlus, d_actionDensity, vol);
    CUDA_CHECK_LAST();
}
