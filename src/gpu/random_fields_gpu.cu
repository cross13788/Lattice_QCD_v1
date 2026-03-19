//-----------------------------------------------
// cuRAND-based random field generation
//
// - Gaussian conjugate momenta
// - Gaussian spinor field for pseudofermion generation
//-----------------------------------------------
#include "gpu_state.cuh"
#include "su3_device.cuh"
#include "colorspinor_device.cuh"
#include "reduction_gpu.cuh"

//-----------------------------------------------
// Generate Gaussian conjugate momenta
// (traceless anti-Hermitian)
//-----------------------------------------------
__global__
void kernel_generate_momenta(SU3matrixDev* momentum,
                             curandState* randStates,
                             int nLinks, int vol)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nLinks) return;

    // Use site-based RNG (idx / 4 gives site)
    int site = idx / 4;
    curandState localState = randStates[site];

    SU3matrixDev& H = momentum[idx];

    // Off-diagonal elements
    for (int i = 0; i < GPU_NC; i++) {
        for (int j = i + 1; j < GPU_NC; j++) {
            gpu_real_t a = curand_normal_double(&localState) / sqrt(2.0);
            gpu_real_t b = curand_normal_double(&localState) / sqrt(2.0);
            H.m[i][j] = gpu_complex_t(a, b);
            H.m[j][i] = gpu_complex_t(-a, b);
        }
    }

    // Diagonal: traceless anti-Hermitian
    gpu_real_t d[GPU_NC];
    gpu_real_t dsum = 0.0;
    for (int i = 0; i < GPU_NC - 1; i++) {
        d[i] = curand_normal_double(&localState) / sqrt(2.0);
        dsum += d[i];
    }
    d[GPU_NC - 1] = -dsum;

    for (int i = 0; i < GPU_NC; i++)
        H.m[i][i] = gpu_complex_t(0.0, d[i]);

    randStates[site] = localState;
}

void gpu_generate_momenta(int vol)
{
    int nLinks = vol * 4;
    kernel_generate_momenta<<<gpu_grid_size(nLinks), BLOCK_SIZE>>>(
        gpuState.d_momentum, gpuState.d_randStates, nLinks, vol);
    CUDA_CHECK_LAST();
}

//-----------------------------------------------
// Kinetic energy: T = (1/2) sum |H_ij|^2
//-----------------------------------------------
__global__
void kernel_kinetic_energy(const SU3matrixDev* momentum,
                           gpu_real_t* partials,
                           int nLinks)
{
    gpu_real_t sum = 0.0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < nLinks; i += stride)
        sum += dev_su3_algebra_norm2(momentum[i]);

    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0)
        partials[blockIdx.x] = sum;
}

gpu_real_t gpu_compute_kinetic_energy(int vol)
{
    int nLinks = vol * 4;
    int nBlocks = gpu_grid_size(nLinks);
    if (nBlocks > gpuState.reductionBufSize) nBlocks = gpuState.reductionBufSize;

    kernel_kinetic_energy<<<nBlocks, BLOCK_SIZE>>>(
        gpuState.d_momentum, gpuState.d_reductionBuf, nLinks);
    CUDA_CHECK_LAST();

    return 0.5 * gpu_reduce_partial(nBlocks);
}

//-----------------------------------------------
// Generate Gaussian spinor field xi
// (for pseudofermion: phi = D† xi)
//-----------------------------------------------
__global__
void kernel_generate_gaussian_spinor(ColorSpinorDev* xi,
                                     curandState* randStates,
                                     int vol)
{
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= vol) return;

    curandState localState = randStates[site];

    for (int s = 0; s < GPU_ND; s++) {
        for (int c = 0; c < GPU_NC; c++) {
            gpu_real_t re = curand_normal_double(&localState) / sqrt(2.0);
            gpu_real_t im = curand_normal_double(&localState) / sqrt(2.0);
            xi[site].v[s][c] = gpu_complex_t(re, im);
        }
    }

    randStates[site] = localState;
}

void gpu_generate_gaussian_spinor(ColorSpinorDev* xi, int vol)
{
    kernel_generate_gaussian_spinor<<<gpu_grid_size(vol), BLOCK_SIZE>>>(
        xi, gpuState.d_randStates, vol);
    CUDA_CHECK_LAST();
}
