//-----------------------------------------------
// Field-level BLAS kernels for ColorSpinor fields
//
// GPU equivalents of colorspinor_module.cpp
// field operations. All vectors live on device.
//-----------------------------------------------
#include "gpu_state.cuh"
#include "colorspinor_device.cuh"
#include "reduction_gpu.cuh"

//-----------------------------------------------
// Zero entire field
//-----------------------------------------------
__global__
void kernel_field_zero(ColorSpinorDev* field, int vol)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vol) return;
    dev_cs_zero(field[idx]);
}

void gpu_field_zero(ColorSpinorDev* field, int vol)
{
    kernel_field_zero<<<gpu_grid_size(vol), BLOCK_SIZE>>>(field, vol);
    CUDA_CHECK_LAST();
}

//-----------------------------------------------
// Copy: result = src
//-----------------------------------------------
__global__
void kernel_field_copy(const ColorSpinorDev* src, ColorSpinorDev* result, int vol)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vol) return;
    dev_cs_copy(src[idx], result[idx]);
}

void gpu_field_copy(const ColorSpinorDev* src, ColorSpinorDev* result, int vol)
{
    kernel_field_copy<<<gpu_grid_size(vol), BLOCK_SIZE>>>(src, result, vol);
    CUDA_CHECK_LAST();
}

//-----------------------------------------------
// Accumulate: result += alpha * x
//-----------------------------------------------
__global__
void kernel_field_accumulate(gpu_complex_t alpha,
                             const ColorSpinorDev* x,
                             ColorSpinorDev* result,
                             int vol)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vol) return;
    dev_cs_accumulate(alpha, x[idx], result[idx]);
}

void gpu_field_accumulate(gpu_complex_t alpha,
                          const ColorSpinorDev* x,
                          ColorSpinorDev* result,
                          int vol)
{
    kernel_field_accumulate<<<gpu_grid_size(vol), BLOCK_SIZE>>>(alpha, x, result, vol);
    CUDA_CHECK_LAST();
}

//-----------------------------------------------
// Scale: result = alpha * x
//-----------------------------------------------
__global__
void kernel_field_scale(gpu_complex_t alpha,
                        const ColorSpinorDev* x,
                        ColorSpinorDev* result,
                        int vol)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vol) return;
    dev_cs_scale(alpha, x[idx], result[idx]);
}

void gpu_field_scale(gpu_complex_t alpha,
                     const ColorSpinorDev* x,
                     ColorSpinorDev* result,
                     int vol)
{
    kernel_field_scale<<<gpu_grid_size(vol), BLOCK_SIZE>>>(alpha, x, result, vol);
    CUDA_CHECK_LAST();
}

//-----------------------------------------------
// AXPY: result = alpha * x + y
//-----------------------------------------------
__global__
void kernel_field_axpy(gpu_complex_t alpha,
                       const ColorSpinorDev* x,
                       const ColorSpinorDev* y,
                       ColorSpinorDev* result,
                       int vol)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vol) return;

    #pragma unroll
    for (int s = 0; s < GPU_ND; s++)
        #pragma unroll
        for (int c = 0; c < GPU_NC; c++)
            result[idx].v[s][c] = alpha * x[idx].v[s][c] + y[idx].v[s][c];
}

void gpu_field_axpy(gpu_complex_t alpha,
                    const ColorSpinorDev* x,
                    const ColorSpinorDev* y,
                    ColorSpinorDev* result,
                    int vol)
{
    kernel_field_axpy<<<gpu_grid_size(vol), BLOCK_SIZE>>>(alpha, x, y, result, vol);
    CUDA_CHECK_LAST();
}

//-----------------------------------------------
// Norm2 reduction: sum |x|^2 over all sites
//-----------------------------------------------
__global__
void kernel_field_norm2(const ColorSpinorDev* x, gpu_real_t* partials, int vol)
{
    gpu_real_t sum = 0.0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < vol; i += stride)
        sum += dev_cs_norm2(x[i]);

    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0)
        partials[blockIdx.x] = sum;
}

gpu_real_t gpu_field_norm2(const ColorSpinorDev* x, int vol)
{
    int nBlocks = gpu_grid_size(vol);
    if (nBlocks > gpuState.reductionBufSize) nBlocks = gpuState.reductionBufSize;

    kernel_field_norm2<<<nBlocks, BLOCK_SIZE>>>(x, gpuState.d_reductionBuf, vol);
    CUDA_CHECK_LAST();

    return gpu_reduce_partial(nBlocks);
}

//-----------------------------------------------
// Dot product reduction: sum conj(x) * y
//-----------------------------------------------
__global__
void kernel_field_dot(const ColorSpinorDev* x,
                      const ColorSpinorDev* y,
                      gpu_real_t* partialsReal,
                      gpu_real_t* partialsImag,
                      int vol)
{
    gpu_real_t sumR = 0.0;
    gpu_real_t sumI = 0.0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < vol; i += stride) {
        gpu_complex_t d = dev_cs_dot(x[i], y[i]);
        sumR += d.real();
        sumI += d.imag();
    }

    sumR = block_reduce_sum(sumR);
    sumI = block_reduce_sum(sumI);

    if (threadIdx.x == 0) {
        partialsReal[blockIdx.x] = sumR;
        partialsImag[blockIdx.x] = sumI;
    }
}

gpu_complex_t gpu_field_dot(const ColorSpinorDev* x, const ColorSpinorDev* y, int vol)
{
    int nBlocks = gpu_grid_size(vol);
    if (nBlocks > gpuState.reductionBufSize / 2) nBlocks = gpuState.reductionBufSize / 2;

    // Use first half for real, second half for imag
    gpu_real_t* partialsReal = gpuState.d_reductionBuf;
    gpu_real_t* partialsImag = gpuState.d_reductionBuf + nBlocks;

    kernel_field_dot<<<nBlocks, BLOCK_SIZE>>>(x, y, partialsReal, partialsImag, vol);
    CUDA_CHECK_LAST();

    // Copy both halves
    CUDA_CHECK(cudaMemcpy(gpuState.h_reductionBuf, gpuState.d_reductionBuf,
                          2 * nBlocks * sizeof(gpu_real_t), cudaMemcpyDeviceToHost));

    gpu_real_t sumR = 0.0, sumI = 0.0;
    for (int i = 0; i < nBlocks; i++) {
        sumR += gpuState.h_reductionBuf[i];
        sumI += gpuState.h_reductionBuf[nBlocks + i];
    }

    return gpu_complex_t(sumR, sumI);
}

//-----------------------------------------------
// CG update: p = r + beta * p (fused kernel)
//-----------------------------------------------
__global__
void kernel_cg_update_p(const ColorSpinorDev* r,
                        ColorSpinorDev* p,
                        gpu_real_t beta,
                        int vol)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vol) return;

    #pragma unroll
    for (int s = 0; s < GPU_ND; s++)
        #pragma unroll
        for (int c = 0; c < GPU_NC; c++)
            p[idx].v[s][c] = r[idx].v[s][c] + beta * p[idx].v[s][c];
}

void gpu_cg_update_p(const ColorSpinorDev* r, ColorSpinorDev* p,
                     gpu_real_t beta, int vol)
{
    kernel_cg_update_p<<<gpu_grid_size(vol), BLOCK_SIZE>>>(r, p, beta, vol);
    CUDA_CHECK_LAST();
}

//-----------------------------------------------
// Fused CG update: x += alpha*p, r -= alpha*Ap,
// and compute |r_new|^2 — all in a single pass.
//
// Eliminates 2 extra full-field traversals per CG
// iteration (was: accumulate x, accumulate r,
// norm2 r = 3 kernel launches with 3 reads).
// Now: 1 kernel, 1 read of each field.
//-----------------------------------------------
__global__
void kernel_cg_fused_update(ColorSpinorDev* __restrict__ x,
                            ColorSpinorDev* __restrict__ r,
                            const ColorSpinorDev* __restrict__ p,
                            const ColorSpinorDev* __restrict__ Ap,
                            gpu_real_t alpha,
                            gpu_real_t* __restrict__ partials,
                            int vol)
{
    gpu_real_t localNorm = 0.0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < vol; i += stride) {
        gpu_real_t siteNorm = 0.0;
        #pragma unroll
        for (int s = 0; s < GPU_ND; s++) {
            #pragma unroll
            for (int c = 0; c < GPU_NC; c++) {
                // x += alpha * p
                x[i].v[s][c] += alpha * p[i].v[s][c];

                // r -= alpha * Ap
                gpu_complex_t r_new = r[i].v[s][c] - alpha * Ap[i].v[s][c];
                r[i].v[s][c] = r_new;

                // |r_new|^2
                siteNorm += gpu_norm(r_new);
            }
        }
        localNorm += siteNorm;
    }

    localNorm = block_reduce_sum(localNorm);
    if (threadIdx.x == 0)
        partials[blockIdx.x] = localNorm;
}

gpu_real_t gpu_cg_fused_update(ColorSpinorDev* x, ColorSpinorDev* r,
                               const ColorSpinorDev* p, const ColorSpinorDev* Ap,
                               gpu_real_t alpha, int vol)
{
    int nBlocks = gpu_grid_size(vol);
    if (nBlocks > gpuState.reductionBufSize) nBlocks = gpuState.reductionBufSize;

    kernel_cg_fused_update<<<nBlocks, BLOCK_SIZE>>>(
        x, r, p, Ap, alpha, gpuState.d_reductionBuf, vol);
    CUDA_CHECK_LAST();

    return gpu_reduce_partial(nBlocks);
}

//-----------------------------------------------
// Fused field zero via cudaMemsetAsync (faster
// than a custom kernel for zeroing)
//-----------------------------------------------
void gpu_field_zero_async(ColorSpinorDev* field, int vol, cudaStream_t stream)
{
    CUDA_CHECK(cudaMemsetAsync(field, 0, vol * sizeof(ColorSpinorDev), stream));
}
