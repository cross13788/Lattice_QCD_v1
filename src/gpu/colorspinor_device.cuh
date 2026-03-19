#pragma once
//-----------------------------------------------
// Device-inline ColorSpinor operations
//
// ColorSpinor has 4 Dirac (spin) x 3 color
// complex components at a single lattice site.
//-----------------------------------------------
#include "gpu_common.cuh"

//-----------------------------------------------
// ColorSpinor struct for device code
// Same memory layout as CPU ColorSpinor
//-----------------------------------------------
struct ColorSpinorDev {
    gpu_complex_t v[GPU_ND][GPU_NC];   // v[spin][color]
};

//-----------------------------------------------
// Single-site operations
//-----------------------------------------------

__device__ __forceinline__
void dev_cs_zero(ColorSpinorDev& result)
{
    #pragma unroll
    for (int s = 0; s < GPU_ND; s++)
        #pragma unroll
        for (int c = 0; c < GPU_NC; c++)
            result.v[s][c] = gpu_complex_t(0.0, 0.0);
}

__device__ __forceinline__
void dev_cs_copy(const ColorSpinorDev& src, ColorSpinorDev& result)
{
    #pragma unroll
    for (int s = 0; s < GPU_ND; s++)
        #pragma unroll
        for (int c = 0; c < GPU_NC; c++)
            result.v[s][c] = src.v[s][c];
}

__device__ __forceinline__
void dev_cs_accumulate(gpu_complex_t alpha, const ColorSpinorDev& x, ColorSpinorDev& result)
{
    #pragma unroll
    for (int s = 0; s < GPU_ND; s++)
        #pragma unroll
        for (int c = 0; c < GPU_NC; c++)
            result.v[s][c] += alpha * x.v[s][c];
}

__device__ __forceinline__
void dev_cs_scale(gpu_complex_t alpha, const ColorSpinorDev& x, ColorSpinorDev& result)
{
    #pragma unroll
    for (int s = 0; s < GPU_ND; s++)
        #pragma unroll
        for (int c = 0; c < GPU_NC; c++)
            result.v[s][c] = alpha * x.v[s][c];
}

__device__ __forceinline__
gpu_complex_t dev_cs_dot(const ColorSpinorDev& x, const ColorSpinorDev& y)
{
    gpu_complex_t sum(0.0, 0.0);
    #pragma unroll
    for (int s = 0; s < GPU_ND; s++)
        #pragma unroll
        for (int c = 0; c < GPU_NC; c++)
            sum += gpu_conj(x.v[s][c]) * y.v[s][c];
    return sum;
}

__device__ __forceinline__
gpu_real_t dev_cs_norm2(const ColorSpinorDev& x)
{
    gpu_real_t sum = 0.0;
    #pragma unroll
    for (int s = 0; s < GPU_ND; s++)
        #pragma unroll
        for (int c = 0; c < GPU_NC; c++)
            sum += gpu_norm(x.v[s][c]);
    return sum;
}
