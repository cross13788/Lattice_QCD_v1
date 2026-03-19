#pragma once
//-----------------------------------------------
// Warp-level and block-level reduction primitives
//
// Uses __shfl_down_sync for warp reduction and
// shared memory for cross-warp reduction.
//-----------------------------------------------
#include "gpu_common.cuh"

//-----------------------------------------------
// Warp-level reduction (sum) for double
//-----------------------------------------------
__device__ __forceinline__
gpu_real_t warp_reduce_sum(gpu_real_t val)
{
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

//-----------------------------------------------
// Block-level reduction (sum) for double
// Returns result in thread 0 only
//-----------------------------------------------
__device__ __forceinline__
gpu_real_t block_reduce_sum(gpu_real_t val)
{
    __shared__ gpu_real_t shared[32];   // one per warp (max 32 warps)

    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;

    val = warp_reduce_sum(val);

    if (lane == 0)
        shared[wid] = val;

    __syncthreads();

    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    val = (threadIdx.x < numWarps) ? shared[threadIdx.x] : 0.0;

    if (wid == 0)
        val = warp_reduce_sum(val);

    return val;
}

//-----------------------------------------------
// Grid-level reduction kernel
// Each block produces a partial sum; host sums
// the partials (typically ~1000 blocks, trivial).
//-----------------------------------------------
static __global__
void kernel_reduce_sum(const gpu_real_t* __restrict__ input,
                       gpu_real_t* __restrict__ output,
                       int n)
{
    gpu_real_t sum = 0.0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
        sum += input[i];

    sum = block_reduce_sum(sum);

    if (threadIdx.x == 0)
        output[blockIdx.x] = sum;
}
