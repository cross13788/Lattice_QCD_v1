//-----------------------------------------------
// GPU measurement kernels (optimized)
//
// All measurements run on GPU to avoid D2H
// transfers during the measurement phase.
//
// - Polyakov loop (reduction kernel)
// - Wilson loop W(R,T) (GPU kernel)
// - Action density (per-site kernel)
//-----------------------------------------------
#include "gpu_state.cuh"
#include "su3_device.cuh"
#include "colorspinor_device.cuh"
#include "reduction_gpu.cuh"

//-----------------------------------------------
// Polyakov loop kernel
//-----------------------------------------------
__global__
void kernel_polyakov_loop(const SU3matrixDev* gaugeField,
                          const int* neighborPlus,
                          gpu_real_t* partials,
                          int nX, int nY, int nZ, int nT,
                          int spatialVolume)
{
    gpu_real_t sum = 0.0;
    int spatIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int sIdx = spatIdx; sIdx < spatialVolume; sIdx += stride) {
        int iz = sIdx / (nX * nY);
        int rem = sIdx % (nX * nY);
        int iy = rem / nX;
        int ix = rem % nX;

        int site = ix + nX * (iy + nY * (iz + nZ * 0));

        SU3matrixDev poly;
        dev_su3_identity(poly);

        for (int it = 0; it < nT; it++) {
            SU3matrixDev tmp;
            dev_su3_multiply(poly, gaugeField[site * 4 + 3], tmp);
            dev_su3_copy(tmp, poly);
            site = neighborPlus[site * 4 + 3];
        }

        sum += dev_su3_retrace(poly);
    }

    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0)
        partials[blockIdx.x] = sum;
}

gpu_real_t gpu_polyakov_loop(int nX, int nY, int nZ, int nT, int spatialVolume)
{
    int nBlocks = gpu_grid_size(spatialVolume);
    if (nBlocks > gpuState.reductionBufSize) nBlocks = gpuState.reductionBufSize;

    kernel_polyakov_loop<<<nBlocks, BLOCK_SIZE>>>(
        gpuState.d_gaugeField, gpuState.d_neighborPlus,
        gpuState.d_reductionBuf, nX, nY, nZ, nT, spatialVolume);
    CUDA_CHECK_LAST();

    gpu_real_t polySum = gpu_reduce_partial(nBlocks);
    return polySum / (GPU_NC * spatialVolume);
}

//-----------------------------------------------
// Wilson loop kernel W(R,T)
//
// Each thread computes one W(R,T) at one site
// for one spatial direction. Grid-stride loop
// with reduction for the average.
//-----------------------------------------------
__global__
void kernel_wilson_loop(const SU3matrixDev* __restrict__ gaugeField,
                        const int* __restrict__ neighborPlus,
                        gpu_real_t* __restrict__ partials,
                        int muS,    // spatial direction
                        int R, int T,
                        int vol)
{
    gpu_real_t sum = 0.0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int site = idx; site < vol; site += stride) {
        // Spatial path of length R
        SU3matrixDev pathSpatial;
        dev_su3_identity(pathSpatial);
        int cs = site;
        for (int r = 0; r < R; r++) {
            SU3matrixDev tmp;
            dev_su3_multiply(pathSpatial, gaugeField[cs * 4 + muS], tmp);
            dev_su3_copy(tmp, pathSpatial);
            cs = neighborPlus[cs * 4 + muS];
        }

        // Temporal path of length T from site+R
        SU3matrixDev pathTemporal;
        dev_su3_identity(pathTemporal);
        int sitePlusR = cs;
        cs = sitePlusR;
        for (int t = 0; t < T; t++) {
            SU3matrixDev tmp;
            dev_su3_multiply(pathTemporal, gaugeField[cs * 4 + 3], tmp);
            dev_su3_copy(tmp, pathTemporal);
            cs = neighborPlus[cs * 4 + 3];
        }

        // Backward spatial path from site+T
        SU3matrixDev pathSpatialBack;
        dev_su3_identity(pathSpatialBack);
        cs = site;
        for (int t = 0; t < T; t++)
            cs = neighborPlus[cs * 4 + 3];
        for (int r = 0; r < R; r++) {
            SU3matrixDev tmp;
            dev_su3_multiply(pathSpatialBack, gaugeField[cs * 4 + muS], tmp);
            dev_su3_copy(tmp, pathSpatialBack);
            cs = neighborPlus[cs * 4 + muS];
        }
        SU3matrixDev pathSBDag;
        dev_su3_dagger(pathSpatialBack, pathSBDag);

        // Backward temporal path from site
        SU3matrixDev pathTemporalOrig;
        dev_su3_identity(pathTemporalOrig);
        cs = site;
        for (int t = 0; t < T; t++) {
            SU3matrixDev tmp;
            dev_su3_multiply(pathTemporalOrig, gaugeField[cs * 4 + 3], tmp);
            dev_su3_copy(tmp, pathTemporalOrig);
            cs = neighborPlus[cs * 4 + 3];
        }
        SU3matrixDev pathTODag;
        dev_su3_dagger(pathTemporalOrig, pathTODag);

        // W = pathSpatial * pathTemporal * pathSBDag * pathTODag
        SU3matrixDev tmp1, tmp2, loop;
        dev_su3_multiply(pathSpatial, pathTemporal, tmp1);
        dev_su3_multiply(tmp1, pathSBDag, tmp2);
        dev_su3_multiply(tmp2, pathTODag, loop);

        sum += dev_su3_retrace(loop);
    }

    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0)
        partials[blockIdx.x] = sum;
}

void gpu_wilson_loop(gpu_real_t* wilsonLoops, int rMax, int tMax, int vol)
{
    for (int i = 0; i < rMax * tMax; i++)
        wilsonLoops[i] = 0.0;

    // Loop over 3 spatial directions
    for (int spatialDir = 0; spatialDir < 3; spatialDir++) {
        for (int R = 1; R <= rMax; R++) {
            for (int T = 1; T <= tMax; T++) {
                int nBlocks = gpu_grid_size(vol);
                if (nBlocks > gpuState.reductionBufSize) nBlocks = gpuState.reductionBufSize;

                kernel_wilson_loop<<<nBlocks, BLOCK_SIZE>>>(
                    gpuState.d_gaugeField, gpuState.d_neighborPlus,
                    gpuState.d_reductionBuf,
                    spatialDir, R, T, vol);
                CUDA_CHECK_LAST();

                gpu_real_t wlSum = gpu_reduce_partial(nBlocks);
                wilsonLoops[(R-1) * tMax + (T-1)] += wlSum / (GPU_NC * vol);
            }
        }
    }

    // Average over 3 spatial directions
    for (int i = 0; i < rMax * tMax; i++)
        wilsonLoops[i] /= 3.0;
}
