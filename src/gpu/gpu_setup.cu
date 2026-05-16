//-----------------------------------------------
// GPU initialization, upload/download, cleanup
//-----------------------------------------------
#include "gpu_state.cuh"
#include <cstring>

//-----------------------------------------------
// Constant memory definitions
//-----------------------------------------------
__constant__ int c_gammaIdx[5 * 4];
__constant__ double c_gammaValRe[5 * 4];
__constant__ double c_gammaValIm[5 * 4];

//-----------------------------------------------
// CUDA streams
//-----------------------------------------------
cudaStream_t stream_compute = 0;
cudaStream_t stream_reduce  = 0;

//-----------------------------------------------
// Initialize GPU: select device, print info,
// allocate core arrays
//-----------------------------------------------
void gpu_init(int vol)
{
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA-capable devices found\n");
        exit(EXIT_FAILURE);
    }

    gpuState.deviceId = 0;
    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("GPU Device: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Global memory: %.1f MB\n", prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("  SMs: %d\n", prop.multiProcessorCount);
    printf("  Max threads/block: %d\n", prop.maxThreadsPerBlock);
    printf("  Warp size: %d\n\n", prop.warpSize);

    gpuState.totalMemory = prop.totalGlobalMem;
    gpuState.vol = vol;
    gpuState.nLinks = vol * 4;

    // Allocate gauge field
    CUDA_CHECK(cudaMalloc(&gpuState.d_gaugeField, gpuState.nLinks * sizeof(SU3matrixDev)));

    // Allocate reduction buffer (enough for any grid size)
    gpuState.reductionBufSize = 4096;
    CUDA_CHECK(cudaMalloc(&gpuState.d_reductionBuf, gpuState.reductionBufSize * sizeof(gpu_real_t)));
    CUDA_CHECK(cudaMallocHost(&gpuState.h_reductionBuf, gpuState.reductionBufSize * sizeof(gpu_real_t)));

    // Initialize flags
    gpuState.solverAllocated = false;
    gpuState.hmcAllocated = false;
    gpuState.flowAllocated = false;
    gpuState.cloverAllocated = false;
    gpuState.randAllocated = false;
    gpuState.cloverCsw = 0.0;

    // Create CUDA streams for async overlap
    gpu_create_streams();

    // Estimate memory usage
    size_t gaugeBytes = gpuState.nLinks * sizeof(SU3matrixDev);
    printf("GPU memory allocated:\n");
    printf("  Gauge field: %.1f MB\n", gaugeBytes / (1024.0 * 1024.0));
}

//-----------------------------------------------
// Upload geometry tables to device
//-----------------------------------------------
void gpu_upload_geometry(const int* neighborPlus, const int* neighborMinus,
                         const int* siteParity, int vol)
{
    size_t nbSize = vol * 4 * sizeof(int);
    size_t parSize = vol * sizeof(int);

    CUDA_CHECK(cudaMalloc(&gpuState.d_neighborPlus, nbSize));
    CUDA_CHECK(cudaMalloc(&gpuState.d_neighborMinus, nbSize));
    CUDA_CHECK(cudaMalloc(&gpuState.d_siteParity, parSize));

    CUDA_CHECK(cudaMemcpy(gpuState.d_neighborPlus, neighborPlus, nbSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuState.d_neighborMinus, neighborMinus, nbSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuState.d_siteParity, siteParity, parSize, cudaMemcpyHostToDevice));
}

//-----------------------------------------------
// Upload gamma matrices to device
//-----------------------------------------------
void gpu_upload_gamma(const int* gammaIdx, const void* gammaVal)
{
    size_t idxSize = 5 * 4 * sizeof(int);
    size_t valSize = 5 * 4 * sizeof(gpu_complex_t);

    CUDA_CHECK(cudaMalloc(&gpuState.d_gammaIdx, idxSize));
    CUDA_CHECK(cudaMalloc(&gpuState.d_gammaVal, valSize));

    CUDA_CHECK(cudaMemcpy(gpuState.d_gammaIdx, gammaIdx, idxSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpuState.d_gammaVal, gammaVal, valSize, cudaMemcpyHostToDevice));

    // Also upload to constant memory for broadcast access
    gpu_upload_gamma_const(gammaIdx, gammaVal);
}

//-----------------------------------------------
// Upload gamma matrices to constant memory
// Splits complex values into separate real/imag
// arrays for coalesced constant cache access
//-----------------------------------------------
void gpu_upload_gamma_const(const int* gammaIdx, const void* gammaVal)
{
    CUDA_CHECK(cudaMemcpyToSymbol(c_gammaIdx, gammaIdx, 5 * 4 * sizeof(int)));

    // Split complex values into real and imaginary parts
    const gpu_complex_t* vals = (const gpu_complex_t*)gammaVal;
    double realParts[20], imagParts[20];
    for (int i = 0; i < 20; i++) {
        realParts[i] = vals[i].real();
        imagParts[i] = vals[i].imag();
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_gammaValRe, realParts, 20 * sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_gammaValIm, imagParts, 20 * sizeof(double)));
}

//-----------------------------------------------
// CUDA streams
//-----------------------------------------------
void gpu_create_streams()
{
    CUDA_CHECK(cudaStreamCreate(&stream_compute));
    CUDA_CHECK(cudaStreamCreate(&stream_reduce));
}

void gpu_destroy_streams()
{
    if (stream_compute) { cudaStreamDestroy(stream_compute); stream_compute = 0; }
    if (stream_reduce)  { cudaStreamDestroy(stream_reduce);  stream_reduce  = 0; }
}

//-----------------------------------------------
// Upload/download gauge field
//-----------------------------------------------
void gpu_upload_gauge(const void* hostGaugeField, int nLinks)
{
    CUDA_CHECK(cudaMemcpy(gpuState.d_gaugeField, hostGaugeField,
                          nLinks * sizeof(SU3matrixDev), cudaMemcpyHostToDevice));
}

void gpu_download_gauge(void* hostGaugeField, int nLinks)
{
    CUDA_CHECK(cudaMemcpy(hostGaugeField, gpuState.d_gaugeField,
                          nLinks * sizeof(SU3matrixDev), cudaMemcpyDeviceToHost));
}

//-----------------------------------------------
// Allocate solver workspace
//-----------------------------------------------
void gpu_alloc_solver(int vol)
{
    if (gpuState.solverAllocated) return;

    size_t fieldSize = vol * sizeof(ColorSpinorDev);

    CUDA_CHECK(cudaMalloc(&gpuState.d_psi,    fieldSize));
    CUDA_CHECK(cudaMalloc(&gpuState.d_result, fieldSize));
    CUDA_CHECK(cudaMalloc(&gpuState.d_r,      fieldSize));
    CUDA_CHECK(cudaMalloc(&gpuState.d_p,      fieldSize));
    CUDA_CHECK(cudaMalloc(&gpuState.d_Ap,     fieldSize));
    CUDA_CHECK(cudaMalloc(&gpuState.d_tmp,    fieldSize));
    CUDA_CHECK(cudaMalloc(&gpuState.d_tmp2,   fieldSize));
    CUDA_CHECK(cudaMalloc(&gpuState.d_r0,     fieldSize));
    CUDA_CHECK(cudaMalloc(&gpuState.d_s,      fieldSize));
    CUDA_CHECK(cudaMalloc(&gpuState.d_t,      fieldSize));

    gpuState.solverAllocated = true;

    printf("  Solver workspace: %.1f MB\n",
           10.0 * fieldSize / (1024.0 * 1024.0));
}

//-----------------------------------------------
// Allocate HMC workspace
//-----------------------------------------------
void gpu_alloc_hmc(int vol)
{
    if (gpuState.hmcAllocated) return;

    int nLinks = vol * 4;
    size_t linkSize = nLinks * sizeof(SU3matrixDev);
    size_t fieldSize = vol * sizeof(ColorSpinorDev);

    CUDA_CHECK(cudaMalloc(&gpuState.d_gaugeFieldSaved, linkSize));
    CUDA_CHECK(cudaMalloc(&gpuState.d_momentum, linkSize));
    CUDA_CHECK(cudaMalloc(&gpuState.d_forceGauge, linkSize));
    CUDA_CHECK(cudaMalloc(&gpuState.d_forceFermion, linkSize));
    CUDA_CHECK(cudaMalloc(&gpuState.d_phi, fieldSize));
    CUDA_CHECK(cudaMalloc(&gpuState.d_X, fieldSize));
    CUDA_CHECK(cudaMalloc(&gpuState.d_Y, fieldSize));

    gpuState.hmcAllocated = true;

    printf("  HMC workspace: %.1f MB\n",
           (4.0 * linkSize + 3.0 * fieldSize) / (1024.0 * 1024.0));
}

//-----------------------------------------------
// Allocate Wilson flow workspace
//-----------------------------------------------
void gpu_alloc_flow(int vol)
{
    if (gpuState.flowAllocated) return;

    int nLinks = vol * 4;
    size_t linkSize = nLinks * sizeof(SU3matrixDev);

    CUDA_CHECK(cudaMalloc(&gpuState.d_Z0, linkSize));
    CUDA_CHECK(cudaMalloc(&gpuState.d_Z1, linkSize));
    CUDA_CHECK(cudaMalloc(&gpuState.d_Z2, linkSize));

    gpuState.flowAllocated = true;
}

//-----------------------------------------------
// Allocate cuRAND states
//-----------------------------------------------
__global__
void kernel_init_curand(curandState* states, int n, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        curand_init(seed, idx, 0, &states[idx]);
}

void gpu_alloc_rand(int nStates, int seed)
{
    if (gpuState.randAllocated) return;

    gpuState.nRandStates = nStates;
    CUDA_CHECK(cudaMalloc(&gpuState.d_randStates, nStates * sizeof(curandState)));

    int grid = gpu_grid_size(nStates);
    kernel_init_curand<<<grid, BLOCK_SIZE>>>(gpuState.d_randStates, nStates, seed);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());

    gpuState.randAllocated = true;

    printf("  cuRAND states: %d (%.1f MB)\n", nStates,
           nStates * sizeof(curandState) / (1024.0 * 1024.0));
}

//-----------------------------------------------
// Allocate clover workspace
//-----------------------------------------------
void gpu_alloc_clover(int vol)
{
    if (gpuState.cloverAllocated) return;

    size_t blockSize = vol * CLOVER_BLOCK_DIM * CLOVER_BLOCK_DIM * sizeof(gpu_complex_t);

    CUDA_CHECK(cudaMalloc(&gpuState.d_cloverUpper, blockSize));
    CUDA_CHECK(cudaMalloc(&gpuState.d_cloverLower, blockSize));
    CUDA_CHECK(cudaMalloc(&gpuState.d_cloverInvUpper, blockSize));
    CUDA_CHECK(cudaMalloc(&gpuState.d_cloverInvLower, blockSize));

    // Sigma matrices: 6 * 4 * 4 complex
    size_t sigSize = 6 * 4 * 4 * sizeof(gpu_complex_t);
    CUDA_CHECK(cudaMalloc(&gpuState.d_sigmaMat, sigSize));

    // Clover-force scatter accumulator: vol * 4 links
    CUDA_CHECK(cudaMalloc(&gpuState.d_cloverForceAcc,
                          (size_t)vol * 4 * sizeof(SU3matrixDev)));

    gpuState.cloverAllocated = true;
}

//-----------------------------------------------
// Upload host sigma matrices to the device.
//
// Host sigmaMat is complex_t[6][4][4] (std::complex<double>,
// guaranteed {re,im} contiguous) — bit-identical layout to
// gpu_complex_t[6][4][4] (pair-major: idx*16 + s1*4 + s2),
// matching kernel_compute_clover_field's indexing. Also records
// c_sw for the clover-field prefactor.
//-----------------------------------------------
void gpu_upload_sigma(const void* hostSigma, double csw)
{
    size_t sigSize = 6 * 4 * 4 * sizeof(gpu_complex_t);
    CUDA_CHECK(cudaMemcpy(gpuState.d_sigmaMat, hostSigma, sigSize,
                          cudaMemcpyHostToDevice));
    gpuState.cloverCsw = csw;
}

//-----------------------------------------------
// Reduction helper: sum partial results on host
//-----------------------------------------------
gpu_real_t gpu_reduce_partial(int nBlocks)
{
    CUDA_CHECK(cudaMemcpy(gpuState.h_reductionBuf, gpuState.d_reductionBuf,
                          nBlocks * sizeof(gpu_real_t), cudaMemcpyDeviceToHost));
    gpu_real_t sum = 0.0;
    for (int i = 0; i < nBlocks; i++)
        sum += gpuState.h_reductionBuf[i];
    return sum;
}

//-----------------------------------------------
// Free all GPU resources
//-----------------------------------------------
void gpu_free()
{
    // Core
    if (gpuState.d_gaugeField) cudaFree(gpuState.d_gaugeField);
    if (gpuState.d_neighborPlus) cudaFree(gpuState.d_neighborPlus);
    if (gpuState.d_neighborMinus) cudaFree(gpuState.d_neighborMinus);
    if (gpuState.d_siteParity) cudaFree(gpuState.d_siteParity);
    if (gpuState.d_gammaIdx) cudaFree(gpuState.d_gammaIdx);
    if (gpuState.d_gammaVal) cudaFree(gpuState.d_gammaVal);
    if (gpuState.d_reductionBuf) cudaFree(gpuState.d_reductionBuf);
    if (gpuState.h_reductionBuf) cudaFreeHost(gpuState.h_reductionBuf);

    // Solver
    if (gpuState.solverAllocated) {
        cudaFree(gpuState.d_psi);
        cudaFree(gpuState.d_result);
        cudaFree(gpuState.d_r);
        cudaFree(gpuState.d_p);
        cudaFree(gpuState.d_Ap);
        cudaFree(gpuState.d_tmp);
        cudaFree(gpuState.d_tmp2);
        cudaFree(gpuState.d_r0);
        cudaFree(gpuState.d_s);
        cudaFree(gpuState.d_t);
    }

    // HMC
    if (gpuState.hmcAllocated) {
        cudaFree(gpuState.d_gaugeFieldSaved);
        cudaFree(gpuState.d_momentum);
        cudaFree(gpuState.d_forceGauge);
        cudaFree(gpuState.d_forceFermion);
        cudaFree(gpuState.d_phi);
        cudaFree(gpuState.d_X);
        cudaFree(gpuState.d_Y);
    }

    // Flow
    if (gpuState.flowAllocated) {
        cudaFree(gpuState.d_Z0);
        cudaFree(gpuState.d_Z1);
        cudaFree(gpuState.d_Z2);
    }

    // cuRAND
    if (gpuState.randAllocated) {
        cudaFree(gpuState.d_randStates);
    }

    // Clover
    if (gpuState.cloverAllocated) {
        cudaFree(gpuState.d_cloverUpper);
        cudaFree(gpuState.d_cloverLower);
        cudaFree(gpuState.d_cloverInvUpper);
        cudaFree(gpuState.d_cloverInvLower);
        cudaFree(gpuState.d_sigmaMat);
    }

    // Action density
    if (gpuState.d_actionDensity) cudaFree(gpuState.d_actionDensity);

    // Streams
    gpu_destroy_streams();

    memset(&gpuState, 0, sizeof(GPUState));
    printf("GPU resources freed.\n");
}
