//-----------------------------------------------
// Wilson flow RK3 on GPU
//
// Luscher's 3-stage Runge-Kutta scheme,
// all stages on device.
//-----------------------------------------------
#include "gpu_state.cuh"
#include "su3_device.cuh"
#include "reduction_gpu.cuh"
#include <cstdio>
#include <cstring>

// Forward declarations
gpu_real_t gpu_compute_plaquette(int vol);

//-----------------------------------------------
// Flow force kernel: Z = eps * (beta/Nc) * [U * staple]_TA
//-----------------------------------------------
__global__
void kernel_flow_force(const SU3matrixDev* gaugeField,
                       SU3matrixDev* forceField,
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
        dev_su3_scale(gpu_complex_t(coeff, 0.0), forceTA, forceField[idx]);
    }
}

//-----------------------------------------------
// RK3 stage update kernel:
//   W_new = exp(c1*Z_a + c2*Z_b + c3*Z_c) * W_old
//
// General form handles all 3 stages with
// different coefficient combinations.
//-----------------------------------------------
__global__
void kernel_flow_update(SU3matrixDev* gaugeField,
                        const SU3matrixDev* Za,
                        const SU3matrixDev* Zb,
                        const SU3matrixDev* Zc,
                        gpu_real_t ca, gpu_real_t cb, gpu_real_t cc,
                        int nLinks)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nLinks) return;

    SU3matrixDev exponent;
    for (int i = 0; i < GPU_NC; i++)
        for (int j = 0; j < GPU_NC; j++) {
            exponent.m[i][j] = ca * Za[idx].m[i][j];
            if (Zb) exponent.m[i][j] += cb * Zb[idx].m[i][j];
            if (Zc) exponent.m[i][j] += cc * Zc[idx].m[i][j];
        }

    SU3matrixDev expZ;
    dev_su3_exp(exponent, expZ);

    SU3matrixDev Wnew;
    dev_su3_multiply(expZ, gaugeField[idx], Wnew);
    dev_su3_copy(Wnew, gaugeField[idx]);
}

//-----------------------------------------------
// Wilson flow: single RK3 step
//-----------------------------------------------
void gpu_wilson_flow_step(SU3matrixDev* gaugeField, gpu_real_t epsilon,
                          gpu_real_t beta, int vol)
{
    int nLinks = vol * 4;
    gpu_real_t coeff = epsilon * beta / (gpu_real_t)GPU_NC;

    SU3matrixDev* Z0 = gpuState.d_Z0;
    SU3matrixDev* Z1 = gpuState.d_Z1;
    SU3matrixDev* Z2 = gpuState.d_Z2;

    // Stage 0: Z0 = flow_force; W1 = exp(1/4 Z0) * W0
    kernel_flow_force<<<gpu_grid_size(vol), BLOCK_SIZE>>>(
        gaugeField, Z0, gpuState.d_neighborPlus, gpuState.d_neighborMinus, coeff, vol);
    CUDA_CHECK_LAST();

    kernel_flow_update<<<gpu_grid_size(nLinks), BLOCK_SIZE>>>(
        gaugeField, Z0, nullptr, nullptr, 0.25, 0.0, 0.0, nLinks);
    CUDA_CHECK_LAST();

    // Stage 1: Z1 = flow_force; W2 = exp(8/9 Z1 - 17/36 Z0) * W1
    kernel_flow_force<<<gpu_grid_size(vol), BLOCK_SIZE>>>(
        gaugeField, Z1, gpuState.d_neighborPlus, gpuState.d_neighborMinus, coeff, vol);
    CUDA_CHECK_LAST();

    kernel_flow_update<<<gpu_grid_size(nLinks), BLOCK_SIZE>>>(
        gaugeField, Z1, Z0, nullptr, 8.0/9.0, -17.0/36.0, 0.0, nLinks);
    CUDA_CHECK_LAST();

    // Stage 2: Z2 = flow_force; V' = exp(3/4 Z2 - 8/9 Z1 + 17/36 Z0) * W2
    kernel_flow_force<<<gpu_grid_size(vol), BLOCK_SIZE>>>(
        gaugeField, Z2, gpuState.d_neighborPlus, gpuState.d_neighborMinus, coeff, vol);
    CUDA_CHECK_LAST();

    kernel_flow_update<<<gpu_grid_size(nLinks), BLOCK_SIZE>>>(
        gaugeField, Z2, Z1, Z0, 3.0/4.0, -8.0/9.0, 17.0/36.0, nLinks);
    CUDA_CHECK_LAST();
}

//-----------------------------------------------
// Energy density (plaquette definition) on GPU
//-----------------------------------------------
__global__
void kernel_flow_energy(const SU3matrixDev* gaugeField,
                        const int* neighborPlus,
                        gpu_real_t* partials,
                        int vol)
{
    gpu_real_t sum = 0.0;
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

                sum += 2.0 * (1.0 - dev_su3_retrace(plaq) / (gpu_real_t)GPU_NC);
            }
        }
    }

    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0)
        partials[blockIdx.x] = sum;
}

gpu_real_t gpu_flow_energy_plaquette(const SU3matrixDev* gaugeField, int vol)
{
    int nBlocks = gpu_grid_size(vol);
    if (nBlocks > gpuState.reductionBufSize) nBlocks = gpuState.reductionBufSize;

    kernel_flow_energy<<<nBlocks, BLOCK_SIZE>>>(
        gaugeField, gpuState.d_neighborPlus, gpuState.d_reductionBuf, vol);
    CUDA_CHECK_LAST();

    return gpu_reduce_partial(nBlocks) / (gpu_real_t)vol;
}

//-----------------------------------------------
// Wilson flow driver on GPU
// Works on a D2D copy of the gauge field
//-----------------------------------------------
void gpu_wilson_flow_driver(gpu_real_t beta, gpu_real_t flowStepSize,
                            gpu_real_t maxFlowTime,
                            const char* outputDirectory, int vol)
{
    int nLinks = vol * 4;
    int nSteps = (int)(maxFlowTime / flowStepSize + 0.5);

    printf("  Wilson flow: eps=%.4f, t_max=%.2f, %d steps\n",
           flowStepSize, maxFlowTime, nSteps);

    // Allocate flow field on device and copy current gauge field
    SU3matrixDev* d_flowField;
    CUDA_CHECK(cudaMalloc(&d_flowField, nLinks * sizeof(SU3matrixDev)));
    CUDA_CHECK(cudaMemcpy(d_flowField, gpuState.d_gaugeField,
                          nLinks * sizeof(SU3matrixDev), cudaMemcpyDeviceToDevice));

    // Open output file
    char filename[512];
    snprintf(filename, sizeof(filename), "%s/wilson_flow.dat", outputDirectory);
    FILE* outFile = fopen(filename, "w");
    if (!outFile) {
        printf("Error: Cannot open %s\n", filename);
        cudaFree(d_flowField);
        return;
    }
    fprintf(outFile, "# t  t^2*E_plaq  E_plaq\n");

    // Measure at t=0
    gpu_real_t t = 0.0;
    gpu_real_t Eplaq = gpu_flow_energy_plaquette(d_flowField, vol);
    fprintf(outFile, "%.6f  %.10e  %.10e\n", t, t*t*Eplaq, Eplaq);

    // Flow evolution
    for (int step = 1; step <= nSteps; step++) {
        gpu_wilson_flow_step(d_flowField, flowStepSize, beta, vol);
        t = step * flowStepSize;

        Eplaq = gpu_flow_energy_plaquette(d_flowField, vol);
        fprintf(outFile, "%.6f  %.10e  %.10e\n", t, t*t*Eplaq, Eplaq);

        if (step % 10 == 0 || step == nSteps)
            printf("    t=%.3f  t^2<E_plaq>=%.6f\n", t, t*t*Eplaq);
    }

    fclose(outFile);
    printf("    Flow data saved to: %s\n", filename);

    cudaFree(d_flowField);
}
