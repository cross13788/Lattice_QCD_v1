//-----------------------------------------------
// Overrelaxation (microcanonical) sweep on GPU
//
// Checkerboard decomposition. Deterministic update
// (no random numbers needed).
//-----------------------------------------------
#include "gpu_state.cuh"
#include "su3_device.cuh"

//-----------------------------------------------
// Overrelaxation update of one link (device)
//-----------------------------------------------
__device__ __forceinline__
void dev_su3_overrelaxation_update(SU3matrixDev& U, const SU3matrixDev& staple)
{
    for (int sub = 0; sub < 3; sub++) {
        SU3matrixDev W;
        dev_su3_multiply(U, staple, W);

        int idx[3][2] = {{0,1}, {0,2}, {1,2}};
        int i0 = idx[sub][0], i1 = idx[sub][1];

        gpu_complex_t Wsub[2][2];
        Wsub[0][0] = W.m[i0][i0];
        Wsub[0][1] = W.m[i0][i1];
        Wsub[1][0] = W.m[i1][i0];
        Wsub[1][1] = W.m[i1][i1];

        gpu_real_t w0 = 0.5 * (Wsub[0][0].real() + Wsub[1][1].real());
        gpu_real_t w3 = 0.5 * (Wsub[0][0].imag() - Wsub[1][1].imag());
        gpu_real_t w1 = 0.5 * (Wsub[0][1].imag() + Wsub[1][0].imag());
        gpu_real_t w2 = 0.5 * (Wsub[0][1].real() - Wsub[1][0].real());
        gpu_real_t k = sqrt(w0*w0 + w1*w1 + w2*w2 + w3*w3);
        if (k < 1.0e-35) continue;

        gpu_real_t v0 = w0/k, v1 = -w1/k, v2 = -w2/k, v3 = -w3/k;
        // V^{-2} = V^dag * V^dag
        gpu_real_t r0 = v0*v0 - v1*v1 - v2*v2 - v3*v3;
        gpu_real_t r1 = 2.0*v0*v1;
        gpu_real_t r2 = 2.0*v0*v2;
        gpu_real_t r3 = 2.0*v0*v3;

        gpu_complex_t Rsub[2][2];
        Rsub[0][0] = gpu_complex_t(r0, r3);
        Rsub[0][1] = gpu_complex_t(r2, r1);
        Rsub[1][0] = gpu_complex_t(-r2, r1);
        Rsub[1][1] = gpu_complex_t(r0, -r3);

        SU3matrixDev R;
        dev_su3_identity(R);
        R.m[i0][i0] = Rsub[0][0];
        R.m[i0][i1] = Rsub[0][1];
        R.m[i1][i0] = Rsub[1][0];
        R.m[i1][i1] = Rsub[1][1];

        SU3matrixDev tmp;
        dev_su3_multiply(R, U, tmp);
        dev_su3_copy(tmp, U);
    }
    dev_su3_reunitarize(U);
}

//-----------------------------------------------
// Overrelaxation kernel
//-----------------------------------------------
__global__
void kernel_overrelaxation(SU3matrixDev* gaugeField,
                           const int* neighborPlus,
                           const int* neighborMinus,
                           const int* siteParity,
                           int targetParity,
                           int vol)
{
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= vol) return;
    if (siteParity[site] != targetParity) return;

    for (int mu = 0; mu < 4; mu++) {
        SU3matrixDev staple;
        dev_compute_staple(gaugeField, neighborPlus, neighborMinus, site, mu, staple);
        dev_su3_overrelaxation_update(gaugeField[site * 4 + mu], staple);
    }
}

void gpu_overrelaxation_sweep(int vol)
{
    for (int parity = 0; parity < 2; parity++) {
        kernel_overrelaxation<<<gpu_grid_size(vol), BLOCK_SIZE>>>(
            gpuState.d_gaugeField,
            gpuState.d_neighborPlus, gpuState.d_neighborMinus,
            gpuState.d_siteParity, parity, vol);
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}
