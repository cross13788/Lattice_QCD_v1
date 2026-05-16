//-----------------------------------------------
// Heat bath sweep on GPU
//
// Checkerboard decomposition: two kernel launches
// (even sites, then odd sites). Each thread updates
// all 4 links at one site using Cabibbo-Marinari
// heat bath with Kennedy-Pendleton algorithm.
//
// Uses persistent cuRAND Philox states.
//-----------------------------------------------
#include "gpu_state.cuh"
#include "su3_device.cuh"

//-----------------------------------------------
// SU(2) heat bath (Kennedy-Pendleton) on device
//-----------------------------------------------
__device__ __forceinline__
void dev_su2_heat_bath(gpu_real_t beta_eff, const gpu_complex_t W[2][2],
                       gpu_complex_t a[2][2], curandState* rngState)
{
    gpu_real_t w0 = 0.5 * (W[0][0].real() + W[1][1].real());
    gpu_real_t w3 = 0.5 * (W[0][0].imag() - W[1][1].imag());
    gpu_real_t w1 = 0.5 * (W[0][1].imag() + W[1][0].imag());
    gpu_real_t w2 = 0.5 * (W[0][1].real() - W[1][0].real());

    gpu_real_t k = sqrt(w0*w0 + w1*w1 + w2*w2 + w3*w3);

    if (k < 1.0e-35) {
        // Random SU(2)
        gpu_real_t r[4];
        gpu_real_t norm = 0.0;
        for (int i = 0; i < 4; i++) {
            r[i] = curand_normal_double(rngState);
            norm += r[i] * r[i];
        }
        norm = rsqrt(norm);
        for (int i = 0; i < 4; i++) r[i] *= norm;
        a[0][0] = gpu_complex_t(r[0], r[3]);
        a[0][1] = gpu_complex_t(r[2], r[1]);
        a[1][0] = gpu_complex_t(-r[2], r[1]);
        a[1][1] = gpu_complex_t(r[0], -r[3]);
        return;
    }

    gpu_complex_t Vdag[2][2];
    gpu_real_t kinv = 1.0 / k;
    Vdag[0][0] = gpu_complex_t( w0, -w3) * kinv;
    Vdag[0][1] = gpu_complex_t(-w2, -w1) * kinv;
    Vdag[1][0] = gpu_complex_t( w2, -w1) * kinv;
    Vdag[1][1] = gpu_complex_t( w0,  w3) * kinv;

    gpu_real_t alpha = 2.0 * beta_eff * k;
    gpu_real_t a0;

    // Kennedy-Pendleton acceptance-rejection
    while (true) {
        gpu_real_t r1 = curand_uniform_double(rngState);
        gpu_real_t r2 = curand_uniform_double(rngState);
        gpu_real_t r3 = curand_uniform_double(rngState);
        gpu_real_t r4 = curand_uniform_double(rngState);

        if (r1 < 1.0e-300) r1 = 1.0e-300;
        if (r3 < 1.0e-300) r3 = 1.0e-300;

        gpu_real_t lambda2 = -log(r1) / alpha;
        gpu_real_t cosAngle = cos(2.0 * GPU_PI * r2);
        gpu_real_t mu2 = -log(r3) / alpha;

        gpu_real_t x = lambda2 + mu2 * cosAngle * cosAngle;
        if (x > 2.0) continue;
        if (r4 * r4 <= 1.0 - 0.5 * x) {
            a0 = 1.0 - x;
            break;
        }
    }

    gpu_real_t rr = sqrt(1.0 - a0 * a0);
    gpu_real_t cosTheta = 2.0 * curand_uniform_double(rngState) - 1.0;
    gpu_real_t sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    gpu_real_t phi = 2.0 * GPU_PI * curand_uniform_double(rngState);

    gpu_real_t a1 = rr * sinTheta * cos(phi);
    gpu_real_t a2 = rr * sinTheta * sin(phi);
    gpu_real_t a3 = rr * cosTheta;

    // aRel * Vdag
    gpu_complex_t aRel[2][2];
    aRel[0][0] = gpu_complex_t(a0, a3);
    aRel[0][1] = gpu_complex_t(a2, a1);
    aRel[1][0] = gpu_complex_t(-a2, a1);
    aRel[1][1] = gpu_complex_t(a0, -a3);

    a[0][0] = aRel[0][0] * Vdag[0][0] + aRel[0][1] * Vdag[1][0];
    a[0][1] = aRel[0][0] * Vdag[0][1] + aRel[0][1] * Vdag[1][1];
    a[1][0] = aRel[1][0] * Vdag[0][0] + aRel[1][1] * Vdag[1][0];
    a[1][1] = aRel[1][0] * Vdag[0][1] + aRel[1][1] * Vdag[1][1];
}

//-----------------------------------------------
// Heat bath update of one SU(2) subgroup
//-----------------------------------------------
__device__ __forceinline__
void dev_su3_heat_bath_subgroup(SU3matrixDev& U, const SU3matrixDev& staple,
                                gpu_real_t beta, int subgroup,
                                curandState* rngState)
{
    SU3matrixDev W;
    dev_su3_multiply(U, staple, W);

    int idx[3][2] = {{0,1}, {0,2}, {1,2}};
    int i0 = idx[subgroup][0], i1 = idx[subgroup][1];

    gpu_complex_t Wsub[2][2];
    Wsub[0][0] = W.m[i0][i0];
    Wsub[0][1] = W.m[i0][i1];
    Wsub[1][0] = W.m[i1][i0];
    Wsub[1][1] = W.m[i1][i1];

    gpu_complex_t a[2][2];
    gpu_real_t beta_eff = beta / (gpu_real_t)GPU_NC;
    dev_su2_heat_bath(beta_eff, Wsub, a, rngState);

    SU3matrixDev R;
    dev_su3_identity(R);
    R.m[i0][i0] = a[0][0];
    R.m[i0][i1] = a[0][1];
    R.m[i1][i0] = a[1][0];
    R.m[i1][i1] = a[1][1];

    SU3matrixDev tmp;
    dev_su3_multiply(R, U, tmp);
    dev_su3_copy(tmp, U);
}

//-----------------------------------------------
// Heat bath kernel: one thread per site, ONE link
// direction (targetMu) per launch.
//
// Site even/odd parity alone is NOT a sufficient
// coloring when a thread updates all 4 directions:
// link (x,mu) and link (x+mu-nu,nu) share the
// negative-staple plaquette, and x and x+mu-nu have
// the SAME site parity (mu and nu each flip parity
// once). Updating them simultaneously (Jacobi, as
// on the GPU's thousands of concurrent threads)
// violates detailed balance -> wrong equilibrium.
//
// Fixing mu per launch removes this: the only
// direction-mu links in the staple of (x,mu) sit
// at OPPOSITE-parity sites (x+/-nu), and direction-
// nu links (nu != mu) are not updated this launch.
// So no two concurrently-updated links share a
// plaquette. (Standard correct GPU heat-bath
// coloring; see Gattringer & Lang Ch. 4.)
//-----------------------------------------------
__global__
void kernel_heat_bath(SU3matrixDev* gaugeField,
                      const int* neighborPlus,
                      const int* neighborMinus,
                      const int* siteParity,
                      curandState* randStates,
                      gpu_real_t beta,
                      int targetParity,
                      int targetMu,
                      int vol)
{
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= vol) return;
    if (siteParity[site] != targetParity) return;

    curandState localState = randStates[site];

    int mu = targetMu;
    SU3matrixDev staple;
    dev_compute_staple(gaugeField, neighborPlus, neighborMinus, site, mu, staple);

    // Cabibbo-Marinari: cycle through 3 SU(2) subgroups
    for (int sub = 0; sub < 3; sub++) {
        dev_su3_heat_bath_subgroup(gaugeField[site * 4 + mu], staple,
                                   beta, sub, &localState);
    }
    dev_su3_reunitarize(gaugeField[site * 4 + mu]);

    randStates[site] = localState;
}

void gpu_heat_bath_sweep(gpu_real_t beta, int vol)
{
    // mu-outer, then even sites / odd sites: race-free coloring.
    for (int mu = 0; mu < 4; mu++) {
        for (int parity = 0; parity < 2; parity++) {
            kernel_heat_bath<<<gpu_grid_size(vol), BLOCK_SIZE>>>(
                gpuState.d_gaugeField,
                gpuState.d_neighborPlus, gpuState.d_neighborMinus,
                gpuState.d_siteParity, gpuState.d_randStates,
                beta, parity, mu, vol);
            CUDA_CHECK_LAST();
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
}
