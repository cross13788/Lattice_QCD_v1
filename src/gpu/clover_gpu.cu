//-----------------------------------------------
// Clover term on GPU (optimized with caching)
//
// Optimizations over baseline:
// 1. Clover field is precomputed once per gauge
//    config and cached in GPU memory, rather than
//    computing 6 field-strength tensors on-the-fly
//    per thread per Dirac application.
// 2. The cached clover matrix is stored as two
//    6x6 blocks per site (upper/lower), matching
//    the CPU layout for efficient application.
// 3. The clover field is invalidated when the
//    gauge field changes (HMC leapfrog).
//-----------------------------------------------
#include "gpu_state.cuh"
#include "su3_device.cuh"
#include "colorspinor_device.cuh"

// Forward declarations
void gpu_wilson_dirac(const ColorSpinorDev* psi, ColorSpinorDev* result,
                      gpu_real_t kappa, gpu_real_t wilsonR, int vol);
void gpu_wilson_dirac_dagger(const ColorSpinorDev* psi, ColorSpinorDev* result,
                             ColorSpinorDev* tmp1, ColorSpinorDev* tmp2,
                             gpu_real_t kappa, gpu_real_t wilsonR, int vol);

//-----------------------------------------------
// Clover cache state
//-----------------------------------------------
static bool gpuCloverFieldComputed = false;

void gpu_invalidate_clover_field()
{
    gpuCloverFieldComputed = false;
}

//-----------------------------------------------
// Field strength tensor F_{mu,nu} via 4-leaf clover
//-----------------------------------------------
__device__ __forceinline__
void dev_compute_field_strength(const SU3matrixDev* gaugeField,
                                const int* neighborPlus,
                                const int* neighborMinus,
                                int site, int mu, int nu,
                                SU3matrixDev& Fmunu)
{
    int sitePlusMu  = neighborPlus[site * 4 + mu];
    int sitePlusNu  = neighborPlus[site * 4 + nu];
    int siteMinusMu = neighborMinus[site * 4 + mu];
    int siteMinusNu = neighborMinus[site * 4 + nu];
    int siteMinusMuPlusNu  = neighborPlus[siteMinusMu * 4 + nu];
    int siteMinusMuMinusNu = neighborMinus[siteMinusMu * 4 + nu];
    int siteMinusNuPlusMu  = neighborPlus[siteMinusNu * 4 + mu];

    SU3matrixDev tmp1, tmp2, tmp3, plaq, Qmunu;
    dev_su3_zero(Qmunu);

    // Leaf 1
    dev_su3_multiply(gaugeField[site * 4 + mu], gaugeField[sitePlusMu * 4 + nu], tmp1);
    dev_su3_dagger(gaugeField[sitePlusNu * 4 + mu], tmp2);
    dev_su3_multiply(tmp1, tmp2, tmp3);
    dev_su3_dagger(gaugeField[site * 4 + nu], tmp2);
    dev_su3_multiply(tmp3, tmp2, plaq);
    dev_su3_accumulate(plaq, Qmunu);

    // Leaf 2
    dev_su3_dagger(gaugeField[siteMinusMuPlusNu * 4 + mu], tmp1);
    dev_su3_multiply(gaugeField[site * 4 + nu], tmp1, tmp2);
    dev_su3_dagger(gaugeField[siteMinusMu * 4 + nu], tmp1);
    dev_su3_multiply(tmp2, tmp1, tmp3);
    dev_su3_multiply(tmp3, gaugeField[siteMinusMu * 4 + mu], plaq);
    dev_su3_accumulate(plaq, Qmunu);

    // Leaf 3
    dev_su3_dagger(gaugeField[siteMinusMu * 4 + mu], tmp1);
    dev_su3_dagger(gaugeField[siteMinusMuMinusNu * 4 + nu], tmp2);
    dev_su3_multiply(tmp1, tmp2, tmp3);
    dev_su3_multiply(tmp3, gaugeField[siteMinusMuMinusNu * 4 + mu], tmp1);
    dev_su3_multiply(tmp1, gaugeField[siteMinusNu * 4 + nu], plaq);
    dev_su3_accumulate(plaq, Qmunu);

    // Leaf 4
    dev_su3_dagger(gaugeField[siteMinusNu * 4 + nu], tmp1);
    dev_su3_multiply(tmp1, gaugeField[siteMinusNu * 4 + mu], tmp2);
    dev_su3_multiply(tmp2, gaugeField[siteMinusNuPlusMu * 4 + nu], tmp3);
    dev_su3_dagger(gaugeField[site * 4 + mu], tmp1);
    dev_su3_multiply(tmp3, tmp1, plaq);
    dev_su3_accumulate(plaq, Qmunu);

    // F = (Q - Q†) / 8
    SU3matrixDev Qdag;
    dev_su3_dagger(Qmunu, Qdag);
    dev_su3_subtract(Qmunu, Qdag, Fmunu);
    dev_su3_scale(gpu_complex_t(1.0 / 8.0, 0.0), Fmunu, Fmunu);
}

//-----------------------------------------------
// Kernel: precompute clover field A(x)
// Assembles two 6x6 Hermitian blocks per site
// Stored in flat arrays: d_cloverUpper[site*36 + row*6 + col]
//-----------------------------------------------
__global__
void kernel_compute_clover_field(
    const SU3matrixDev* __restrict__ gaugeField,
    const int* __restrict__ neighborPlus,
    const int* __restrict__ neighborMinus,
    const gpu_complex_t* __restrict__ sigmaMat,
    gpu_real_t c_sw,
    gpu_complex_t* __restrict__ cloverUpper,
    gpu_complex_t* __restrict__ cloverLower,
    int vol)
{
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= vol) return;

    // Zero blocks
    #pragma unroll
    for (int i = 0; i < 36; i++) {
        cloverUpper[site * 36 + i] = gpu_complex_t(0.0, 0.0);
        cloverLower[site * 36 + i] = gpu_complex_t(0.0, 0.0);
    }

    gpu_complex_t prefactor(0.0, 0.25 * c_sw);

    int pairIdx = 0;
    for (int mu = 0; mu < 4; mu++) {
        for (int nu = mu + 1; nu < 4; nu++) {
            SU3matrixDev Fmunu;
            dev_compute_field_strength(gaugeField, neighborPlus, neighborMinus,
                                       site, mu, nu, Fmunu);

            // Upper block: spin 0,1
            for (int s1 = 0; s1 < 2; s1++) {
                for (int s2 = 0; s2 < 2; s2++) {
                    gpu_complex_t sig = prefactor * sigmaMat[pairIdx * 16 + s1 * 4 + s2];
                    if (gpu_abs(sig) < 1e-15) continue;

                    for (int a = 0; a < GPU_NC; a++)
                        for (int b = 0; b < GPU_NC; b++) {
                            int row = s1 * GPU_NC + a;
                            int col = s2 * GPU_NC + b;
                            cloverUpper[site * 36 + row * 6 + col] += sig * Fmunu.m[a][b];
                        }
                }
            }

            // Lower block: spin 2,3
            for (int s1 = 2; s1 < 4; s1++) {
                for (int s2 = 2; s2 < 4; s2++) {
                    gpu_complex_t sig = prefactor * sigmaMat[pairIdx * 16 + s1 * 4 + s2];
                    if (gpu_abs(sig) < 1e-15) continue;

                    for (int a = 0; a < GPU_NC; a++)
                        for (int b = 0; b < GPU_NC; b++) {
                            int row = (s1 - 2) * GPU_NC + a;
                            int col = (s2 - 2) * GPU_NC + b;
                            cloverLower[site * 36 + row * 6 + col] += sig * Fmunu.m[a][b];
                        }
                }
            }
            pairIdx++;
        }
    }
}

//-----------------------------------------------
// Ensure clover field is computed and cached
//-----------------------------------------------
static void ensure_clover_field(gpu_real_t c_sw, int vol)
{
    if (gpuCloverFieldComputed) return;

    kernel_compute_clover_field<<<gpu_grid_size(vol), BLOCK_SIZE>>>(
        gpuState.d_gaugeField,
        gpuState.d_neighborPlus, gpuState.d_neighborMinus,
        gpuState.d_sigmaMat, c_sw,
        gpuState.d_cloverUpper, gpuState.d_cloverLower,
        vol);
    CUDA_CHECK_LAST();

    gpuCloverFieldComputed = true;
}

//-----------------------------------------------
// Apply cached clover term: result += A(x) * psi(x)
// Reads from precomputed 6x6 blocks
//-----------------------------------------------
__global__
void kernel_apply_cached_clover(
    const gpu_complex_t* __restrict__ cloverUpper,
    const gpu_complex_t* __restrict__ cloverLower,
    const ColorSpinorDev* __restrict__ psi,
    ColorSpinorDev* __restrict__ result,
    int vol)
{
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= vol) return;

    // Upper block: spin 0,1
    for (int s1 = 0; s1 < 2; s1++) {
        for (int a = 0; a < GPU_NC; a++) {
            int row = s1 * GPU_NC + a;
            gpu_complex_t sum(0.0, 0.0);
            for (int s2 = 0; s2 < 2; s2++) {
                for (int b = 0; b < GPU_NC; b++) {
                    int col = s2 * GPU_NC + b;
                    sum += cloverUpper[site * 36 + row * 6 + col] * psi[site].v[s2][b];
                }
            }
            result[site].v[s1][a] += sum;
        }
    }

    // Lower block: spin 2,3
    for (int s1 = 2; s1 < 4; s1++) {
        for (int a = 0; a < GPU_NC; a++) {
            int row = (s1 - 2) * GPU_NC + a;
            gpu_complex_t sum(0.0, 0.0);
            for (int s2 = 2; s2 < 4; s2++) {
                for (int b = 0; b < GPU_NC; b++) {
                    int col = (s2 - 2) * GPU_NC + b;
                    sum += cloverLower[site * 36 + row * 6 + col] * psi[site].v[s2][b];
                }
            }
            result[site].v[s1][a] += sum;
        }
    }
}

//-----------------------------------------------
// Clover Dirac operator: D_clover = D_wilson + A
// Uses cached clover field
//-----------------------------------------------
void gpu_clover_dirac(const ColorSpinorDev* psi, ColorSpinorDev* result,
                      gpu_real_t kappa, gpu_real_t wilsonR, int vol)
{
    // Ensure clover field is computed
    ensure_clover_field(gpuState.cloverCsw, vol);

    // Apply Wilson operator
    gpu_wilson_dirac(psi, result, kappa, wilsonR, vol);

    // Add cached clover term
    kernel_apply_cached_clover<<<gpu_grid_size(vol), BLOCK_SIZE>>>(
        gpuState.d_cloverUpper, gpuState.d_cloverLower,
        psi, result, vol);
    CUDA_CHECK_LAST();
}

void gpu_clover_dirac_dagger(const ColorSpinorDev* psi, ColorSpinorDev* result,
                             ColorSpinorDev* tmp1, ColorSpinorDev* tmp2,
                             gpu_real_t kappa, gpu_real_t wilsonR, int vol)
{
    // D_clover^dag = D_wilson^dag + A (A is Hermitian)
    ensure_clover_field(gpuState.cloverCsw, vol);

    gpu_wilson_dirac_dagger(psi, result, tmp1, tmp2, kappa, wilsonR, vol);

    kernel_apply_cached_clover<<<gpu_grid_size(vol), BLOCK_SIZE>>>(
        gpuState.d_cloverUpper, gpuState.d_cloverLower,
        psi, result, vol);
    CUDA_CHECK_LAST();
}
