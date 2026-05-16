//-----------------------------------------------
// Clover (Sheikholeslami-Wohlert) fermion force on GPU.
//
// Direct port of the validated CPU compute_clover_force
// (src/clover_fermion_force.cpp, gradient-check verified).
// Derived from the EXACT clover Q-loops; per link occurrence:
//   V=U   : Phi += U*B*R*A  +  A^dag*R*B^dag*U^dag
//   V=U^d : Phi += -B*R*A*U^dag  -  U*A^dag*R*B^dag
// with R(z)_{dc} = sum_{ss'} sigma_{ss'}
//        [ X_{s'}(z)_d Y*_s(z)_c + Y_{s'}(z)_d X*_s(z)_c ],
// F^cl += (c_sw/32) * TA( i * sum Phi ).
//
// Scatter form: one thread per site z, atomicAdd into a
// per-link accumulator (multiple z touch the same link),
// then a finalize kernel applies TA(i*.) and the prefactor.
//-----------------------------------------------
#include "gpu_state.cuh"
#include "su3_device.cuh"
#include "colorspinor_device.cuh"

static inline __device__ int dev_shiftP(const int* nP, int s, int d) { return nP[s * 4 + d]; }
static inline __device__ int dev_shiftM(const int* nM, int s, int d) { return nM[s * 4 + d]; }

// sigma pair index for ordered (a<b): (0,1)0 (0,2)1 (0,3)2 (1,2)3 (1,3)4 (2,3)5
static inline __device__ int dev_sigma_pair(int a, int b)
{
    if (a == 0) return b - 1;
    if (a == 1) return b + 1;
    return 5;
}

struct DevLinkRef { int site, dir, dag; };

// Build the 4 links of clover leaf k of Q_{a,b}(z) (matches CPU build_leaf
// and dev_compute_field_strength leaf ordering).
static __device__ void dev_build_leaf(const int* nP, const int* nM,
                                       int z, int a, int b, int k,
                                       DevLinkRef L[4])
{
    int za  = dev_shiftP(nP, z, a);
    int zb  = dev_shiftP(nP, z, b);
    int zma = dev_shiftM(nM, z, a);
    int zmb = dev_shiftM(nM, z, b);

    if (k == 0) {
        L[0] = {z,  a, 0}; L[1] = {za, b, 0};
        L[2] = {zb, a, 1}; L[3] = {z,  b, 1};
    } else if (k == 1) {
        int zmapb = dev_shiftP(nP, zma, b);
        L[0] = {z,     b, 0}; L[1] = {zmapb, a, 1};
        L[2] = {zma,   b, 1}; L[3] = {zma,   a, 0};
    } else if (k == 2) {
        int zmamb = dev_shiftM(nM, zma, b);
        L[0] = {zma,   a, 1}; L[1] = {zmamb, b, 1};
        L[2] = {zmamb, a, 0}; L[3] = {zmb,   b, 0};
    } else {
        int zmbpa = dev_shiftP(nP, zmb, a);
        L[0] = {zmb,   b, 1}; L[1] = {zmb,   a, 0};
        L[2] = {zmbpa, b, 0}; L[3] = {z,     a, 1};
    }
}

static inline __device__ void dev_link_matrix(const SU3matrixDev* g,
                                               const DevLinkRef& L,
                                               SU3matrixDev& V)
{
    if (L.dag) dev_su3_dagger(g[L.site * 4 + L.dir], V);
    else       V = g[L.site * 4 + L.dir];
}

//-----------------------------------------------
// Accumulate clover-force contributions (one thread per site z).
//-----------------------------------------------
__global__
void kernel_clover_force_accumulate(
    const SU3matrixDev* __restrict__ gaugeField,
    const ColorSpinorDev* __restrict__ X,
    const ColorSpinorDev* __restrict__ Y,
    const gpu_complex_t* __restrict__ sigmaMat,
    const int* __restrict__ neighborPlus,
    const int* __restrict__ neighborMinus,
    SU3matrixDev* __restrict__ acc,
    int vol)
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    if (z >= vol) return;

    for (int a = 0; a < 4; a++) {
        for (int b = a + 1; b < 4; b++) {
            int pair = dev_sigma_pair(a, b);
            const gpu_complex_t* sig = &sigmaMat[pair * 16];

            // R(z)_{dc}
            SU3matrixDev R;
            dev_su3_zero(R);
            for (int s = 0; s < GPU_ND; s++) {
                for (int sp = 0; sp < GPU_ND; sp++) {
                    gpu_complex_t sg = sig[s * 4 + sp];
                    if (gpu_abs(sg) < 1e-15) continue;
                    for (int d = 0; d < GPU_NC; d++)
                        for (int c = 0; c < GPU_NC; c++)
                            R.m[d][c] += sg *
                                ( X[z].v[sp][d] * gpu_conj(Y[z].v[s][c])
                                + Y[z].v[sp][d] * gpu_conj(X[z].v[s][c]) );
                }
            }

            for (int k = 0; k < 4; k++) {
                DevLinkRef Lp[4];
                dev_build_leaf(neighborPlus, neighborMinus, z, a, b, k, Lp);

                SU3matrixDev V[4];
                for (int q = 0; q < 4; q++)
                    dev_link_matrix(gaugeField, Lp[q], V[q]);

                for (int i = 0; i < 4; i++) {
                    SU3matrixDev A, B, tmp;
                    dev_su3_identity(A);
                    for (int q = 0; q < i; q++) {
                        dev_su3_multiply(A, V[q], tmp); A = tmp;
                    }
                    dev_su3_identity(B);
                    for (int q = i + 1; q < 4; q++) {
                        dev_su3_multiply(B, V[q], tmp); B = tmp;
                    }

                    SU3matrixDev BRA, t2;
                    dev_su3_multiply(B, R, t2);
                    dev_su3_multiply(t2, A, BRA);

                    SU3matrixDev Ad, Bd, ARBd, t3;
                    dev_su3_dagger(A, Ad);
                    dev_su3_dagger(B, Bd);
                    dev_su3_multiply(Ad, R, t3);
                    dev_su3_multiply(t3, Bd, ARBd);

                    const DevLinkRef& Li = Lp[i];
                    const SU3matrixDev& U = gaugeField[Li.site * 4 + Li.dir];
                    SU3matrixDev Ud; dev_su3_dagger(U, Ud);

                    SU3matrixDev Phi, ta, tb;
                    if (Li.dag == 0) {
                        dev_su3_multiply(U, BRA, ta);    // U*B*R*A
                        dev_su3_multiply(ARBd, Ud, tb);  // A^d*R*B^d*U^d
                        for (int p = 0; p < GPU_NC; p++)
                            for (int qq = 0; qq < GPU_NC; qq++)
                                Phi.m[p][qq] = ta.m[p][qq] + tb.m[p][qq];
                    } else {
                        dev_su3_multiply(BRA, Ud, ta);   // B*R*A*U^d
                        dev_su3_multiply(U, ARBd, tb);   // U*A^d*R*B^d
                        for (int p = 0; p < GPU_NC; p++)
                            for (int qq = 0; qq < GPU_NC; qq++)
                                Phi.m[p][qq] = gpu_complex_t(0.0,0.0) - ta.m[p][qq] - tb.m[p][qq];
                    }

                    int li = Li.site * 4 + Li.dir;
                    for (int p = 0; p < GPU_NC; p++)
                        for (int qq = 0; qq < GPU_NC; qq++) {
                            atomicAdd(&acc[li].m[p][qq].re, Phi.m[p][qq].re);
                            atomicAdd(&acc[li].m[p][qq].im, Phi.m[p][qq].im);
                        }
                }
            }
        }
    }
}

//-----------------------------------------------
// Finalize: F^cl[link] += (c_sw/32) * TA( i * acc[link] )
//-----------------------------------------------
__global__
void kernel_clover_force_finalize(const SU3matrixDev* __restrict__ acc,
                                  SU3matrixDev* __restrict__ force,
                                  gpu_real_t C, int nLinks)
{
    int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= nLinks) return;

    SU3matrixDev accI;
    for (int p = 0; p < GPU_NC; p++)
        for (int q = 0; q < GPU_NC; q++)
            accI.m[p][q] = gpu_complex_t(0.0, 1.0) * acc[li].m[p][q];

    SU3matrixDev ta;
    dev_su3_traceless_antiherm(accI, ta);

    for (int p = 0; p < GPU_NC; p++)
        for (int q = 0; q < GPU_NC; q++)
            force[li].m[p][q] += gpu_complex_t(C, 0.0) * ta.m[p][q];
}

//-----------------------------------------------
// Add the clover fermion force into d_forceFermion.
// X = d_X = (D^dag D)^{-1} phi, Y = d_Y = D X (clover D),
// already computed by gpu_compute_fermion_force.
//-----------------------------------------------
void gpu_compute_clover_force(gpu_real_t c_sw, int vol)
{
    int nLinks = vol * 4;

    CUDA_CHECK(cudaMemset(gpuState.d_cloverForceAcc, 0,
                          (size_t)nLinks * sizeof(SU3matrixDev)));

    kernel_clover_force_accumulate<<<gpu_grid_size(vol), BLOCK_SIZE>>>(
        gpuState.d_gaugeField, gpuState.d_X, gpuState.d_Y,
        gpuState.d_sigmaMat,
        gpuState.d_neighborPlus, gpuState.d_neighborMinus,
        gpuState.d_cloverForceAcc, vol);
    CUDA_CHECK_LAST();

    kernel_clover_force_finalize<<<gpu_grid_size(nLinks), BLOCK_SIZE>>>(
        gpuState.d_cloverForceAcc, gpuState.d_forceFermion,
        (gpu_real_t)(c_sw / 32.0), nLinks);
    CUDA_CHECK_LAST();
}
