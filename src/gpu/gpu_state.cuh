#pragma once
//-----------------------------------------------
// GPUState: global struct holding all device
// pointers for the GPU build.
//
// Allocated once at startup, freed at cleanup.
// The gauge field lives on GPU throughout the
// simulation; only transferred to host for I/O.
//-----------------------------------------------
#include "gpu_common.cuh"
#include "su3_device.cuh"
#include "colorspinor_device.cuh"

struct GPUState {
    // Device info
    int deviceId;
    size_t totalMemory;

    // Lattice geometry on device
    int* d_neighborPlus;
    int* d_neighborMinus;
    int* d_siteParity;           // precomputed parity per site

    // Gamma matrices on device
    int* d_gammaIdx;             // [5][4] flattened
    gpu_complex_t* d_gammaVal;   // [5][4] flattened

    // Gauge field on device: vol * 4 SU3matrixDev
    SU3matrixDev* d_gaugeField;
    SU3matrixDev* d_gaugeFieldSaved;  // for HMC reject

    // Momentum field (HMC): vol * 4 SU3matrixDev
    SU3matrixDev* d_momentum;

    // Force arrays (HMC): vol * 4 SU3matrixDev
    SU3matrixDev* d_forceGauge;
    SU3matrixDev* d_forceFermion;

    // Solver vectors (allocated once, reused)
    ColorSpinorDev* d_psi;       // input spinor
    ColorSpinorDev* d_result;    // output spinor
    ColorSpinorDev* d_r;         // CG residual
    ColorSpinorDev* d_p;         // CG search direction
    ColorSpinorDev* d_Ap;        // CG: A*p
    ColorSpinorDev* d_tmp;       // temporary
    ColorSpinorDev* d_tmp2;      // temporary 2

    // BiCGstab extra vectors
    ColorSpinorDev* d_r0;        // shadow residual
    ColorSpinorDev* d_s;         // BiCGstab s
    ColorSpinorDev* d_t;         // BiCGstab t

    // Pseudofermion field (HMC)
    ColorSpinorDev* d_phi;

    // Fermion force solution vectors
    ColorSpinorDev* d_X;         // (D†D)^{-1} phi
    ColorSpinorDev* d_Y;         // D * X

    // Reduction buffer
    gpu_real_t* d_reductionBuf;
    gpu_real_t* h_reductionBuf;  // pinned host buffer
    int reductionBufSize;

    // cuRAND states for heat bath
    curandState* d_randStates;
    int nRandStates;

    // Wilson flow workspace: 3 force arrays
    SU3matrixDev* d_Z0;
    SU3matrixDev* d_Z1;
    SU3matrixDev* d_Z2;

    // Clover fields
    // (stored as arrays of 2 CloverBlockDev per site)
    gpu_complex_t* d_cloverUpper;   // vol * 6 * 6
    gpu_complex_t* d_cloverLower;   // vol * 6 * 6
    gpu_complex_t* d_cloverInvUpper;
    gpu_complex_t* d_cloverInvLower;

    // Sigma matrices (for clover term)
    gpu_complex_t* d_sigmaMat;      // [6][4][4] flattened

    // Action density workspace
    gpu_real_t* d_actionDensity;

    // Lattice parameters (cached on host for convenience)
    int vol;
    int nLinks;

    // Flags
    bool solverAllocated;
    bool hmcAllocated;
    bool flowAllocated;
    bool cloverAllocated;
    bool randAllocated;
};

// Global GPU state instance
extern GPUState gpuState;

//-----------------------------------------------
// GPU setup / cleanup functions
//-----------------------------------------------
void gpu_init(int vol);
void gpu_free();

// Upload/download gauge field
void gpu_upload_gauge(const void* hostGaugeField, int nLinks);
void gpu_download_gauge(void* hostGaugeField, int nLinks);

// Upload geometry and gamma matrices
void gpu_upload_geometry(const int* neighborPlus, const int* neighborMinus,
                         const int* siteParity, int vol);
void gpu_upload_gamma(const int* gammaIdx, const void* gammaVal);

// Allocate solver workspace
void gpu_alloc_solver(int vol);

// Allocate HMC workspace
void gpu_alloc_hmc(int vol);

// Allocate Wilson flow workspace
void gpu_alloc_flow(int vol);

// Allocate cuRAND states
void gpu_alloc_rand(int nStates, int seed);

// Allocate clover workspace
void gpu_alloc_clover(int vol);

// Reduction helper: sum d_reductionBuf[0..nBlocks-1] on host
gpu_real_t gpu_reduce_partial(int nBlocks);

// Invalidate cached clover field (call after gauge field changes)
void gpu_invalidate_clover_field();
