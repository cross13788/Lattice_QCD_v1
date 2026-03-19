//-----------------------------------------------
// Leapfrog (Verlet) integrator on GPU
//
// Orchestrates GPU force kernels and exp_update
// kernels. All data stays on device.
//-----------------------------------------------
#include "gpu_state.cuh"
#include <cstdio>

// Forward declarations
void gpu_compute_gauge_force(gpu_real_t beta, int vol);
void gpu_compute_fermion_force(gpu_real_t kappa, gpu_real_t wilsonR,
                               bool useClover, int cgMaxIter,
                               gpu_real_t cgTolerance, int vol);
void gpu_momentum_update(gpu_real_t stepSize, int nLinks);
void gpu_gauge_exp_update(gpu_real_t stepSize, int nLinks);
void gpu_invalidate_clover_cache();

void gpu_leapfrog_integrator(gpu_real_t beta, gpu_real_t kappa,
                             gpu_real_t wilsonR, bool useClover,
                             int cgMaxIter, gpu_real_t cgTolerance,
                             int nSteps, gpu_real_t stepSize, int vol)
{
    int nLinks = vol * 4;

    // Step 1: half-step momentum update
    gpu_compute_gauge_force(beta, vol);
    gpu_compute_fermion_force(kappa, wilsonR, useClover, cgMaxIter, cgTolerance, vol);
    gpu_momentum_update(0.5 * stepSize, nLinks);

    // Steps 2-3: full steps
    for (int step = 0; step < nSteps; step++) {
        // Full gauge field update
        gpu_gauge_exp_update(stepSize, nLinks);

        if (useClover) gpu_invalidate_clover_cache();

        // Full momentum update (except last step)
        if (step < nSteps - 1) {
            gpu_compute_gauge_force(beta, vol);
            gpu_compute_fermion_force(kappa, wilsonR, useClover, cgMaxIter, cgTolerance, vol);
            gpu_momentum_update(stepSize, nLinks);
        }
    }

    // Step 4: final half-step momentum
    gpu_compute_gauge_force(beta, vol);
    gpu_compute_fermion_force(kappa, wilsonR, useClover, cgMaxIter, cgTolerance, vol);
    gpu_momentum_update(0.5 * stepSize, nLinks);
}
