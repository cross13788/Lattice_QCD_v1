//-----------------------------------------------
// HMC trajectory and driver on GPU
//
// All gauge field operations happen on device.
// Only scalars (deltaH, plaquette) transfer D2H.
// HMC reject uses D2D copy (no host roundtrip).
//-----------------------------------------------
#include "gpu_state.cuh"
#include "su3_device.cuh"
#include "colorspinor_device.cuh"
#include <cstdio>
#include <cmath>
#include <cstdlib>

//-----------------------------------------------
// Simple host-side RNG for Metropolis accept/reject
// (only generates one number per HMC trajectory,
// so quality doesn't matter much — LCG is fine)
//-----------------------------------------------
static unsigned long long hmc_rng_state = 42;
static double hmc_random_uniform()
{
    hmc_rng_state = hmc_rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(hmc_rng_state >> 11) / (double)(1ULL << 53);
}

// Forward declarations
void gpu_generate_momenta(int vol);
void gpu_generate_gaussian_spinor(ColorSpinorDev* xi, int vol);
void gpu_apply_dirac_dagger(const ColorSpinorDev* psi, ColorSpinorDev* result,
                            ColorSpinorDev* tmp1, ColorSpinorDev* tmp2,
                            gpu_real_t kappa, gpu_real_t wilsonR,
                            bool useClover, int vol);
gpu_real_t gpu_compute_kinetic_energy(int vol);
gpu_real_t gpu_compute_plaquette(int vol);
gpu_real_t gpu_compute_pseudofermion_action(gpu_real_t kappa, gpu_real_t wilsonR,
                                            bool useClover, int cgMaxIter,
                                            gpu_real_t cgTolerance, int vol);
void gpu_leapfrog_integrator(gpu_real_t beta, gpu_real_t kappa,
                             gpu_real_t wilsonR, bool useClover,
                             int cgMaxIter, gpu_real_t cgTolerance,
                             int nSteps, gpu_real_t stepSize, int vol);

//-----------------------------------------------
// Single HMC trajectory (all on GPU)
// Returns 1 if accepted, 0 if rejected
//-----------------------------------------------
int gpu_hmc_trajectory(gpu_real_t beta, gpu_real_t kappa, gpu_real_t wilsonR,
                       bool useClover, int cgMaxIter, gpu_real_t cgTolerance,
                       int nMDsteps, gpu_real_t mdStepSize,
                       gpu_real_t& deltaH, int vol)
{
    int nLinks = vol * 4;

    // Save gauge field (D2D copy)
    CUDA_CHECK(cudaMemcpy(gpuState.d_gaugeFieldSaved, gpuState.d_gaugeField,
                          nLinks * sizeof(SU3matrixDev), cudaMemcpyDeviceToDevice));

    // Generate momenta
    gpu_generate_momenta(vol);

    // Generate pseudofermion: phi = D† xi
    gpu_generate_gaussian_spinor(gpuState.d_tmp, vol);
    gpu_apply_dirac_dagger(gpuState.d_tmp, gpuState.d_phi,
                           gpuState.d_tmp2, gpuState.d_psi,
                           kappa, wilsonR, useClover, vol);

    // Initial Hamiltonian
    gpu_real_t T_init = gpu_compute_kinetic_energy(vol);
    gpu_real_t avgPlaq = gpu_compute_plaquette(vol);
    gpu_real_t S_G_init = 6.0 * beta * vol * (1.0 - avgPlaq);
    gpu_real_t S_PF_init = gpu_compute_pseudofermion_action(
        kappa, wilsonR, useClover, cgMaxIter, cgTolerance, vol);
    gpu_real_t H_init = T_init + S_G_init + S_PF_init;

    // Leapfrog integration
    gpu_leapfrog_integrator(beta, kappa, wilsonR, useClover,
                            cgMaxIter, cgTolerance,
                            nMDsteps, mdStepSize, vol);

    // Final Hamiltonian
    gpu_real_t T_final = gpu_compute_kinetic_energy(vol);
    avgPlaq = gpu_compute_plaquette(vol);
    gpu_real_t S_G_final = 6.0 * beta * vol * (1.0 - avgPlaq);
    gpu_real_t S_PF_final = gpu_compute_pseudofermion_action(
        kappa, wilsonR, useClover, cgMaxIter, cgTolerance, vol);
    gpu_real_t H_final = T_final + S_G_final + S_PF_final;

    deltaH = H_final - H_init;

    // Metropolis accept/reject
    int accepted = 0;
    if (deltaH <= 0.0) {
        accepted = 1;
    } else {
        if (hmc_random_uniform() < exp(-deltaH))
            accepted = 1;
    }

    if (!accepted) {
        // Reject: restore gauge field (D2D copy, no host roundtrip)
        CUDA_CHECK(cudaMemcpy(gpuState.d_gaugeField, gpuState.d_gaugeFieldSaved,
                              nLinks * sizeof(SU3matrixDev), cudaMemcpyDeviceToDevice));
    }

    return accepted;
}

//-----------------------------------------------
// HMC driver: thermalization + production
//-----------------------------------------------
void gpu_hmc_driver(gpu_real_t beta, gpu_real_t kappa, gpu_real_t wilsonR,
                    bool useClover, int cgMaxIter, gpu_real_t cgTolerance,
                    int nTherm, int nMDsteps, gpu_real_t mdStepSize,
                    int nHMCtrajectories, bool measurePolyakov,
                    bool measureWilsonFlow, bool measureCorrelators,
                    bool saveBinaryConfigs, int configSaveFrequency,
                    const char* outputDirectory, int vol)
{
    // Thermalization
    printf("===============================================\n");
    printf("  GPU HMC Thermalization: %d trajectories\n", nTherm);
    printf("  MD steps: %d, step size: %.4f\n", nMDsteps, mdStepSize);
    printf("  Kappa: %.6f\n", kappa);
    printf("===============================================\n\n");

    int acceptedTherm = 0;
    for (int traj = 1; traj <= nTherm; traj++) {
        gpu_real_t deltaH;
        int accepted = gpu_hmc_trajectory(beta, kappa, wilsonR, useClover,
                                          cgMaxIter, cgTolerance,
                                          nMDsteps, mdStepSize, deltaH, vol);
        acceptedTherm += accepted;

        if (traj % 10 == 0 || traj == 1) {
            gpu_real_t avgPlaq = gpu_compute_plaquette(vol);
            gpu_real_t acceptRate = (gpu_real_t)acceptedTherm / traj * 100.0;
            printf("  HMC therm %4d / %4d  <P>=%.6f  dH=%.4e  %s  (%.0f%% acc)\n",
                   traj, nTherm, avgPlaq, deltaH,
                   accepted ? "ACC" : "REJ", acceptRate);
        }
    }

    if (nTherm > 0)
        printf("\nThermalization complete: %.1f%% acceptance rate\n\n",
               (gpu_real_t)acceptedTherm / nTherm * 100.0);

    // Open output files
    char plaqFileName[512];
    snprintf(plaqFileName, sizeof(plaqFileName), "%s/plaquette.dat", outputDirectory);
    FILE* plaqFile = fopen(plaqFileName, "w");
    fprintf(plaqFile, "# traj  avg_plaquette  action_density  deltaH  accepted\n");

    // Production
    printf("===============================================\n");
    printf("  GPU HMC Production: %d trajectories\n", nHMCtrajectories);
    printf("===============================================\n\n");

    int acceptedProd = 0;
    for (int traj = 1; traj <= nHMCtrajectories; traj++) {
        gpu_real_t deltaH;
        int accepted = gpu_hmc_trajectory(beta, kappa, wilsonR, useClover,
                                          cgMaxIter, cgTolerance,
                                          nMDsteps, mdStepSize, deltaH, vol);
        acceptedProd += accepted;

        gpu_real_t avgPlaq = gpu_compute_plaquette(vol);
        gpu_real_t actionDensity = 6.0 * beta * (1.0 - avgPlaq);
        fprintf(plaqFile, "%6d  %.12f  %.12f  %16.8e  %d\n",
                traj, avgPlaq, actionDensity, deltaH, accepted);

        gpu_real_t acceptRate = (gpu_real_t)acceptedProd / traj * 100.0;
        printf("  HMC traj %4d / %4d  <P>=%.6f  dH=%+.4e  %s  (%.0f%% acc)\n",
               traj, nHMCtrajectories, avgPlaq, deltaH,
               accepted ? "ACC" : "REJ", acceptRate);

        if (traj % 10 == 0) fflush(plaqFile);
    }

    printf("\nHMC production complete: %.1f%% acceptance rate\n",
           (gpu_real_t)acceptedProd / nHMCtrajectories * 100.0);
    printf("Output saved to: %s/\n\n", outputDirectory);

    fclose(plaqFile);
}
