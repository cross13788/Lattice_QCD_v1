//-----------------------------------------------
// Single HMC trajectory
//
// A complete HMC trajectory:
//   1. Save current gauge field
//   2. Generate conjugate momenta (Gaussian)
//   3. Generate pseudofermion field
//   4. Compute initial Hamiltonian
//   5. Integrate equations of motion (leapfrog)
//   6. Compute final Hamiltonian
//   7. Metropolis accept/reject based on delta H
//
// Returns 1 if accepted, 0 if rejected.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "lattice_module.hpp"
#include "colorspinor_module.hpp"
#include "interfaces_module.hpp"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <random>

// rng_local declared in su3_module.hpp

int hmc_trajectory(SU3matrix* gaugeField, real_t& deltaH)
{
    int nLinks = latticeVolume * 4;

//-----------------------------------------------
//   Allocate arrays
//-----------------------------------------------
    SU3matrix* gaugeFieldSaved = new SU3matrix[nLinks];
    SU3matrix* momentum        = new SU3matrix[nLinks];
    ColorSpinor* phi            = new ColorSpinor[latticeVolume];

//-----------------------------------------------
//   Step 1: Save current gauge field
//-----------------------------------------------
    std::memcpy(gaugeFieldSaved, gaugeField, nLinks * sizeof(SU3matrix));

//-----------------------------------------------
//   Step 2: Generate conjugate momenta
//-----------------------------------------------
    generate_conjugate_momenta(momentum, nLinks);

//-----------------------------------------------
//   Step 3: Generate pseudofermion field
//-----------------------------------------------
    generate_pseudofermion(gaugeField, phi, latticeVolume);

//-----------------------------------------------
//   Step 4: Compute initial Hamiltonian
//   H = T + S_G + S_PF
//-----------------------------------------------
    real_t T_init = compute_kinetic_energy(momentum, nLinks);

    real_t avgPlaquette;
    compute_plaquette(gaugeField, avgPlaquette);
    real_t S_G_init = 6.0 * beta * latticeVolume * (1.0 - avgPlaquette);

    real_t S_PF_init = compute_pseudofermion_action(gaugeField, phi, latticeVolume);

    real_t H_init = T_init + S_G_init + S_PF_init;

//-----------------------------------------------
//   Step 5: Leapfrog integration
//-----------------------------------------------
    leapfrog_integrator(gaugeField, momentum, phi,
                        nMDsteps, mdStepSize, latticeVolume);

//-----------------------------------------------
//   Step 6: Compute final Hamiltonian
//-----------------------------------------------
    real_t T_final = compute_kinetic_energy(momentum, nLinks);

    compute_plaquette(gaugeField, avgPlaquette);
    real_t S_G_final = 6.0 * beta * latticeVolume * (1.0 - avgPlaquette);

    real_t S_PF_final = compute_pseudofermion_action(gaugeField, phi, latticeVolume);

    real_t H_final = T_final + S_G_final + S_PF_final;

    deltaH = H_final - H_init;

//-----------------------------------------------
//   Step 7: Metropolis accept/reject
//-----------------------------------------------
    int accepted = 0;

    if (deltaH <= 0.0) {
        accepted = 1;
    } else {
        ensure_rng_seeded();
        std::uniform_real_distribution<real_t> uniform(0.0, 1.0);
        real_t r = uniform(rng_local);
        if (r < std::exp(-deltaH)) {
            accepted = 1;
        }
    }

    if (!accepted) {
        // Reject: restore saved gauge field
        std::memcpy(gaugeField, gaugeFieldSaved, nLinks * sizeof(SU3matrix));
    }

//-----------------------------------------------
//   Cleanup
//-----------------------------------------------
    delete[] gaugeFieldSaved;
    delete[] momentum;
    delete[] phi;

    return accepted;
}
