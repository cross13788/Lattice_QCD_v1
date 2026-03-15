//-----------------------------------------------
// Leapfrog (Verlet) integrator for HMC
//
// Integrates Hamilton's equations:
//   dU/dt = H * U           (gauge field update)
//   dH/dt = -F              (momentum update)
//
// where F = dS/dU is the total force (gauge + fermion).
//
// Leapfrog scheme for N steps of size epsilon:
//   1. H(epsilon/2) = H(0) - (epsilon/2) * F(U(0))
//   2. For n = 0, ..., N-2:
//      a. U((n+1)*epsilon) = exp(epsilon * H) * U(n*epsilon)
//      b. H((n+3/2)*epsilon) = H - epsilon * F(U)
//   3. U(N*epsilon) = exp(epsilon * H) * U
//   4. H(N*epsilon) = H - (epsilon/2) * F(U)
//
// This is a symplectic, time-reversible integrator
// with O(epsilon^2) errors per trajectory.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "lattice_module.hpp"
#include "colorspinor_module.hpp"
#include "interfaces_module.hpp"
#include <cstdio>

void leapfrog_integrator(SU3matrix* gaugeField,
                         SU3matrix* momentum,
                         const ColorSpinor* phi,
                         int nSteps,
                         real_t stepSize,
                         int vol)
{
    int nLinks = vol * 4;

//-----------------------------------------------
//   Allocate force arrays
//-----------------------------------------------
    SU3matrix* forceGauge  = new SU3matrix[nLinks];
    SU3matrix* forceFermion = new SU3matrix[nLinks];

//-----------------------------------------------
//   Step 1: Half-step momentum update
//   H(epsilon/2) = H(0) - (epsilon/2) * F(U(0))
//-----------------------------------------------
    compute_gauge_force(gaugeField, forceGauge, vol);
    compute_fermion_force(gaugeField, phi, forceFermion, vol);

    real_t halfStep = 0.5 * stepSize;

    #pragma omp parallel for
    for (int idx = 0; idx < nLinks; idx++) {
        for (int i = 0; i < NC; i++)
            for (int j = 0; j < NC; j++)
                momentum[idx].m[i][j] -= halfStep * (
                    forceGauge[idx].m[i][j] + forceFermion[idx].m[i][j]
                );
    }

//-----------------------------------------------
//   Steps 2-3: Full steps
//-----------------------------------------------

    for (int step = 0; step < nSteps; step++) {

        // Full gauge field update:
        //   U = exp(epsilon * H) * U
        #pragma omp parallel for
        for (int idx = 0; idx < nLinks; idx++) {
            su3_exp_update(stepSize, momentum[idx], gaugeField[idx]);
        }

        // Invalidate clover cache after gauge field change
        if (useClover) invalidate_clover_cache();

        // Full momentum update (except last step = half)
        if (step < nSteps - 1) {
            compute_gauge_force(gaugeField, forceGauge, vol);
            compute_fermion_force(gaugeField, phi, forceFermion, vol);

            #pragma omp parallel for
            for (int idx = 0; idx < nLinks; idx++) {
                for (int i = 0; i < NC; i++)
                    for (int j = 0; j < NC; j++)
                        momentum[idx].m[i][j] -= stepSize * (
                            forceGauge[idx].m[i][j] + forceFermion[idx].m[i][j]
                        );
            }
        }
    }

//-----------------------------------------------
//   Step 4: Final half-step momentum update
//-----------------------------------------------
    compute_gauge_force(gaugeField, forceGauge, vol);
    compute_fermion_force(gaugeField, phi, forceFermion, vol);

    #pragma omp parallel for
    for (int idx = 0; idx < nLinks; idx++) {
        for (int i = 0; i < NC; i++)
            for (int j = 0; j < NC; j++)
                momentum[idx].m[i][j] -= halfStep * (
                    forceGauge[idx].m[i][j] + forceFermion[idx].m[i][j]
                );
    }

//-----------------------------------------------
//   Cleanup
//-----------------------------------------------
    delete[] forceGauge;
    delete[] forceFermion;
}
