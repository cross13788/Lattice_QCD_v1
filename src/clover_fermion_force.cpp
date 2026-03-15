//-----------------------------------------------
// Clover fermion force for HMC
//
// NOTE: The full clover fermion force involves the
// derivative of F_{mu,nu} through the 4 clover leaves,
// contracted with sigma matrices and solution vectors.
// This is one of the most complex pieces in lattice
// QCD — see e.g. DeGrand & DeTar Ch. 8 or the MILC
// code for reference implementations.
//
// Status: NOT YET IMPLEMENTED.
// The quenched clover (spectroscopy on fixed gauge
// backgrounds) works correctly without this force.
// For dynamical clover HMC, this force must be properly
// implemented before production use.
//
// When called, this prints a one-time warning and
// contributes zero force, which means clover HMC
// trajectories will have incorrect dynamics and
// large |deltaH|.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "lattice_module.hpp"
#include "colorspinor_module.hpp"
#include "gamma_matrices.hpp"
#include "clover_module.hpp"
#include "interfaces_module.hpp"
#include <cstdio>

static bool warned = false;

void compute_clover_force(const SU3matrix* gaugeField,
                          const ColorSpinor* X,
                          const ColorSpinor* Y,
                          SU3matrix* force,
                          int vol)
{
    if (!warned) {
        printf("\n  WARNING: Clover fermion force not yet implemented.\n");
        printf("  Clover HMC trajectories will have incorrect dynamics.\n");
        printf("  Use quenched clover (spectroscopy) for correct results.\n\n");
        warned = true;
    }
    // Force contribution is zero — dynamics will be wrong for HMC.
    // Quenched clover (no force needed) is unaffected.
}
