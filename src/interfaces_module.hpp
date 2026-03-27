#pragma once
//-----------------------------------------------
// Forward declarations of all user-defined functions
//
// This header serves the same role as the Fortran
// interfaces_module: a single place declaring every
// function signature in the code. Include it in every
// .cpp file for compile-time checking.
//-----------------------------------------------
#include "constants_module.hpp"
#include "su3_module.hpp"
#include "colorspinor_module.hpp"

// Forward declaration for clover types
struct CloverField;
struct CloverBlock;

//===========================================================
//  INITIALIZATION & SETUP
//===========================================================
void read_input_file();
void set_parameter(const char* section, const char* key, const char* value);
void openmp_parallel_setup(int& numberProcessors, int& maxThreads);
void initialize_lattice_geometry();
void free_lattice_geometry();

//===========================================================
//  GAUGE FIELD OPERATIONS (Phase 1)
//===========================================================
void initialize_gauge_field(SU3matrix* gaugeField, const char* startType);
void compute_plaquette(const SU3matrix* gaugeField, real_t& avgPlaquette);
void compute_staple(const SU3matrix* gaugeField, int site, int mu, SU3matrix& staple);
void heat_bath_sweep(SU3matrix* gaugeField);
void overrelaxation_sweep(SU3matrix* gaugeField);
void metropolis_sweep(SU3matrix* gaugeField, real_t& acceptanceRate);
void gauge_update_combined(SU3matrix* gaugeField);
void thermalization(SU3matrix* gaugeField);
void generate_configurations(SU3matrix* gaugeField);

//===========================================================
//  MEASUREMENTS (Phase 1)
//===========================================================
void measure_plaquette(const SU3matrix* gaugeField, int configNum, FILE* outFile);
void wilson_loop(const SU3matrix* gaugeField, real_t* wilsonLoops, int rMax, int tMax);
void polyakov_loop(const SU3matrix* gaugeField, real_t& polyakovAvg);
void static_potential(const real_t* wilsonLoops, int rMax, int tMax, real_t* potential);
void jackknife_analysis(const real_t* data, int nData, real_t& mean, real_t& error);

//===========================================================
//  ACTION DENSITY & COOLING (Visualization)
//===========================================================
void compute_action_density(const SU3matrix* gaugeField, real_t* actionDensity);
void write_action_density(const real_t* actionDensity, int configNum,
                          int coolStep, const char* directory);
void ape_smearing_step(SU3matrix* gaugeField, real_t alpha);
void cooling_sweep(SU3matrix* gaugeField, real_t* actionDensity,
                   int configNum, int nCool, real_t alpha,
                   const char* directory);

//===========================================================
//  I/O (Phase 1)
//===========================================================
void write_gauge_config(const SU3matrix* gaugeField, int configNum, const char* directory);
void read_gauge_config(SU3matrix* gaugeField, int configNum, const char* directory);
void write_observables(int configNum, real_t plaquette, real_t polyakov,
                       const real_t* wilsonLoops, int rMax, int tMax,
                       const char* directory);

//===========================================================
//  WILSON FERMIONS (Phase 2)
//===========================================================
void initialize_gamma_matrices();

// Dirac operators
void wilson_dirac_operator(const SU3matrix* gaugeField,
                           const ColorSpinor* psi,
                           ColorSpinor* result,
                           int vol);

void wilson_dirac_dagger(const SU3matrix* gaugeField,
                         const ColorSpinor* psi,
                         ColorSpinor* result,
                         int vol);

// Even-odd preconditioning
void apply_Deo(const SU3matrix* gaugeField,
               const ColorSpinor* psi,
               ColorSpinor* result,
               int vol);

void apply_Doe(const SU3matrix* gaugeField,
               const ColorSpinor* psi,
               ColorSpinor* result,
               int vol);

void apply_preconditioned_operator(const SU3matrix* gaugeField,
                                    const ColorSpinor* psi,
                                    ColorSpinor* result,
                                    int vol);

void reconstruct_odd_sites(const SU3matrix* gaugeField,
                           const ColorSpinor* x_even,
                           const ColorSpinor* b,
                           ColorSpinor* x,
                           int vol);

// Solvers (return iteration count)
int cg_solver(const SU3matrix* gaugeField,
              const ColorSpinor* b,
              ColorSpinor* x,
              int vol);

int bicgstab_solver(const SU3matrix* gaugeField,
                    const ColorSpinor* b,
                    ColorSpinor* x,
                    int vol);

// Sources and propagators
void point_source(ColorSpinor* source, int vol,
                  int spin0, int color0,
                  int srcX, int srcY, int srcZ, int srcT);

void compute_propagator(const SU3matrix* gaugeField,
                        ColorSpinor* propagator,
                        int vol);

// Meson correlators
void meson_correlator(const ColorSpinor* propagator,
                      int channel,
                      real_t* corr,
                      int vol);

void effective_mass(const real_t* corr, real_t* meff, int nT_local);

void effective_mass_cosh(const real_t* corr, real_t* meff, int nT_local);

void write_correlators(int configNum, int channel,
                       const real_t* corr, const real_t* meff,
                       const char* directory);

//===========================================================
//  WILSON FLOW (Scale Setting)
//===========================================================
void wilson_flow_step(SU3matrix* gaugeField, real_t epsilon, int vol);
real_t wilson_flow_energy_plaquette(const SU3matrix* gaugeField, int vol);
void compute_field_strength_tensor(const SU3matrix* gaugeField,
                                   int site, int mu, int nu,
                                   SU3matrix& Fmunu);
real_t wilson_flow_energy_clover(const SU3matrix* gaugeField, int vol);
void wilson_flow_driver(SU3matrix* gaugeField, int vol);
real_t wilson_flow_reversibility_test(const SU3matrix* gaugeField, int vol);

//===========================================================
//  CLOVER IMPROVEMENT (Phase 2+)
//===========================================================

// Sigma matrices initialization
void initialize_sigma_matrices();

// Clover field computation
void compute_clover_field(const SU3matrix* gaugeField,
                          CloverField* cloverField, int vol);
void apply_clover_term(const CloverField* cloverField,
                       const ColorSpinor* psi,
                       ColorSpinor* result, int site);
void invert_clover_block(CloverBlock& block);
void compute_clover_inverse(const CloverField* cloverField,
                            CloverField* cloverFieldInv, int vol);

// Clover Dirac operators
void clover_dirac_operator(const SU3matrix* gaugeField,
                           const CloverField* cloverField,
                           const ColorSpinor* psi,
                           ColorSpinor* result, int vol);
void clover_dirac_dagger(const SU3matrix* gaugeField,
                         const CloverField* cloverField,
                         const ColorSpinor* psi,
                         ColorSpinor* result, int vol);

// Dispatch wrappers (check useClover, manage clover cache)
void apply_dirac(const SU3matrix* gaugeField,
                 const ColorSpinor* psi,
                 ColorSpinor* result, int vol);
void apply_dirac_dagger(const SU3matrix* gaugeField,
                        const ColorSpinor* psi,
                        ColorSpinor* result, int vol);
void invalidate_clover_cache();
void free_clover_cache();

// Clover even-odd preconditioning
void apply_clover_preconditioned_operator(
    const SU3matrix* gaugeField,
    const CloverField* cloverFieldInv,
    const ColorSpinor* psi,
    ColorSpinor* result, int vol);

// Clover fermion force for HMC
void compute_clover_force(const SU3matrix* gaugeField,
                          const ColorSpinor* X,
                          const ColorSpinor* Y,
                          SU3matrix* force, int vol);

// Sigma matrix access for clover force (from clover_term.cpp)
const complex_t* get_sigma_matrix(int mu, int nu);
int get_sigma_pair_index(int mu, int nu);

// Numerical gradient check for fermion force
void gradient_check(SU3matrix* gaugeField,
                    const ColorSpinor* phi, int vol);

// Clover cache invalidation (from clover_dirac_operator.cpp)
void invalidate_clover_cache();

//===========================================================
//  HMC — DYNAMICAL FERMIONS (Phase 3)
//===========================================================

// SU(3) Lie algebra operations
void su3_traceless_antiherm(const SU3matrix& A, SU3matrix& result);
void su3_exp(const SU3matrix& A, SU3matrix& result);
void su3_exp_update(real_t epsilon, const SU3matrix& H, SU3matrix& U);
real_t su3_algebra_norm2(const SU3matrix& A);

// Conjugate momenta
void generate_conjugate_momenta(SU3matrix* momentum, int nLinks);
real_t compute_kinetic_energy(const SU3matrix* momentum, int nLinks);

// Pseudofermions
void generate_pseudofermion(const SU3matrix* gaugeField,
                            ColorSpinor* phi, int vol);
real_t compute_pseudofermion_action(const SU3matrix* gaugeField,
                                    const ColorSpinor* phi, int vol);

// Forces
void compute_gauge_force(const SU3matrix* gaugeField,
                         SU3matrix* force, int vol);
void compute_fermion_force(const SU3matrix* gaugeField,
                           const ColorSpinor* phi,
                           SU3matrix* force, int vol);

// Integrator
void leapfrog_integrator(SU3matrix* gaugeField,
                         SU3matrix* momentum,
                         const ColorSpinor* phi,
                         int nSteps, real_t stepSize, int vol);

// HMC trajectory (returns 1 if accepted, 0 if rejected)
int hmc_trajectory(SU3matrix* gaugeField, real_t& deltaH);

// HMC driver (main loop)
void hmc_driver(SU3matrix* gaugeField);
