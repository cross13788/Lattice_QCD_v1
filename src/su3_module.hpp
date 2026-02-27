#pragma once
//-----------------------------------------------
// SU(3) matrix type and operations
//
// All operations are free functions operating on
// a plain struct (no OOP methods).
//-----------------------------------------------
#include "constants_module.hpp"
#include <random>

// Thread-local random number generator
// Defined in su3_module.cpp, accessible from any file
extern thread_local std::mt19937_64 rng_local;
void ensure_rng_seeded();

struct SU3matrix {
    complex_t m[3][3];
};

//-----------------------------------------------
// Basic operations
//-----------------------------------------------

// Set result to the 3x3 identity
void su3_identity(SU3matrix& result);

// Set all elements to zero
void su3_zero(SU3matrix& result);

// Copy: result = A
void su3_copy(const SU3matrix& A, SU3matrix& result);

// result = A * B
void su3_multiply(const SU3matrix& A, const SU3matrix& B, SU3matrix& result);

// Conjugate transpose: result = A^dagger
void su3_dagger(const SU3matrix& A, SU3matrix& result);

// Trace
complex_t su3_trace(const SU3matrix& A);

// Real part of trace
real_t su3_retrace(const SU3matrix& A);

// result = A + B
void su3_add(const SU3matrix& A, const SU3matrix& B, SU3matrix& result);

// result = A - B
void su3_subtract(const SU3matrix& A, const SU3matrix& B, SU3matrix& result);

// result = c * A (complex scalar multiply)
void su3_scale(complex_t c, const SU3matrix& A, SU3matrix& result);

// result += A (accumulate)
void su3_accumulate(const SU3matrix& A, SU3matrix& result);

// Determinant
complex_t su3_determinant(const SU3matrix& A);

//-----------------------------------------------
// SU(3) specific operations
//-----------------------------------------------

// Reunitarize: project back to SU(3) via modified Gram-Schmidt
void su3_reunitarize(SU3matrix& U);

// Random SU(3) matrix near identity (for Metropolis updates)
// epsilon controls how far from identity
void su3_random_near_identity(SU3matrix& result, real_t epsilon);

// Generate a random SU(3) matrix uniformly over the group
void su3_random_uniform(SU3matrix& result);

//-----------------------------------------------
// Heat bath operations (Cabibbo-Marinari method)
//-----------------------------------------------

// SU(2) heat bath step (Kennedy-Pendleton algorithm)
// Generates SU(2) element distributed as exp(beta_eff * Re Tr(a * W))
// where W is the 2x2 submatrix of the staple
void su2_heat_bath(real_t beta_eff, const complex_t W[2][2], complex_t a[2][2]);

// Update an SU(3) matrix using heat bath on a given SU(2) subgroup
// subgroup: 0 = (0,1), 1 = (0,2), 2 = (1,2)
void su3_heat_bath_subgroup(SU3matrix& U, const SU3matrix& staple,
                            real_t beta, int subgroup);

// Full Cabibbo-Marinari heat bath update of one link
// Cycles through all 3 SU(2) subgroups
void su3_heat_bath_update(SU3matrix& U, const SU3matrix& staple, real_t beta);

// Overrelaxation (microcanonical) update of one link
void su3_overrelaxation_update(SU3matrix& U, const SU3matrix& staple);

//-----------------------------------------------
// Debugging
//-----------------------------------------------

// Print matrix to stdout
void su3_print(const SU3matrix& A);
