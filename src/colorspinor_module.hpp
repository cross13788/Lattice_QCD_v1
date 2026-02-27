#pragma once
//-----------------------------------------------
// Color-spinor type and field operations
//
// A ColorSpinor has 4 Dirac (spin) and 3 color
// components at a single lattice site.
//
// Field-level operations act on arrays of
// ColorSpinor of size latticeVolume, with OpenMP.
//-----------------------------------------------
#include "constants_module.hpp"

// Number of Dirac components
constexpr int ND = 4;

struct ColorSpinor {
    complex_t v[ND][NC];    // v[spin][color]
};

//-----------------------------------------------
// Single-site operations
//-----------------------------------------------

// Zero a spinor
void cs_zero(ColorSpinor& result);

// Copy: result = src
void cs_copy(const ColorSpinor& src, ColorSpinor& result);

// result = alpha * x + y  (single site)
void cs_axpy(complex_t alpha, const ColorSpinor& x, const ColorSpinor& y,
             ColorSpinor& result);

// result = alpha * x  (single site)
void cs_scale(complex_t alpha, const ColorSpinor& x, ColorSpinor& result);

// result += alpha * x  (single site accumulate)
void cs_accumulate(complex_t alpha, const ColorSpinor& x, ColorSpinor& result);

// Dot product: sum_{spin,color} conj(x) * y  (single site)
complex_t cs_dot(const ColorSpinor& x, const ColorSpinor& y);

// Squared norm: sum_{spin,color} |x|^2  (single site)
real_t cs_norm2(const ColorSpinor& x);

//-----------------------------------------------
// Field-level operations (arrays of size latticeVolume)
// These use OpenMP parallel for.
//-----------------------------------------------

// Zero entire field
void field_zero(ColorSpinor* field, int vol);

// Copy: result = src (entire field)
void field_copy(const ColorSpinor* src, ColorSpinor* result, int vol);

// result = alpha * x + y  (field)
void field_axpy(complex_t alpha, const ColorSpinor* x, const ColorSpinor* y,
                ColorSpinor* result, int vol);

// result = alpha * x  (field)
void field_scale(complex_t alpha, const ColorSpinor* x, ColorSpinor* result, int vol);

// result += alpha * x  (field accumulate)
void field_accumulate(complex_t alpha, const ColorSpinor* x, ColorSpinor* result, int vol);

// Dot product over entire lattice: sum_sites sum_{spin,color} conj(x) * y
complex_t field_dot(const ColorSpinor* x, const ColorSpinor* y, int vol);

// Squared norm over entire lattice
real_t field_norm2(const ColorSpinor* x, int vol);
