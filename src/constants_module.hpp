#pragma once
//-----------------------------------------------
// Physical and mathematical constants for
// Lattice QCD simulations
//-----------------------------------------------
#include <complex>

using real_t = double;
using complex_t = std::complex<double>;

// Number of colors
constexpr int NC = 3;

// Mathematical constants
constexpr real_t PI = 3.14159265358979323846;
constexpr complex_t I_COMPLEX(0.0, 1.0);

// Small value for numerical safety
constexpr real_t SMALL_VALUE = 1.0e-35;
