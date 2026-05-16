#pragma once
//-----------------------------------------------
// Clover (Sheikholeslami-Wohlert) term data structures
//
// The clover term adds O(a)-improvement to Wilson fermions:
//   D_clover = D_wilson + c_sw * (i/4) * sigma_{mu,nu} * F_{mu,nu}
//
// In the DeGrand-Rossi basis, sigma_{mu,nu} is block-diagonal
// with two 2x2 blocks. Combined with color (NC=3), the clover
// matrix at each site consists of two Hermitian 6x6 blocks:
//   upper: spin components 0,1 with color
//   lower: spin components 2,3 with color
//-----------------------------------------------
#include "constants_module.hpp"
#include "su3_module.hpp"
#include "colorspinor_module.hpp"

// Block size: 2 (spin) x NC (color) = 6
constexpr int CLOVER_BLOCK_SIZE = 2 * NC;

struct CloverBlock {
    complex_t m[CLOVER_BLOCK_SIZE][CLOVER_BLOCK_SIZE];
};

struct CloverField {
    CloverBlock upper;  // spin 0,1
    CloverBlock lower;  // spin 2,3
};

//-----------------------------------------------
// Diagnostic isolation selectors for debugging the
// clover fermion force against the numerical gradient
// check. Read once from the environment:
//   CLOVER_DIAG_PLANE : -1 (all, default) or sigma
//                       pair index 0..5 — restricts
//                       BOTH the clover field (hence
//                       S_PF / numerical gradient) and
//                       the analytic clover force to a
//                       single (mu,nu) plane.
//   CLOVER_DIAG_LEAF  : -1 (all, default) or 0..3 —
//                       restricts the analytic clover
//                       force to a single site-leaf
//                       (for term-by-term inspection).
// Production runs leave both unset => -1 => no effect.
//-----------------------------------------------
int clover_diag_plane();
int clover_diag_leaf();
