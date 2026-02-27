#pragma once
//-----------------------------------------------
// Euclidean gamma matrices (DeGrand-Rossi basis)
//
// Sparse representation: each row of gamma_mu
// has exactly one nonzero element. We store
// the column index and the complex value.
//
// gamma_mu[s] acts on spin index s:
//   (gamma_mu * psi)[s] = gammaVal[mu][s] * psi[gammaIdx[mu][s]]
//
// Gamma matrices:
//   gamma1 = ( 0    0   0  -i )    gamma2 = ( 0    0   0  -1 )
//            ( 0    0  -i   0 )             ( 0    0   1   0 )
//            ( 0    i   0   0 )             ( 0    1   0   0 )
//            ( i    0   0   0 )             (-1    0   0   0 )
//
//   gamma3 = ( 0    0  -i   0 )    gamma4 = ( 0    0   1   0 )
//            ( 0    0   0   i )             ( 0    0   0   1 )
//            ( i    0   0   0 )             ( 1    0   0   0 )
//            ( 0   -i   0   0 )             ( 0    1   0   0 )
//
//   gamma5 = gamma1*gamma2*gamma3*gamma4
//          = ( 1    0   0   0 )
//            ( 0    1   0   0 )
//            ( 0    0  -1   0 )
//            ( 0    0   0  -1 )
//-----------------------------------------------
#include "constants_module.hpp"

// gammaIdx[mu][s] = column index of nonzero element in row s
// gammaVal[mu][s] = complex value of that element
// mu: 0=gamma1, 1=gamma2, 2=gamma3, 3=gamma4, 4=gamma5
extern int gammaIdx[5][4];
extern complex_t gammaVal[5][4];

// Initialize the gamma matrix tables
void initialize_gamma_matrices();
