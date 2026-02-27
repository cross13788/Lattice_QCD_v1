//-----------------------------------------------
// Euclidean gamma matrices (DeGrand-Rossi basis)
//
// Initializes the sparse representation tables
// for gamma1..gamma4 and gamma5.
//-----------------------------------------------
#include "gamma_matrices.hpp"

//-----------------------------------------------
//   Global definitions
//-----------------------------------------------
int gammaIdx[5][4];
complex_t gammaVal[5][4];

void initialize_gamma_matrices()
{
//-----------------------------------------------
//   gamma1
//-----------------------------------------------
    gammaIdx[0][0] = 3;  gammaVal[0][0] = complex_t(0.0, -1.0);
    gammaIdx[0][1] = 2;  gammaVal[0][1] = complex_t(0.0, -1.0);
    gammaIdx[0][2] = 1;  gammaVal[0][2] = complex_t(0.0,  1.0);
    gammaIdx[0][3] = 0;  gammaVal[0][3] = complex_t(0.0,  1.0);

//-----------------------------------------------
//   gamma2
//-----------------------------------------------
    gammaIdx[1][0] = 3;  gammaVal[1][0] = complex_t(-1.0, 0.0);
    gammaIdx[1][1] = 2;  gammaVal[1][1] = complex_t( 1.0, 0.0);
    gammaIdx[1][2] = 1;  gammaVal[1][2] = complex_t( 1.0, 0.0);
    gammaIdx[1][3] = 0;  gammaVal[1][3] = complex_t(-1.0, 0.0);

//-----------------------------------------------
//   gamma3
//-----------------------------------------------
    gammaIdx[2][0] = 2;  gammaVal[2][0] = complex_t(0.0, -1.0);
    gammaIdx[2][1] = 3;  gammaVal[2][1] = complex_t(0.0,  1.0);
    gammaIdx[2][2] = 0;  gammaVal[2][2] = complex_t(0.0,  1.0);
    gammaIdx[2][3] = 1;  gammaVal[2][3] = complex_t(0.0, -1.0);

//-----------------------------------------------
//   gamma4 (temporal direction)
//-----------------------------------------------
    gammaIdx[3][0] = 2;  gammaVal[3][0] = complex_t( 1.0, 0.0);
    gammaIdx[3][1] = 3;  gammaVal[3][1] = complex_t( 1.0, 0.0);
    gammaIdx[3][2] = 0;  gammaVal[3][2] = complex_t( 1.0, 0.0);
    gammaIdx[3][3] = 1;  gammaVal[3][3] = complex_t( 1.0, 0.0);

//-----------------------------------------------
//   gamma5 = gamma1 * gamma2 * gamma3 * gamma4
//-----------------------------------------------
    gammaIdx[4][0] = 0;  gammaVal[4][0] = complex_t( 1.0, 0.0);
    gammaIdx[4][1] = 1;  gammaVal[4][1] = complex_t( 1.0, 0.0);
    gammaIdx[4][2] = 2;  gammaVal[4][2] = complex_t(-1.0, 0.0);
    gammaIdx[4][3] = 3;  gammaVal[4][3] = complex_t(-1.0, 0.0);
}
