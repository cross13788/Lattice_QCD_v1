//-----------------------------------------------
// Wilson-Dirac dagger operator: D^dag_W * psi = result
//
// D^dag_W = gamma5 * D_W * gamma5
//
// This is implemented by:
//   1. Apply gamma5 to the input
//   2. Apply D_W
//   3. Apply gamma5 to the output
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "colorspinor_module.hpp"
#include "gamma_matrices.hpp"
#include "su3_module.hpp"
#include "interfaces_module.hpp"

void wilson_dirac_dagger(const SU3matrix* gaugeField,
                         const ColorSpinor* psi,
                         ColorSpinor* result,
                         int vol)
{
//-----------------------------------------------
//   Allocate temporary fields
//-----------------------------------------------
    ColorSpinor* tmp1 = new ColorSpinor[vol];
    ColorSpinor* tmp2 = new ColorSpinor[vol];

//-----------------------------------------------
//   Step 1: tmp1 = gamma5 * psi
//-----------------------------------------------
    #pragma omp parallel for
    for (int site = 0; site < vol; site++) {
        for (int s = 0; s < ND; s++) {
            int gs = gammaIdx[4][s];
            complex_t gv = gammaVal[4][s];
            for (int c = 0; c < NC; c++) {
                tmp1[site].v[s][c] = gv * psi[site].v[gs][c];
            }
        }
    }

//-----------------------------------------------
//   Step 2: tmp2 = D_W * tmp1
//-----------------------------------------------
    wilson_dirac_operator(gaugeField, tmp1, tmp2, vol);

//-----------------------------------------------
//   Step 3: result = gamma5 * tmp2
//-----------------------------------------------
    #pragma omp parallel for
    for (int site = 0; site < vol; site++) {
        for (int s = 0; s < ND; s++) {
            int gs = gammaIdx[4][s];
            complex_t gv = gammaVal[4][s];
            for (int c = 0; c < NC; c++) {
                result[site].v[s][c] = gv * tmp2[site].v[gs][c];
            }
        }
    }

//-----------------------------------------------
//   Cleanup
//-----------------------------------------------
    delete[] tmp1;
    delete[] tmp2;
}
