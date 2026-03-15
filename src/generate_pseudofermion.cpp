//-----------------------------------------------
// Generate pseudofermion field for HMC
//
// The pseudofermion trick: replace det(D†D) with
// a bosonic integral over φ:
//   det(D†D) = ∫ Dφ Dφ† exp(-φ† (D†D)^{-1} φ)
//
// To sample φ with the correct distribution, we
// generate Gaussian random ξ and set:
//   φ = D† ξ
//
// Then φ† (D†D)^{-1} φ = ξ† D (D†D)^{-1} D† ξ = ξ† ξ
// which is just a Gaussian weight for ξ.
//
// During the MD trajectory, φ is held fixed and
// the action S_PF = φ† (D†D)^{-1} φ depends on U
// through D[U].
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "colorspinor_module.hpp"
#include "interfaces_module.hpp"
#include <random>

// rng_local declared in su3_module.hpp

void generate_pseudofermion(const SU3matrix* gaugeField,
                            ColorSpinor* phi,
                            int vol)
{
//-----------------------------------------------
//   Generate Gaussian random spinor field ξ
//-----------------------------------------------
    ColorSpinor* xi = new ColorSpinor[vol];

    #pragma omp parallel
    {
        ensure_rng_seeded();
        std::normal_distribution<real_t> gauss(0.0, 1.0);

        #pragma omp for
        for (int site = 0; site < vol; site++) {
            for (int s = 0; s < ND; s++) {
                for (int c = 0; c < NC; c++) {
                    real_t re = gauss(rng_local) / std::sqrt(2.0);
                    real_t im = gauss(rng_local) / std::sqrt(2.0);
                    xi[site].v[s][c] = complex_t(re, im);
                }
            }
        }
    }

//-----------------------------------------------
//   φ = D† ξ  (uses clover operator when enabled)
//-----------------------------------------------
    apply_dirac_dagger(gaugeField, xi, phi, vol);

//-----------------------------------------------
//   Cleanup
//-----------------------------------------------
    delete[] xi;
}
