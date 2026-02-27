//-----------------------------------------------
// Conjugate momentum for HMC
//
// The conjugate momenta H_mu(x) are traceless
// anti-Hermitian matrices drawn from a Gaussian
// distribution: P(H) ~ exp(-1/2 Tr(H^2))
//
// For su(3), this means generating 8 independent
// Gaussian random numbers (one per Gell-Mann
// generator). We construct the matrix directly
// and project to ensure exact tracelessness.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "interfaces_module.hpp"
#include <random>

// rng_local declared in su3_module.hpp

//-----------------------------------------------
// Generate Gaussian-distributed conjugate momenta
// for all links.
//
// momentum array has same layout as gaugeField:
//   momentum[site * 4 + mu]
//-----------------------------------------------
void generate_conjugate_momenta(SU3matrix* momentum, int nLinks)
{
    std::normal_distribution<real_t> gauss(0.0, 1.0);

    #pragma omp parallel
    {
        ensure_rng_seeded();
        std::normal_distribution<real_t> gaussLocal(0.0, 1.0);

        #pragma omp for
        for (int idx = 0; idx < nLinks; idx++) {
            SU3matrix& H = momentum[idx];

            // Generate a random Hermitian matrix, then multiply by i
            // to get anti-Hermitian.
            //
            // The traceless anti-Hermitian matrix has 8 real parameters.
            // We use the direct construction:
            //   H_ij = (a_ij + i*b_ij) for i < j  (upper triangle)
            //   H_ji = -(a_ij - i*b_ij) = -conj(H_ij)
            //   H_ii = i*d_i (purely imaginary diagonal)
            //   with sum d_i = 0 (traceless)

            // Off-diagonal elements
            for (int i = 0; i < NC; i++) {
                for (int j = i + 1; j < NC; j++) {
                    real_t a = gaussLocal(rng_local) / std::sqrt(2.0);
                    real_t b = gaussLocal(rng_local) / std::sqrt(2.0);
                    H.m[i][j] = complex_t(a, b);
                    H.m[j][i] = complex_t(-a, b);   // -conj(H_ij)
                }
            }

            // Diagonal: traceless anti-Hermitian means purely imaginary
            // with sum = 0. Generate NC-1 independent values.
            real_t d[NC];
            real_t dsum = 0.0;
            for (int i = 0; i < NC - 1; i++) {
                d[i] = gaussLocal(rng_local) / std::sqrt(2.0);
                dsum += d[i];
            }
            d[NC - 1] = -dsum;

            for (int i = 0; i < NC; i++) {
                H.m[i][i] = complex_t(0.0, d[i]);
            }
        }
    }
}

//-----------------------------------------------
// Compute kinetic energy:
//   T = (1/2) sum_{x,mu} Tr(H_mu(x)^2)
//
// For anti-Hermitian H: Tr(H^2) = -sum |H_ij|^2
// so T = -(1/2) sum Tr(H^2) = (1/2) sum |H_ij|^2
//-----------------------------------------------
real_t compute_kinetic_energy(const SU3matrix* momentum, int nLinks)
{
    real_t kinetic = 0.0;

    #pragma omp parallel for reduction(+:kinetic)
    for (int idx = 0; idx < nLinks; idx++) {
        kinetic += su3_algebra_norm2(momentum[idx]);
    }

    return 0.5 * kinetic;
}
