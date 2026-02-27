//-----------------------------------------------
// Gauge force for HMC molecular dynamics
//
// The Wilson gauge action at link U_mu(x) involves:
//   S_G = -(beta/NC) * Re Tr(U_mu * V_mu) + const
//
// where V_mu(x) is the sum of staples. Taking the
// derivative w.r.t. anti-Hermitian direction X:
//   dS_G/deps = -(beta/NC) Re Tr(X * U * V)
//             = -Re Tr(X * F_G)
//
// This gives:
//   F_G = (beta/NC) * [U * V]_TA
//
// Convention: dS = -Re Tr(X * F) for the leapfrog
// momentum update H -= (eps/2) * F.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "lattice_module.hpp"
#include "interfaces_module.hpp"

void compute_gauge_force(const SU3matrix* gaugeField,
                         SU3matrix* force,
                         int vol)
{
    // The gauge action contribution for link U_mu(x) is:
    //   S_G = -(beta/NC) * Re Tr(U_mu * V_mu)
    // where V_mu is the staple sum.
    //
    // F = (beta/NC) * [U * V]_TA
    real_t coeff = beta / (real_t)NC;

    #pragma omp parallel for
    for (int site = 0; site < vol; site++) {
        for (int mu = 0; mu < 4; mu++) {
            int idx = site * 4 + mu;

            // Get staple sum V_mu(x)
            SU3matrix staple;
            compute_staple(gaugeField, site, mu, staple);

            // Compute U * V (NOT V†!)
            SU3matrix UV;
            su3_multiply(gaugeField[idx], staple, UV);

            // Project to traceless anti-Hermitian and scale
            SU3matrix forceTA;
            su3_traceless_antiherm(UV, forceTA);

            // F = (beta/(2*NC)) * [U V]_TA
            su3_scale(complex_t(coeff, 0.0), forceTA, force[idx]);
        }
    }
}
