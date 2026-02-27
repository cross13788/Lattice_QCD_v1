//-----------------------------------------------
// Metropolis sweep over all lattice links
//
// For each link U(x,mu):
// 1. Generate random SU(3) X near identity
// 2. Propose U' = X * U
// 3. Compute DeltaS = -(beta/NC) * Re Tr[(U'-U)*R]
// 4. Accept with probability min(1, exp(-DeltaS))
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "lattice_module.hpp"
#include "interfaces_module.hpp"
#include <cmath>
#include <random>

void metropolis_sweep(SU3matrix* gaugeField, real_t& acceptanceRate)
{
//-----------------------------------------------
//   Local variables
//-----------------------------------------------
    int nAccepted = 0;
    int nTotal = 0;

//-----------------------------------------------
//   Loop over even sites, then odd sites
//-----------------------------------------------
    for (int parity = 0; parity < 2; parity++) {
        // Cannot trivially parallelize Metropolis with shared acceptance counter
        // But we can parallelize within each parity since staples don't overlap
        #pragma omp parallel for reduction(+:nAccepted,nTotal)
        for (int site = 0; site < latticeVolume; site++) {
            if (site_parity(site) != parity) continue;

            for (int mu = 0; mu < 4; mu++) {
                nTotal++;

                // Compute staple
                SU3matrix staple;
                compute_staple(gaugeField, site, mu, staple);

                // Current link
                SU3matrix& U = gaugeField[site * 4 + mu];

                // Generate random SU(3) near identity
                SU3matrix X;
                su3_random_near_identity(X, metropolisEpsilon);

                // Propose U' = X * U
                SU3matrix Uprime;
                su3_multiply(X, U, Uprime);

                // DeltaS = -(beta/NC) * Re Tr[(U' - U) * staple]
                SU3matrix diff, product;
                su3_subtract(Uprime, U, diff);
                su3_multiply(diff, staple, product);
                real_t deltaS = -(beta / NC) * su3_retrace(product);

                // Metropolis accept/reject
                // Thread-local RNG via static thread_local
                static thread_local std::mt19937_64 rng(std::random_device{}());
                static thread_local std::uniform_real_distribution<real_t> dist(0.0, 1.0);

                if (deltaS < 0.0 || dist(rng) < std::exp(-deltaS)) {
                    su3_copy(Uprime, U);
                    nAccepted++;
                }
            }
        }
    }

    acceptanceRate = (nTotal > 0) ? (real_t)nAccepted / nTotal : 0.0;
}
