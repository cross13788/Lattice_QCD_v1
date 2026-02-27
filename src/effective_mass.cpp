//-----------------------------------------------
// Effective mass extraction from correlator C(t)
//
// Two methods:
//   1. Log ratio: m_eff(t) = log(C(t) / C(t+1))
//      Valid when C(t) ~ exp(-m*t) dominates.
//
//   2. Cosh ratio: solve
//      C(t)/C(t+1) = cosh(m*(t - T/2)) / cosh(m*(t+1 - T/2))
//      Better for periodic BC where
//      C(t) ~ cosh(m*(t - T/2)).
//
// The effective mass should show a plateau at
// intermediate t values, giving the ground-state
// mass in the channel.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "interfaces_module.hpp"
#include <cstdio>
#include <cmath>

void effective_mass(const real_t* corr, real_t* meff, int nT_local)
{
//-----------------------------------------------
//   Log ratio method
//-----------------------------------------------
    for (int t = 0; t < nT_local - 1; t++) {
        if (corr[t] > 0.0 && corr[t + 1] > 0.0) {
            meff[t] = std::log(corr[t] / corr[t + 1]);
        } else if (corr[t] < 0.0 && corr[t + 1] < 0.0) {
            // Both negative (can happen for non-pion channels)
            meff[t] = std::log(std::abs(corr[t]) / std::abs(corr[t + 1]));
        } else {
            // Sign change or zero — cannot extract mass
            meff[t] = 0.0;
        }
    }

    // Last time slice: no forward neighbor
    meff[nT_local - 1] = 0.0;
}

//-----------------------------------------------
// Cosh effective mass (better for periodic BC)
//
// Uses Newton's method to solve
//   cosh(m*(t-T/2)) / cosh(m*(t+1-T/2)) = ratio
// for m.
//-----------------------------------------------
void effective_mass_cosh(const real_t* corr, real_t* meff, int nT_local)
{
    real_t halfT = nT_local / 2.0;

    for (int t = 0; t < nT_local - 1; t++) {
        if (std::abs(corr[t + 1]) < SMALL_VALUE) {
            meff[t] = 0.0;
            continue;
        }

        real_t ratio = corr[t] / corr[t + 1];
        real_t tShift = t - halfT;
        real_t tShift1 = t + 1.0 - halfT;

        // Initial guess from log ratio
        real_t m;
        if (ratio > 0.0) {
            m = std::log(std::abs(ratio));
        } else {
            meff[t] = 0.0;
            continue;
        }

        // Newton iterations
        for (int iter = 0; iter < 50; iter++) {
            real_t coshT  = std::cosh(m * tShift);
            real_t coshT1 = std::cosh(m * tShift1);
            real_t sinhT  = std::sinh(m * tShift);
            real_t sinhT1 = std::sinh(m * tShift1);

            real_t f  = coshT / coshT1 - ratio;
            real_t fp = (tShift * sinhT * coshT1 - tShift1 * coshT * sinhT1)
                        / (coshT1 * coshT1);

            if (std::abs(fp) < SMALL_VALUE) break;

            real_t dm = -f / fp;
            m += dm;

            if (std::abs(dm) < 1.0e-12) break;
        }

        meff[t] = (m > 0.0) ? m : 0.0;
    }

    meff[nT_local - 1] = 0.0;
}
