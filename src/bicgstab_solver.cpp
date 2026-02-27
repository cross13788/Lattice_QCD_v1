//-----------------------------------------------
// BiCGstab solver for D_W * x = b
//
// Bi-Conjugate Gradient Stabilized method for
// solving the non-symmetric Wilson-Dirac system
// directly (without forming normal equations).
//
// Typically converges faster than CG on normal
// equations, with roughly half the operator
// applications per effective iteration.
//
// Returns the number of iterations used.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "colorspinor_module.hpp"
#include "interfaces_module.hpp"
#include <cstdio>
#include <cmath>
#include <random>

int bicgstab_solver(const SU3matrix* gaugeField,
                    const ColorSpinor* b,
                    ColorSpinor* x,
                    int vol)
{
//-----------------------------------------------
//   Allocate working vectors
//-----------------------------------------------
    ColorSpinor* r    = new ColorSpinor[vol];   // residual
    ColorSpinor* r0   = new ColorSpinor[vol];   // shadow residual (fixed)
    ColorSpinor* p    = new ColorSpinor[vol];   // search direction
    ColorSpinor* v    = new ColorSpinor[vol];   // A * p
    ColorSpinor* s    = new ColorSpinor[vol];   // second residual
    ColorSpinor* t    = new ColorSpinor[vol];   // A * s

//-----------------------------------------------
//   Initial residual: r = b - A*x, start x = 0
//-----------------------------------------------
    field_zero(x, vol);
    field_copy(b, r, vol);

    // Use a random shadow residual to avoid breakdown
    // when source has special structure (e.g. point source)
    {
        std::mt19937_64 rng(12345);
        std::normal_distribution<real_t> dist(0.0, 1.0);
        for (int i = 0; i < vol; i++)
            for (int s = 0; s < ND; s++)
                for (int c = 0; c < NC; c++)
                    r0[i].v[s][c] = complex_t(dist(rng), dist(rng));
    }

    field_copy(r, p, vol);

    real_t bnorm2 = field_norm2(b, vol);
    if (bnorm2 < SMALL_VALUE) {
        delete[] r; delete[] r0; delete[] p; delete[] v; delete[] s; delete[] t;
        return 0;
    }

    complex_t rho_old(1.0, 0.0);
    complex_t alpha_bic(1.0, 0.0);
    complex_t omega(1.0, 0.0);

    // rho = (r0, r)
    complex_t rho = field_dot(r0, r, vol);

    if (verboseOutput) {
        printf("  BiCGstab: initial |r|^2 = %.6e, |b|^2 = %.6e\n",
               field_norm2(r, vol), bnorm2);
    }

//-----------------------------------------------
//   BiCGstab iteration loop
//-----------------------------------------------
    int iter;
    for (iter = 0; iter < cgMaxIter; iter++) {

        // v = A * p
        wilson_dirac_operator(gaugeField, p, v, vol);

        // alpha = rho / (r0, v)
        complex_t r0v = field_dot(r0, v, vol);
        if (std::abs(r0v) < SMALL_VALUE) {
            printf("  WARNING: BiCGstab breakdown: (r0, v) ~ 0\n");
            break;
        }
        alpha_bic = rho / r0v;

        // s = r - alpha * v
        field_copy(r, s, vol);
        complex_t minalpha = -alpha_bic;
        field_accumulate(minalpha, v, s, vol);

        // Check if s is small enough
        real_t snorm2 = field_norm2(s, vol);
        if (std::sqrt(snorm2 / bnorm2) < cgTolerance) {
            // x = x + alpha * p
            field_accumulate(alpha_bic, p, x, vol);
            iter++;
            if (verboseOutput) {
                printf("  BiCGstab converged (early) in %d iterations: |s|/|b| = %.6e\n",
                       iter, std::sqrt(snorm2 / bnorm2));
            }
            break;
        }

        // t = A * s
        wilson_dirac_operator(gaugeField, s, t, vol);

        // omega = (t, s) / (t, t)
        complex_t ts = field_dot(t, s, vol);
        real_t tt = field_norm2(t, vol);
        if (tt < SMALL_VALUE) {
            printf("  WARNING: BiCGstab breakdown: |t|^2 ~ 0\n");
            break;
        }
        omega = ts / complex_t(tt, 0.0);

        // x = x + alpha * p + omega * s
        field_accumulate(alpha_bic, p, x, vol);
        field_accumulate(omega, s, x, vol);

        // r = s - omega * t
        field_copy(s, r, vol);
        complex_t minomega = -omega;
        field_accumulate(minomega, t, r, vol);

        // Check convergence
        real_t rnorm2 = field_norm2(r, vol);
        real_t relResidual = std::sqrt(rnorm2 / bnorm2);

        if (verboseOutput && (iter + 1) % 100 == 0) {
            printf("  BiCGstab iter %5d: |r|/|b| = %.6e\n", iter + 1, relResidual);
        }

        if (relResidual < cgTolerance) {
            iter++;
            if (verboseOutput) {
                printf("  BiCGstab converged in %d iterations: |r|/|b| = %.6e\n",
                       iter, relResidual);
            }
            break;
        }

        // rho_new = (r0, r)
        complex_t rho_new = field_dot(r0, r, vol);
        if (std::abs(rho_new) < SMALL_VALUE) {
            printf("  WARNING: BiCGstab breakdown: rho ~ 0\n");
            break;
        }

        // beta = (rho_new / rho) * (alpha / omega)
        complex_t betaBic = (rho_new / rho) * (alpha_bic / omega);

        // p = r + beta * (p - omega * v)
        #pragma omp parallel for
        for (int i = 0; i < vol; i++) {
            for (int sp = 0; sp < ND; sp++)
                for (int c = 0; c < NC; c++)
                    p[i].v[sp][c] = r[i].v[sp][c] +
                        betaBic * (p[i].v[sp][c] - omega * v[i].v[sp][c]);
        }

        rho = rho_new;
    }

    if (iter >= cgMaxIter) {
        real_t finalRes = std::sqrt(field_norm2(r, vol) / bnorm2);
        printf("  WARNING: BiCGstab did not converge in %d iterations. |r|/|b| = %.6e\n",
               cgMaxIter, finalRes);
    }

//-----------------------------------------------
//   Cleanup
//-----------------------------------------------
    delete[] r;
    delete[] r0;
    delete[] p;
    delete[] v;
    delete[] s;
    delete[] t;

    return iter;
}
