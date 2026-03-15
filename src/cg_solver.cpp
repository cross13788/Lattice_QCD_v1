//-----------------------------------------------
// Conjugate Gradient solver for D^dag D x = D^dag b
//
// Solves the normal equations for the Wilson-Dirac
// operator. The operator D^dag D is Hermitian and
// positive-definite, so CG is guaranteed to converge.
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

int cg_solver(const SU3matrix* gaugeField,
              const ColorSpinor* b,
              ColorSpinor* x,
              int vol)
{
//-----------------------------------------------
//   Allocate working vectors
//-----------------------------------------------
    ColorSpinor* r     = new ColorSpinor[vol];  // residual
    ColorSpinor* p     = new ColorSpinor[vol];  // search direction
    ColorSpinor* Ap    = new ColorSpinor[vol];  // A * p  (A = D^dag D)
    ColorSpinor* tmp   = new ColorSpinor[vol];  // temporary for D application
    ColorSpinor* Ddagb = new ColorSpinor[vol];  // D^dag b (RHS of normal eqn)

//-----------------------------------------------
//   Compute RHS: Ddagb = D^dag * b
//-----------------------------------------------
    apply_dirac_dagger(gaugeField, b, Ddagb, vol);

//-----------------------------------------------
//   Initial residual: r = Ddagb - A*x
//   Start with x = 0
//-----------------------------------------------
    field_zero(x, vol);
    field_copy(Ddagb, r, vol);
    field_copy(r, p, vol);

    real_t rr = field_norm2(r, vol);
    real_t rr0 = rr;
    real_t bnorm2 = field_norm2(Ddagb, vol);

    if (bnorm2 < SMALL_VALUE) {
        // Zero RHS, solution is zero
        delete[] r; delete[] p; delete[] Ap; delete[] tmp; delete[] Ddagb;
        return 0;
    }

    if (verboseOutput) {
        printf("  CG: initial |r|^2 = %.6e, |b|^2 = %.6e\n", rr, bnorm2);
    }

//-----------------------------------------------
//   CG iteration loop
//-----------------------------------------------
    int iter;
    for (iter = 0; iter < cgMaxIter; iter++) {

        // Ap = D^dag D * p
        apply_dirac(gaugeField, p, tmp, vol);
        apply_dirac_dagger(gaugeField, tmp, Ap, vol);

        // alpha = (r, r) / (p, Ap)
        complex_t pAp = field_dot(p, Ap, vol);
        real_t alpha = rr / pAp.real();

        // x = x + alpha * p
        complex_t calpha(alpha, 0.0);
        field_accumulate(calpha, p, x, vol);

        // r = r - alpha * Ap
        complex_t minalpha(-alpha, 0.0);
        field_accumulate(minalpha, Ap, r, vol);

        // Check convergence
        real_t rr_new = field_norm2(r, vol);
        real_t relResidual = std::sqrt(rr_new / bnorm2);

        if (verboseOutput && (iter + 1) % 100 == 0) {
            printf("  CG iter %5d: |r|/|b| = %.6e\n", iter + 1, relResidual);
        }

        if (relResidual < cgTolerance) {
            iter++;
            if (verboseOutput) {
                printf("  CG converged in %d iterations: |r|/|b| = %.6e\n",
                       iter, relResidual);
            }
            break;
        }

        // beta = rr_new / rr_old
        real_t betaCG = rr_new / rr;

        // p = r + beta * p
        #pragma omp parallel for
        for (int i = 0; i < vol; i++) {
            for (int s = 0; s < ND; s++)
                for (int c = 0; c < NC; c++)
                    p[i].v[s][c] = r[i].v[s][c] + betaCG * p[i].v[s][c];
        }

        rr = rr_new;
    }

    if (iter >= cgMaxIter) {
        real_t finalRes = std::sqrt(field_norm2(r, vol) / bnorm2);
        printf("  WARNING: CG did not converge in %d iterations. |r|/|b| = %.6e\n",
               cgMaxIter, finalRes);
    }

//-----------------------------------------------
//   Cleanup
//-----------------------------------------------
    delete[] r;
    delete[] p;
    delete[] Ap;
    delete[] tmp;
    delete[] Ddagb;

    return iter;
}
