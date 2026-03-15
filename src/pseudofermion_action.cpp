//-----------------------------------------------
// Pseudofermion action for HMC
//
// S_PF = φ† (D†D)^{-1} φ = X† (D†D) X
//      where X = (D†D)^{-1} φ is obtained by CG.
//
// More simply: S_PF = φ† X where D†D X = φ,
// i.e., S_PF = Re(φ, X) where X solves the
// normal equations.
//
// This requires one CG solve per evaluation.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "colorspinor_module.hpp"
#include "interfaces_module.hpp"
#include <cstdio>

real_t compute_pseudofermion_action(const SU3matrix* gaugeField,
                                    const ColorSpinor* phi,
                                    int vol)
{
//-----------------------------------------------
//   Solve D†D X = φ using CG
//   (CG on normal equations: solve D†D X = φ)
//-----------------------------------------------
    ColorSpinor* X = new ColorSpinor[vol];

    // We need to solve D†D X = phi
    // The cg_solver solves D†D x = D†b, so we'd need to
    // factor out the D† on the RHS. Instead, let's solve
    // D x = phi using BiCGstab, then compute X = D^{-1} phi.
    // But we actually need X = (D†D)^{-1} phi.
    //
    // Use CG directly on D†D X = phi:
    // This requires a dedicated CG that applies D†D as the operator.
    // Our cg_solver solves via normal equations: it applies D and D†
    // separately. We can use it by treating phi as the "b" in
    // D†D X = D† b where b is such that D† b = phi.
    //
    // Simpler approach: solve D x1 = phi with BiCGstab to get x1 = D^{-1} phi,
    // then solve D† x2 = phi to get x2 = (D†)^{-1} phi,
    // then S_PF = phi† * (D†D)^{-1} phi = (D^{-1} phi)† * (D†)^{-1} phi
    //           = x1† D† (D†)^{-1} phi... no, this gets complicated.
    //
    // Direct approach: implement CG for D†D X = phi.
    // CG needs: A = D†D (Hermitian positive-definite).
    // r = phi - D†D X, with X=0 initially, r = phi.
    // Standard CG iterations.

    ColorSpinor* r   = new ColorSpinor[vol];
    ColorSpinor* p   = new ColorSpinor[vol];
    ColorSpinor* Ap  = new ColorSpinor[vol];
    ColorSpinor* tmp = new ColorSpinor[vol];

    // Initial: X = 0, r = phi, p = phi
    field_zero(X, vol);
    field_copy(phi, r, vol);
    field_copy(phi, p, vol);

    real_t rr = field_norm2(r, vol);
    real_t rr0 = rr;

    for (int iter = 0; iter < cgMaxIter; iter++) {
        // Ap = D†D p (uses clover operator when enabled)
        apply_dirac(gaugeField, p, tmp, vol);
        apply_dirac_dagger(gaugeField, tmp, Ap, vol);

        // alpha = (r,r) / (p, Ap)
        complex_t pAp = field_dot(p, Ap, vol);
        real_t alpha = rr / pAp.real();

        // X = X + alpha * p
        field_accumulate(complex_t(alpha, 0.0), p, X, vol);

        // r = r - alpha * Ap
        field_accumulate(complex_t(-alpha, 0.0), Ap, r, vol);

        real_t rr_new = field_norm2(r, vol);

        if (std::sqrt(rr_new / rr0) < cgTolerance) break;

        real_t betaCG = rr_new / rr;

        // p = r + beta * p
        #pragma omp parallel for
        for (int i = 0; i < vol; i++)
            for (int s = 0; s < ND; s++)
                for (int c = 0; c < NC; c++)
                    p[i].v[s][c] = r[i].v[s][c] + betaCG * p[i].v[s][c];

        rr = rr_new;
    }

//-----------------------------------------------
//   S_PF = Re(phi† X) = Re sum_site phi*(site) . X(site)
//-----------------------------------------------
    complex_t action = field_dot(phi, X, vol);

    delete[] X;
    delete[] r;
    delete[] p;
    delete[] Ap;
    delete[] tmp;

    return action.real();
}
