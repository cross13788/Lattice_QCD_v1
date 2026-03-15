//-----------------------------------------------
// Clover-improved Wilson-Dirac operator:
//
//   D_clover = D_wilson + A(x)
//
// where A(x) is the clover term (diagonal in site space).
// When c_sw = 0 this reduces to the standard Wilson operator.
//
// Also provides D_clover^dagger and dispatch wrappers
// that check useClover to route to the appropriate operator.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "lattice_module.hpp"
#include "colorspinor_module.hpp"
#include "gamma_matrices.hpp"
#include "clover_module.hpp"
#include "interfaces_module.hpp"
#include <cstdio>

//-----------------------------------------------
// Cached clover field for dispatch wrappers.
// Recomputed when gauge field changes (HMC).
//-----------------------------------------------
static CloverField* cachedCloverField = nullptr;
static int cachedCloverVol = 0;
static bool cloverFieldValid = false;

void invalidate_clover_cache()
{
    cloverFieldValid = false;
}

static void ensure_clover_field(const SU3matrix* gaugeField, int vol)
{
    if (cloverFieldValid && cachedCloverVol == vol && cachedCloverField != nullptr)
        return;

    if (cachedCloverField == nullptr || cachedCloverVol != vol) {
        delete[] cachedCloverField;
        cachedCloverField = new CloverField[vol];
        cachedCloverVol = vol;
    }

    compute_clover_field(gaugeField, cachedCloverField, vol);
    cloverFieldValid = true;
}

//-----------------------------------------------
// D_clover * psi = D_wilson * psi + A(x) * psi
//-----------------------------------------------
void clover_dirac_operator(const SU3matrix* gaugeField,
                           const CloverField* cloverField,
                           const ColorSpinor* psi,
                           ColorSpinor* result,
                           int vol)
{
    // First apply Wilson operator
    wilson_dirac_operator(gaugeField, psi, result, vol);

    // Add clover term
    #pragma omp parallel for
    for (int site = 0; site < vol; site++) {
        apply_clover_term(cloverField, psi, result, site);
    }
}

//-----------------------------------------------
// D_clover^dagger * psi
//
// D_clover^dag = D_wilson^dag + A^dag(x)
// Since A is Hermitian (sigma_{mu,nu} * F_{mu,nu} is anti-Hermitian,
// times i makes it Hermitian, times c_sw * i/4 makes it Hermitian),
// we have A^dag = A. So D_clover^dag = D_wilson^dag + A.
//-----------------------------------------------
void clover_dirac_dagger(const SU3matrix* gaugeField,
                         const CloverField* cloverField,
                         const ColorSpinor* psi,
                         ColorSpinor* result,
                         int vol)
{
    // First apply Wilson dagger
    wilson_dirac_dagger(gaugeField, psi, result, vol);

    // Add clover term (same as forward since A is Hermitian)
    #pragma omp parallel for
    for (int site = 0; site < vol; site++) {
        apply_clover_term(cloverField, psi, result, site);
    }
}

//-----------------------------------------------
// Dispatch wrappers: check useClover and route
// to the appropriate operator. These manage the
// clover field cache internally.
//
// Solvers and HMC call these instead of
// wilson_dirac_operator/dagger directly.
//-----------------------------------------------
void apply_dirac(const SU3matrix* gaugeField,
                 const ColorSpinor* psi,
                 ColorSpinor* result,
                 int vol)
{
    if (useClover) {
        ensure_clover_field(gaugeField, vol);
        clover_dirac_operator(gaugeField, cachedCloverField, psi, result, vol);
    } else {
        wilson_dirac_operator(gaugeField, psi, result, vol);
    }
}

void apply_dirac_dagger(const SU3matrix* gaugeField,
                        const ColorSpinor* psi,
                        ColorSpinor* result,
                        int vol)
{
    if (useClover) {
        ensure_clover_field(gaugeField, vol);
        clover_dirac_dagger(gaugeField, cachedCloverField, psi, result, vol);
    } else {
        wilson_dirac_dagger(gaugeField, psi, result, vol);
    }
}

//-----------------------------------------------
// Free the cached clover field (called at cleanup)
//-----------------------------------------------
void free_clover_cache()
{
    delete[] cachedCloverField;
    cachedCloverField = nullptr;
    cachedCloverVol = 0;
    cloverFieldValid = false;
}
