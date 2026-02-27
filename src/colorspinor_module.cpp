//-----------------------------------------------
// Color-spinor type and field operations
//
// Implementation of single-site and field-level
// operations for 4-component Dirac spinors with
// 3 color components.
//-----------------------------------------------
#include "colorspinor_module.hpp"
#include <cstring>

//===============================================
//  SINGLE-SITE OPERATIONS
//===============================================

void cs_zero(ColorSpinor& result)
{
    for (int s = 0; s < ND; s++)
        for (int c = 0; c < NC; c++)
            result.v[s][c] = complex_t(0.0, 0.0);
}

void cs_copy(const ColorSpinor& src, ColorSpinor& result)
{
    for (int s = 0; s < ND; s++)
        for (int c = 0; c < NC; c++)
            result.v[s][c] = src.v[s][c];
}

void cs_axpy(complex_t alpha, const ColorSpinor& x, const ColorSpinor& y,
             ColorSpinor& result)
{
    for (int s = 0; s < ND; s++)
        for (int c = 0; c < NC; c++)
            result.v[s][c] = alpha * x.v[s][c] + y.v[s][c];
}

void cs_scale(complex_t alpha, const ColorSpinor& x, ColorSpinor& result)
{
    for (int s = 0; s < ND; s++)
        for (int c = 0; c < NC; c++)
            result.v[s][c] = alpha * x.v[s][c];
}

void cs_accumulate(complex_t alpha, const ColorSpinor& x, ColorSpinor& result)
{
    for (int s = 0; s < ND; s++)
        for (int c = 0; c < NC; c++)
            result.v[s][c] += alpha * x.v[s][c];
}

complex_t cs_dot(const ColorSpinor& x, const ColorSpinor& y)
{
    complex_t sum(0.0, 0.0);
    for (int s = 0; s < ND; s++)
        for (int c = 0; c < NC; c++)
            sum += std::conj(x.v[s][c]) * y.v[s][c];
    return sum;
}

real_t cs_norm2(const ColorSpinor& x)
{
    real_t sum = 0.0;
    for (int s = 0; s < ND; s++)
        for (int c = 0; c < NC; c++)
            sum += std::norm(x.v[s][c]);
    return sum;
}

//===============================================
//  FIELD-LEVEL OPERATIONS (OpenMP)
//===============================================

void field_zero(ColorSpinor* field, int vol)
{
    #pragma omp parallel for
    for (int i = 0; i < vol; i++)
        cs_zero(field[i]);
}

void field_copy(const ColorSpinor* src, ColorSpinor* result, int vol)
{
    #pragma omp parallel for
    for (int i = 0; i < vol; i++)
        cs_copy(src[i], result[i]);
}

void field_axpy(complex_t alpha, const ColorSpinor* x, const ColorSpinor* y,
                ColorSpinor* result, int vol)
{
    #pragma omp parallel for
    for (int i = 0; i < vol; i++)
        cs_axpy(alpha, x[i], y[i], result[i]);
}

void field_scale(complex_t alpha, const ColorSpinor* x, ColorSpinor* result, int vol)
{
    #pragma omp parallel for
    for (int i = 0; i < vol; i++)
        cs_scale(alpha, x[i], result[i]);
}

void field_accumulate(complex_t alpha, const ColorSpinor* x, ColorSpinor* result, int vol)
{
    #pragma omp parallel for
    for (int i = 0; i < vol; i++)
        cs_accumulate(alpha, x[i], result[i]);
}

complex_t field_dot(const ColorSpinor* x, const ColorSpinor* y, int vol)
{
    real_t sumReal = 0.0;
    real_t sumImag = 0.0;

    #pragma omp parallel for reduction(+:sumReal, sumImag)
    for (int i = 0; i < vol; i++) {
        complex_t d = cs_dot(x[i], y[i]);
        sumReal += d.real();
        sumImag += d.imag();
    }

    return complex_t(sumReal, sumImag);
}

real_t field_norm2(const ColorSpinor* x, int vol)
{
    real_t sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < vol; i++)
        sum += cs_norm2(x[i]);

    return sum;
}
