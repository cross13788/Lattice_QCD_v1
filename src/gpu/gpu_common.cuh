#pragma once
//-----------------------------------------------
// Common CUDA utilities for Lattice QCD GPU code
//
// Error-checking macros, kernel launch helpers,
// and a lightweight device complex type.
//
// We use our own complex struct instead of
// thrust::complex<double> because CUDA 12.6's
// thrust has broken common_type SFINAE that
// causes infinite template recursion with GCC 15.
// Our struct has the same binary layout (two
// doubles) so cudaMemcpy with std::complex works.
//-----------------------------------------------
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

//-----------------------------------------------
// CUDA error checking macro
//-----------------------------------------------
#define CUDA_CHECK(call) do {                                          \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while(0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

//-----------------------------------------------
// Lightweight device-safe complex type
//
// Binary-compatible with std::complex<double>
// and thrust::complex<double> (two contiguous
// doubles: real, imag). All operators defined
// explicitly — no template metaprogramming.
//-----------------------------------------------
using gpu_real_t = double;

struct gpu_complex_t {
    gpu_real_t re, im;

    __device__ __host__ __forceinline__
    gpu_complex_t() : re(0.0), im(0.0) {}

    __device__ __host__ __forceinline__
    gpu_complex_t(gpu_real_t r) : re(r), im(0.0) {}

    __device__ __host__ __forceinline__
    gpu_complex_t(gpu_real_t r, gpu_real_t i) : re(r), im(i) {}

    __device__ __host__ __forceinline__
    gpu_real_t real() const { return re; }

    __device__ __host__ __forceinline__
    gpu_real_t imag() const { return im; }

    // complex + complex
    __device__ __host__ __forceinline__
    gpu_complex_t operator+(const gpu_complex_t& b) const {
        return gpu_complex_t(re + b.re, im + b.im);
    }

    // complex - complex
    __device__ __host__ __forceinline__
    gpu_complex_t operator-(const gpu_complex_t& b) const {
        return gpu_complex_t(re - b.re, im - b.im);
    }

    // unary minus
    __device__ __host__ __forceinline__
    gpu_complex_t operator-() const {
        return gpu_complex_t(-re, -im);
    }

    // complex * complex
    __device__ __host__ __forceinline__
    gpu_complex_t operator*(const gpu_complex_t& b) const {
        return gpu_complex_t(re * b.re - im * b.im,
                             re * b.im + im * b.re);
    }

    // complex / complex
    __device__ __host__ __forceinline__
    gpu_complex_t operator/(const gpu_complex_t& b) const {
        gpu_real_t denom = b.re * b.re + b.im * b.im;
        return gpu_complex_t((re * b.re + im * b.im) / denom,
                             (im * b.re - re * b.im) / denom);
    }

    // compound assignment
    __device__ __host__ __forceinline__
    gpu_complex_t& operator+=(const gpu_complex_t& b) {
        re += b.re; im += b.im; return *this;
    }

    __device__ __host__ __forceinline__
    gpu_complex_t& operator-=(const gpu_complex_t& b) {
        re -= b.re; im -= b.im; return *this;
    }

    __device__ __host__ __forceinline__
    gpu_complex_t& operator*=(const gpu_complex_t& b) {
        gpu_real_t r = re * b.re - im * b.im;
        gpu_real_t i = re * b.im + im * b.re;
        re = r; im = i; return *this;
    }

    __device__ __host__ __forceinline__
    gpu_complex_t& operator/=(const gpu_complex_t& b) {
        *this = *this / b; return *this;
    }

    // mixed complex-real arithmetic (avoids thrust's broken SFINAE)
    __device__ __host__ __forceinline__
    gpu_complex_t operator*(gpu_real_t b) const {
        return gpu_complex_t(re * b, im * b);
    }

    __device__ __host__ __forceinline__
    gpu_complex_t operator/(gpu_real_t b) const {
        return gpu_complex_t(re / b, im / b);
    }

    __device__ __host__ __forceinline__
    gpu_complex_t operator+(gpu_real_t b) const {
        return gpu_complex_t(re + b, im);
    }

    __device__ __host__ __forceinline__
    gpu_complex_t operator-(gpu_real_t b) const {
        return gpu_complex_t(re - b, im);
    }

    __device__ __host__ __forceinline__
    gpu_complex_t& operator+=(gpu_real_t b) { re += b; return *this; }

    __device__ __host__ __forceinline__
    gpu_complex_t& operator-=(gpu_real_t b) { re -= b; return *this; }

    __device__ __host__ __forceinline__
    gpu_complex_t& operator*=(gpu_real_t b) { re *= b; im *= b; return *this; }

    __device__ __host__ __forceinline__
    gpu_complex_t& operator/=(gpu_real_t b) { re /= b; im /= b; return *this; }
};

// real * complex (non-member for commutativity)
__device__ __host__ __forceinline__
gpu_complex_t operator*(gpu_real_t a, const gpu_complex_t& b) {
    return gpu_complex_t(a * b.re, a * b.im);
}

__device__ __host__ __forceinline__
gpu_complex_t operator+(gpu_real_t a, const gpu_complex_t& b) {
    return gpu_complex_t(a + b.re, b.im);
}

__device__ __host__ __forceinline__
gpu_complex_t operator-(gpu_real_t a, const gpu_complex_t& b) {
    return gpu_complex_t(a - b.re, -b.im);
}

//-----------------------------------------------
// Complex math functions (replacements for
// thrust::conj, thrust::norm, thrust::abs)
//-----------------------------------------------
__device__ __host__ __forceinline__
gpu_complex_t gpu_conj(const gpu_complex_t& z) {
    return gpu_complex_t(z.re, -z.im);
}

// |z|^2 = re^2 + im^2  (squared magnitude)
__device__ __host__ __forceinline__
gpu_real_t gpu_norm(const gpu_complex_t& z) {
    return z.re * z.re + z.im * z.im;
}

// |z| = sqrt(re^2 + im^2)
__device__ __host__ __forceinline__
gpu_real_t gpu_abs(const gpu_complex_t& z) {
    return sqrt(z.re * z.re + z.im * z.im);
}

//-----------------------------------------------
// Constants matching CPU code
//-----------------------------------------------
constexpr int GPU_NC = 3;
constexpr int GPU_ND = 4;
constexpr gpu_real_t GPU_PI = 3.14159265358979323846;
constexpr int CLOVER_BLOCK_DIM = 2 * GPU_NC;   // 6

//-----------------------------------------------
// Default kernel launch parameters
//-----------------------------------------------
constexpr int BLOCK_SIZE     = 256;
constexpr int BLOCK_SIZE_RED = 256;   // for reductions
constexpr int BLOCK_SIZE_EXP = 128;   // for su3_exp (high register pressure)

//-----------------------------------------------
// Kernel launch helper: compute grid size
//-----------------------------------------------
inline int gpu_grid_size(int n, int blockSize = BLOCK_SIZE)
{
    return (n + blockSize - 1) / blockSize;
}

//-----------------------------------------------
// Constant memory for gamma matrices
//
// 5 gamma matrices x 4 rows = 20 entries each.
// Stored in constant memory for broadcast access
// across all threads in a warp (single cache line).
//-----------------------------------------------
extern __constant__ int c_gammaIdx[5 * 4];
extern __constant__ double c_gammaValRe[5 * 4];
extern __constant__ double c_gammaValIm[5 * 4];

// Upload gamma matrices to constant memory
void gpu_upload_gamma_const(const int* gammaIdx, const void* gammaVal);

//-----------------------------------------------
// CUDA streams for async overlap
//-----------------------------------------------
extern cudaStream_t stream_compute;
extern cudaStream_t stream_reduce;
void gpu_create_streams();
void gpu_destroy_streams();
