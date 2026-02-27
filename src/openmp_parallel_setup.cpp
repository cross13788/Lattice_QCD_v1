//-----------------------------------------------
// OpenMP parallel environment setup
//
// Reports thread count and processor info.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

void openmp_parallel_setup(int& numberProcessors, int& maxThreads)
{
#ifdef _OPENMP
    numberProcessors = omp_get_num_procs();
    maxThreads = omp_get_max_threads();
    printf("OpenMP enabled\n");
    printf("  Available processors: %d\n", numberProcessors);
    printf("  Max threads:          %d\n", maxThreads);
#else
    numberProcessors = 1;
    maxThreads = 1;
    printf("OpenMP not available, running in serial mode\n");
#endif
    printf("\n");
}
