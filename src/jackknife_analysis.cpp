//-----------------------------------------------
// Jackknife error estimation
//
// Delete-1 jackknife for scalar observables:
//   mean = (1/N) sum_i x_i
//   var  = (N-1)/N * sum_i (x_i_bar - mean)^2
// where x_i_bar is the mean with sample i removed.
//-----------------------------------------------
#include "constants_module.hpp"
#include "interfaces_module.hpp"
#include <cmath>

void jackknife_analysis(const real_t* data, int nData,
                        real_t& mean, real_t& error)
{
//-----------------------------------------------
//   Compute mean
//-----------------------------------------------
    real_t sum = 0.0;
    for (int i = 0; i < nData; i++)
        sum += data[i];
    mean = sum / nData;

//-----------------------------------------------
//   Jackknife error
//-----------------------------------------------
    if (nData < 2) {
        error = 0.0;
        return;
    }

    real_t jackVar = 0.0;
    for (int i = 0; i < nData; i++) {
        // Mean with sample i removed
        real_t jackMean = (sum - data[i]) / (nData - 1);
        real_t diff = jackMean - mean;
        jackVar += diff * diff;
    }
    jackVar *= (real_t)(nData - 1) / (real_t)nData;

    error = std::sqrt(jackVar);
}
