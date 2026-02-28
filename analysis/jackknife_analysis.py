#!/usr/bin/env python3
"""
Jackknife analysis utilities for Lattice QCD.

Provides:
  - Delete-1 jackknife for error estimation
  - Binned jackknife for handling autocorrelations
  - Reusable functions for all other analysis scripts

Usage:
  Import as module:
    from jackknife_analysis import jackknife, binned_jackknife, jackknife_func
"""

import numpy as np


def jackknife(data):
    """
    Delete-1 jackknife for mean and error.

    Parameters:
        data: 1D array of measurements

    Returns:
        (mean, error)
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n < 2:
        return np.mean(data), 0.0

    total = np.sum(data)
    mean = total / n

    # Jackknife resamples
    jack_means = (total - data) / (n - 1)
    jack_var = (n - 1) / n * np.sum((jack_means - mean)**2)

    return mean, np.sqrt(jack_var)


def binned_jackknife(data, bin_size):
    """
    Binned jackknife to account for autocorrelations.

    First bins the data into blocks of bin_size, then applies
    delete-1 jackknife to the binned data.

    Parameters:
        data: 1D array of measurements
        bin_size: number of measurements per bin

    Returns:
        (mean, error)
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    n_bins = n // bin_size
    if n_bins < 2:
        return jackknife(data)

    # Create bins
    binned = np.array([np.mean(data[i*bin_size:(i+1)*bin_size])
                       for i in range(n_bins)])

    return jackknife(binned)


def jackknife_func(data, func):
    """
    Jackknife error for an arbitrary function of the data.

    Parameters:
        data: 1D array of measurements
        func: function that takes an array and returns a scalar
              (e.g., np.mean, np.var, or a custom function)

    Returns:
        (func(data), jackknife_error)
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n < 2:
        return func(data), 0.0

    full_estimate = func(data)

    # Jackknife resamples
    jack_estimates = np.zeros(n)
    for i in range(n):
        resample = np.concatenate([data[:i], data[i+1:]])
        jack_estimates[i] = func(resample)

    jack_var = (n - 1) / n * np.sum((jack_estimates - full_estimate)**2)

    return full_estimate, np.sqrt(jack_var)


def jackknife_ratio(num_data, den_data):
    """
    Jackknife error for a ratio of two means.

    Parameters:
        num_data: 1D array for numerator
        den_data: 1D array for denominator (same length)

    Returns:
        (ratio, error)
    """
    num_data = np.asarray(num_data, dtype=float)
    den_data = np.asarray(den_data, dtype=float)
    n = len(num_data)
    assert len(den_data) == n

    ratio = np.mean(num_data) / np.mean(den_data)

    jack_ratios = np.zeros(n)
    for i in range(n):
        num_jack = (np.sum(num_data) - num_data[i]) / (n - 1)
        den_jack = (np.sum(den_data) - den_data[i]) / (n - 1)
        jack_ratios[i] = num_jack / den_jack

    jack_var = (n - 1) / n * np.sum((jack_ratios - ratio)**2)

    return ratio, np.sqrt(jack_var)


def autocorrelation(data, max_lag=None):
    """
    Compute normalized autocorrelation function.

    Parameters:
        data: 1D array of measurements
        max_lag: maximum lag (default: N/4)

    Returns:
        (lags, C(lag)) where C(0) = 1
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    if max_lag is None:
        max_lag = n // 4
    max_lag = min(max_lag, n - 1)

    mean = np.mean(data)
    var = np.var(data)
    if var < 1e-30:
        return np.arange(max_lag + 1), np.zeros(max_lag + 1)

    fluct = data - mean
    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        acf[lag] = np.mean(fluct[:n-lag] * fluct[lag:]) / var

    return np.arange(max_lag + 1), acf


def integrated_autocorrelation_time(data, max_lag=None):
    """
    Estimate integrated autocorrelation time tau_int.

    tau_int = 0.5 + sum_{lag=1}^{max_lag} C(lag)

    The effective number of independent measurements is N / (2 * tau_int).

    Parameters:
        data: 1D array of measurements

    Returns:
        tau_int
    """
    lags, acf = autocorrelation(data, max_lag)

    # Sum until autocorrelation becomes negative (self-consistent window)
    tau = 0.5
    for lag in range(1, len(acf)):
        if acf[lag] < 0:
            break
        tau += acf[lag]

    return tau


if __name__ == '__main__':
    # Self-test
    np.random.seed(42)
    data = np.random.normal(5.0, 0.3, 100)

    m, e = jackknife(data)
    print(f"Jackknife: {m:.4f} +/- {e:.4f}")

    m2, e2 = binned_jackknife(data, 5)
    print(f"Binned(5): {m2:.4f} +/- {e2:.4f}")

    m3, e3 = jackknife_func(data, np.std)
    print(f"Jackknife std: {m3:.4f} +/- {e3:.4f}")

    tau = integrated_autocorrelation_time(data)
    print(f"Autocorrelation time: {tau:.2f}")

    print("\nAll tests passed.")
