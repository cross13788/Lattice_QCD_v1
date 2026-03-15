#!/usr/bin/env python3
"""
Autocorrelation analysis for Lattice QCD observables.

Implements Madras-Sokal self-consistent windowing for robust
estimation of integrated autocorrelation times, and uses these
to determine optimal jackknife bin sizes.

Usage:
  python3 autocorrelation_analysis.py [plaquette_file] [polyakov_file]
  python3 autocorrelation_analysis.py   (auto-detect from run/)
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jackknife_analysis import autocorrelation, integrated_autocorrelation_time, \
    jackknife, binned_jackknife

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


def madras_sokal_tau_int(data, c=6.0, max_lag=None):
    """
    Madras-Sokal self-consistent windowing for tau_int.

    Sums the autocorrelation function up to t_max where
    t_max = c * tau_int(t_max), with c ~ 6 (Madras & Sokal, 1988).
    This avoids premature truncation at the first negative ACF value.

    Parameters:
        data: 1D array of measurements
        c: window constant (default 6.0)
        max_lag: absolute maximum lag to consider (default N/4)

    Returns:
        (tau_int, t_window, running_tau) where:
          tau_int: final integrated autocorrelation time
          t_window: window size used
          running_tau: array of tau_int(t) for t = 0..max_lag
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    if max_lag is None:
        max_lag = n // 4
    max_lag = min(max_lag, n - 1)

    lags, acf = autocorrelation(data, max_lag)

    # Compute running tau_int as a function of window size
    running_tau = np.zeros(max_lag + 1)
    running_tau[0] = 0.5  # C(0)/2 = 0.5 since C(0) is normalized to 1

    for t in range(1, max_lag + 1):
        running_tau[t] = running_tau[t - 1] + acf[t]

    # Self-consistent window: find smallest t_max where t_max >= c * tau_int(t_max)
    t_window = max_lag
    for t in range(1, max_lag + 1):
        if t >= c * running_tau[t]:
            t_window = t
            break

    tau_int = running_tau[t_window]
    # Ensure tau_int >= 0.5 (uncorrelated limit)
    tau_int = max(tau_int, 0.5)

    return tau_int, t_window, running_tau


def optimal_bin_size(tau_int):
    """
    Compute optimal jackknife bin size from tau_int.
    bin_size = ceil(2 * tau_int)
    """
    return max(1, int(np.ceil(2.0 * tau_int)))


def effective_samples(n, tau_int):
    """
    Effective number of independent samples.
    N_eff = N / (2 * tau_int)
    """
    return n / (2.0 * tau_int) if tau_int > 0 else n


def read_plaquette_data(filename):
    """Read plaquette time series from plaquette.dat."""
    configs = []
    plaquettes = []
    with open(filename) as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            configs.append(int(parts[0]))
            plaquettes.append(float(parts[1]))
    return np.array(configs), np.array(plaquettes)


def read_polyakov_data(filename):
    """Read Polyakov loop time series from polyakov.dat."""
    configs = []
    values = []
    with open(filename) as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            configs.append(int(parts[0]))
            values.append(float(parts[1]))
    return np.array(configs), np.array(values)


def analyze_observable(name, data, verbose=True):
    """
    Full autocorrelation analysis for one observable.
    Returns dict with tau_int, bin_size, N_eff, naive/binned errors.
    """
    n = len(data)
    if n < 10:
        if verbose:
            print(f"  {name}: too few measurements ({n})")
        return None

    # Madras-Sokal tau_int
    tau_int, t_window, running_tau = madras_sokal_tau_int(data)

    # Also compute naive (first-negative) tau_int for comparison
    tau_naive = integrated_autocorrelation_time(data)

    # Optimal bin size
    bin_sz = optimal_bin_size(tau_int)
    n_eff = effective_samples(n, tau_int)

    # Naive jackknife (no binning)
    mean_naive, err_naive = jackknife(data)

    # Binned jackknife
    mean_binned, err_binned = binned_jackknife(data, bin_sz)

    # Error ratio (should be >= 1 if autocorrelations are present)
    err_ratio = err_binned / err_naive if err_naive > 0 else 1.0

    # Scan bin sizes for error vs bin_size plot
    max_bin = min(n // 4, 50)
    bin_sizes = list(range(1, max_bin + 1))
    bin_errors = []
    for bs in bin_sizes:
        _, err = binned_jackknife(data, bs)
        bin_errors.append(err)

    result = {
        'name': name,
        'n': n,
        'mean': mean_naive,
        'tau_int_ms': tau_int,
        'tau_int_naive': tau_naive,
        't_window': t_window,
        'running_tau': running_tau,
        'bin_size': bin_sz,
        'n_eff': n_eff,
        'err_naive': err_naive,
        'err_binned': err_binned,
        'err_ratio': err_ratio,
        'bin_sizes': bin_sizes,
        'bin_errors': bin_errors,
    }

    if verbose:
        print(f"\n  {name}:")
        print(f"    N measurements:         {n}")
        print(f"    Mean:                    {mean_naive:.10f}")
        print(f"    tau_int (Madras-Sokal):  {tau_int:.2f}  (window={t_window})")
        print(f"    tau_int (first-negative):{tau_naive:.2f}")
        print(f"    N_eff:                   {n_eff:.1f}")
        print(f"    Optimal bin size:        {bin_sz}")
        print(f"    Naive error:             {err_naive:.2e}")
        print(f"    Binned error:            {err_binned:.2e}")
        print(f"    Error ratio (bin/naive): {err_ratio:.3f}")

    return result


def plot_autocorrelation(results, output_file):
    """
    Generate 3-panel plot for each observable:
      (a) C(t) vs lag
      (b) Running tau_int vs window
      (c) Jackknife error vs bin size
    """
    if not HAS_PLOT:
        return

    n_obs = len(results)
    fig, axes = plt.subplots(n_obs, 3, figsize=(15, 4.5 * n_obs), squeeze=False)

    colors = ['#1565C0', '#2E7D32', '#E65100', '#6A1B9A']

    for i, res in enumerate(results):
        if res is None:
            continue

        color = colors[i % len(colors)]

        # Panel (a): ACF
        ax_a = axes[i, 0]
        lags_plot, acf_plot = autocorrelation(
            np.zeros(1),  # dummy, we'll compute manually
        ) if False else (None, None)

        # Recompute ACF for plotting
        max_lag = min(res['n'] // 4, 200)
        from jackknife_analysis import autocorrelation as acf_func
        lags, acf = acf_func(np.zeros(1), 0)  # dummy

        # Read the actual data again for ACF plot
        # We stored running_tau, derive ACF from it:
        # running_tau[0] = 0.5, running_tau[t] = running_tau[t-1] + acf[t]
        # => acf[t] = running_tau[t] - running_tau[t-1] for t >= 1
        rt = res['running_tau']
        t_max = len(rt)
        acf_derived = np.zeros(t_max)
        acf_derived[0] = 1.0  # C(0) = 1
        for t in range(1, t_max):
            acf_derived[t] = rt[t] - rt[t - 1]

        lags = np.arange(t_max)
        plot_max = min(t_max, max(4 * res['t_window'], 30))

        ax_a.plot(lags[:plot_max], acf_derived[:plot_max], '-', color=color, linewidth=1.5)
        ax_a.axhline(0, color='black', linewidth=0.5)
        ax_a.axvline(res['t_window'], color='red', linestyle='--', alpha=0.7,
                      label=f'window={res["t_window"]}')
        ax_a.set_xlabel('Lag $t$')
        ax_a.set_ylabel('$C(t)$')
        ax_a.set_title(f'{res["name"]}: Autocorrelation Function')
        ax_a.legend(fontsize=9)
        ax_a.grid(True, alpha=0.3)

        # Panel (b): Running tau_int
        ax_b = axes[i, 1]
        ax_b.plot(lags[:plot_max], rt[:plot_max], '-', color=color, linewidth=1.5)
        ax_b.axhline(res['tau_int_ms'], color='red', linestyle='--', alpha=0.7,
                      label=f'$\\tau_{{int}}={res["tau_int_ms"]:.2f}$')
        ax_b.axhline(res['tau_int_naive'], color='gray', linestyle=':', alpha=0.7,
                      label=f'$\\tau_{{int}}^{{naive}}={res["tau_int_naive"]:.2f}$')
        ax_b.set_xlabel('Window $W$')
        ax_b.set_ylabel('$\\tau_{int}(W)$')
        ax_b.set_title(f'{res["name"]}: Running $\\tau_{{int}}$')
        ax_b.legend(fontsize=9)
        ax_b.grid(True, alpha=0.3)

        # Panel (c): Error vs bin size
        ax_c = axes[i, 2]
        ax_c.plot(res['bin_sizes'], res['bin_errors'], 'o-', color=color,
                  markersize=3, linewidth=1)
        ax_c.axhline(res['err_naive'], color='gray', linestyle=':', alpha=0.7,
                      label=f'naive={res["err_naive"]:.2e}')
        ax_c.axvline(res['bin_size'], color='red', linestyle='--', alpha=0.7,
                      label=f'optimal bin={res["bin_size"]}')
        ax_c.set_xlabel('Bin size')
        ax_c.set_ylabel('Jackknife error')
        ax_c.set_title(f'{res["name"]}: Error vs Bin Size')
        ax_c.legend(fontsize=9)
        ax_c.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    import glob

    run_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run')

    print("=" * 65)
    print("  Autocorrelation Analysis")
    print("  Madras-Sokal self-consistent windowing")
    print("=" * 65)

    # Find data files
    plaq_files = []
    poly_files = []

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if os.path.isfile(arg):
                if 'plaquette' in arg:
                    plaq_files.append(arg)
                elif 'polyakov' in arg:
                    poly_files.append(arg)
            elif os.path.isdir(arg):
                pf = os.path.join(arg, 'plaquette.dat')
                if os.path.exists(pf):
                    plaq_files.append(pf)
                pf2 = os.path.join(arg, 'polyakov.dat')
                if os.path.exists(pf2):
                    poly_files.append(pf2)
    else:
        # Auto-detect
        dirs = sorted(glob.glob(os.path.join(run_dir, 'SU3_*')))
        for d in dirs:
            pf = os.path.join(d, 'plaquette.dat')
            if os.path.exists(pf):
                plaq_files.append(pf)
            pf2 = os.path.join(d, 'polyakov.dat')
            if os.path.exists(pf2):
                poly_files.append(pf2)

    if not plaq_files and not poly_files:
        print("\n  No data files found.")
        print("  Usage: python3 autocorrelation_analysis.py [plaquette.dat] [polyakov.dat]")
        sys.exit(1)

    all_results = []

    for pf in plaq_files:
        dirname = os.path.basename(os.path.dirname(pf))
        print(f"\n--- {dirname} ---")

        configs, plaq = read_plaquette_data(pf)
        res = analyze_observable(f"Plaquette ({dirname})", plaq)
        if res is not None:
            all_results.append(res)

    for pf in poly_files:
        dirname = os.path.basename(os.path.dirname(pf))
        configs, poly = read_polyakov_data(pf)
        res = analyze_observable(f"Polyakov ({dirname})", poly)
        if res is not None:
            all_results.append(res)

    # Summary table
    if all_results:
        print("\n" + "=" * 65)
        print("  Summary")
        print("=" * 65)
        print(f"  {'Observable':<35} {'tau_int':>8} {'bin':>5} {'N_eff':>8} {'err_ratio':>10}")
        print(f"  {'-'*35} {'-'*8} {'-'*5} {'-'*8} {'-'*10}")
        for res in all_results:
            print(f"  {res['name']:<35} {res['tau_int_ms']:8.2f} {res['bin_size']:5d} "
                  f"{res['n_eff']:8.1f} {res['err_ratio']:10.3f}")

    # Verification checks
    print("\n  Verification:")
    for res in all_results:
        ok_tau = res['tau_int_ms'] > 0
        ok_err = res['err_ratio'] >= 0.99  # binned error >= naive (within rounding)
        status_tau = "PASS" if ok_tau else "FAIL"
        status_err = "PASS" if ok_err else "WARN"
        print(f"    {res['name']}: tau_int > 0: {status_tau}, "
              f"binned_err >= naive_err: {status_err} ({res['err_ratio']:.3f})")

    # Plot
    if HAS_PLOT and all_results:
        outfile = os.path.join(run_dir, 'autocorrelation_analysis.png')
        plot_autocorrelation(all_results, outfile)
        print(f"\n  Plot saved to: {outfile}")

    print()


if __name__ == '__main__':
    main()
