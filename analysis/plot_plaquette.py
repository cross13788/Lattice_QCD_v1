#!/usr/bin/env python3
"""
Plaquette analysis for Lattice QCD validation tests.

Reads plaquette data from simulation output directories,
computes jackknife averages with errors, and plots:
  1. Thermalization history (plaquette vs config)
  2. Plaquette vs beta comparison with reference values

Usage:
  python3 plot_plaquette.py [output_dir1] [output_dir2] ...

If no arguments given, scans for SU3_heatbath_*_8x8x8x8 directories in run/.
"""

import sys
import os
import numpy as np
import glob

# Try to import matplotlib; if not available, print results only
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("Warning: matplotlib not available. Printing results only.")


def read_plaquette_data(filename):
    """Read plaquette.dat file, return config numbers and plaquette values."""
    configs = []
    plaquettes = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            configs.append(int(parts[0]))
            plaquettes.append(float(parts[1]))
    return np.array(configs), np.array(plaquettes)


def jackknife_mean_error(data):
    """Compute mean and jackknife error estimate."""
    n = len(data)
    if n < 2:
        return np.mean(data), 0.0
    mean = np.mean(data)
    jack_means = np.zeros(n)
    for i in range(n):
        jack_means[i] = (np.sum(data) - data[i]) / (n - 1)
    jack_var = (n - 1) / n * np.sum((jack_means - mean)**2)
    return mean, np.sqrt(jack_var)


def extract_beta_from_dirname(dirname):
    """Extract beta value from directory name like SU3_heatbath_beta6.00_8x8x8x8."""
    basename = os.path.basename(dirname)
    parts = basename.split('_')
    for part in parts:
        if part.startswith('beta'):
            return float(part[4:])
    return None


def main():
    run_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run')

    if len(sys.argv) > 1:
        dirs = sys.argv[1:]
    else:
        # Auto-detect: look for SU3_heatbath_*_8x8x8x8 directories
        pattern = os.path.join(run_dir, 'SU3_heatbath_*_8x8x8x8')
        dirs = sorted(glob.glob(pattern))
        if not dirs:
            print("No output directories found. Provide paths as arguments.")
            sys.exit(1)

    # Reference values (infinite volume, from literature)
    reference = {5.6: 0.548, 5.8: 0.571, 6.0: 0.5935, 6.2: 0.613}

    print("=" * 60)
    print("  Plaquette Analysis - Test 1a: <P> vs Beta")
    print("=" * 60)

    results = []

    for d in dirs:
        plaq_file = os.path.join(d, 'plaquette.dat')
        if not os.path.exists(plaq_file):
            print(f"  Warning: {plaq_file} not found, skipping.")
            continue

        beta = extract_beta_from_dirname(d)
        configs, plaquettes = read_plaquette_data(plaq_file)
        mean, err = jackknife_mean_error(plaquettes)

        ref = reference.get(beta, None)
        ref_str = f"{ref:.4f}" if ref else "N/A"
        dev_str = ""
        if ref:
            dev = (mean - ref) / ref * 100
            dev_str = f"  ({dev:+.1f}%)"

        print(f"\n  Beta = {beta:.2f}")
        print(f"    <P>       = {mean:.6f} +/- {err:.6f}")
        print(f"    Reference = {ref_str}{dev_str}")
        print(f"    N_configs = {len(plaquettes)}")

        results.append((beta, mean, err))

    results.sort(key=lambda x: x[0])

    if not results:
        print("No data to analyze.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  Summary Table")
    print("=" * 60)
    print(f"  {'Beta':>6}  {'<P> measured':>16}  {'<P> reference':>14}  {'Deviation':>10}")
    print(f"  {'-'*6}  {'-'*16}  {'-'*14}  {'-'*10}")
    for beta, mean, err in results:
        ref = reference.get(beta, None)
        ref_str = f"{ref:.4f}" if ref else "N/A"
        dev_str = ""
        if ref:
            dev = (mean - ref) / ref * 100
            dev_str = f"{dev:+.2f}%"
        print(f"  {beta:6.2f}  {mean:10.6f}({err*1e4:.1f})  {ref_str:>14}  {dev_str:>10}")

    # Plotting
    if HAS_PLOT and results:
        betas = [r[0] for r in results]
        means = [r[1] for r in results]
        errs = [r[2] for r in results]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Thermalization histories
        ax1 = axes[0]
        for d in dirs:
            plaq_file = os.path.join(d, 'plaquette.dat')
            if not os.path.exists(plaq_file):
                continue
            beta = extract_beta_from_dirname(d)
            configs, plaquettes = read_plaquette_data(plaq_file)
            ax1.plot(configs, plaquettes, '-', alpha=0.7, label=f'$\\beta={beta:.1f}$')

        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Average Plaquette $\\langle P \\rangle$')
        ax1.set_title('Plaquette History (Post-Thermalization)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: <P> vs beta with reference
        ax2 = axes[1]
        ax2.errorbar(betas, means, yerr=errs, fmt='o-', color='blue',
                     markersize=8, capsize=4, label='This work (8$^4$)')

        ref_betas = sorted(reference.keys())
        ref_vals = [reference[b] for b in ref_betas]
        ax2.plot(ref_betas, ref_vals, 's--', color='red', markersize=8,
                 alpha=0.7, label='Reference (large volume)')

        ax2.set_xlabel('$\\beta$')
        ax2.set_ylabel('$\\langle P \\rangle$')
        ax2.set_title('Plaquette vs $\\beta$ - SU(3) Wilson Gauge')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        outfile = os.path.join(run_dir, 'plaquette_vs_beta.png')
        plt.savefig(outfile, dpi=150)
        print(f"\n  Plot saved to: {outfile}")

        plt.close()

    print()


if __name__ == '__main__':
    main()
