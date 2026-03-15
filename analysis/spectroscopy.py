#!/usr/bin/env python3
"""
Meson spectroscopy analysis for Test 2b.

Reads pion correlator data from multiple kappa runs,
extracts pion masses, and checks the PCAC relation:
  m_pi^2 ~ (1/kappa - 1/kappa_c)

Also compares rho mass to pion mass for each kappa.

Usage:
  python3 spectroscopy.py [dir1] [dir2] ...
  python3 spectroscopy.py   (auto-detect from run/)
"""

import os
import sys
import re
import numpy as np
from collections import defaultdict
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jackknife_analysis import jackknife, binned_jackknife, jackknife_func
from autocorrelation_analysis import madras_sokal_tau_int, optimal_bin_size
from effective_mass import read_correlators, cosh_effective_mass

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


def extract_kappa_from_input(directory):
    """Try to extract kappa from the run, using directory naming or config."""
    # Try reading correlator file to get hint from directory name
    basename = os.path.basename(directory)
    # Our spectro directories don't encode kappa in the name since the code
    # names them SU3_heatbath_betaX.XX_NxNxNxN. Need another approach.
    # Try reading the pion correlator data and inferring from effective mass,
    # or check if there's a log file.
    return None


def get_pion_mass(directory, nT=None):
    """
    Extract pion mass from correlator data in directory.
    Returns (mass, error, nT, n_configs) or None.
    """
    pion_file = os.path.join(directory, 'correlator_pion.dat')
    if not os.path.exists(pion_file):
        return None

    configs = read_correlators(pion_file)
    if not configs:
        return None

    first = list(configs.values())[0]
    nT = len(first[0])
    n_configs = len(configs)

    # Average correlator
    C_avg = np.zeros(nT)
    C_all = np.zeros((n_configs, nT))
    for i, (cfg, (ts, Cs, ms)) in enumerate(sorted(configs.items())):
        C_all[i, :] = Cs
        C_avg += Cs
    C_avg /= n_configs

    # Cosh effective mass from averaged correlator
    t_eff, m_eff = cosh_effective_mass(C_avg, nT)

    if len(t_eff) < 2:
        return None

    # Jackknife over configs for each timeslice
    m_jack = np.zeros((n_configs, len(t_eff)))
    for j, t in enumerate(t_eff):
        T2 = nT / 2.0
        for i in range(n_configs):
            ct = C_all[i, t]
            ctp1 = C_all[i, t+1]
            if ct > 0 and ctp1 > 0:
                ratio = ct / ctp1
                m = np.log(ratio) if ratio > 1 else 0.1
                for _ in range(200):
                    try:
                        num = np.cosh(m * (T2 - t))
                        den = np.cosh(m * (T2 - t - 1))
                        f = num / den - ratio
                        d_num = (T2 - t) * np.sinh(m * (T2 - t))
                        d_den = (T2 - t - 1) * np.sinh(m * (T2 - t - 1))
                        df = (d_num * den - num * d_den) / (den * den)
                        if abs(df) < 1e-20:
                            break
                        m -= f / df
                        if abs(f / df) < 1e-12:
                            break
                    except:
                        break
                m_jack[i, j] = m

    # Find plateau (use middle timeslices)
    n_t = len(t_eff)
    if n_t >= 4:
        t_start = max(1, n_t // 3)
        t_end = min(n_t - 1, 2 * n_t // 3 + 1)
    else:
        t_start = 0
        t_end = n_t

    # Jackknife the plateau average with binning for autocorrelations
    plateau_masses = np.mean(m_jack[:, t_start:t_end], axis=1)
    if len(plateau_masses) > 10:
        tau, _, _ = madras_sokal_tau_int(plateau_masses)
        bin_sz = optimal_bin_size(tau)
        m_pi, m_pi_err = binned_jackknife(plateau_masses, bin_sz)
    else:
        m_pi, m_pi_err = jackknife(plateau_masses)

    return {
        'mass': m_pi,
        'error': m_pi_err,
        'nT': nT,
        'n_configs': n_configs,
        't_eff': t_eff,
        'm_eff': np.mean(m_jack, axis=0),
        'm_eff_err': np.array([np.sqrt((n_configs-1)/n_configs * np.sum((m_jack[:, j] - np.mean(m_jack[:, j]))**2))
                               for j in range(len(t_eff))]),
        'plateau_range': (t_eff[t_start], t_eff[t_end-1]),
    }


def get_rho_mass(directory, nT=None):
    """Extract rho mass (average of 3 polarizations)."""
    rho_files = [
        os.path.join(directory, 'correlator_rho_x.dat'),
        os.path.join(directory, 'correlator_rho_y.dat'),
        os.path.join(directory, 'correlator_rho_z.dat'),
    ]

    all_corr = []
    for rf in rho_files:
        if not os.path.exists(rf):
            continue
        configs = read_correlators(rf)
        if not configs:
            continue

        first = list(configs.values())[0]
        nT_local = len(first[0])
        n_configs = len(configs)

        C_avg = np.zeros(nT_local)
        for cfg, (ts, Cs, ms) in configs.items():
            C_avg += np.abs(Cs)  # rho correlator can be negative, take abs for eff mass
        C_avg /= n_configs
        all_corr.append(C_avg)

    if not all_corr:
        return None

    # Average over polarizations
    C_rho = np.mean(all_corr, axis=0)
    nT = len(C_rho)
    t_eff, m_eff = cosh_effective_mass(C_rho, nT)

    if len(t_eff) < 2:
        return None

    # Simple plateau (middle third)
    n_t = len(t_eff)
    t_start = max(1, n_t // 3)
    t_end = min(n_t - 1, 2 * n_t // 3 + 1)
    m_rho = np.mean(m_eff[t_start:t_end])

    return {'mass': m_rho, 't_eff': t_eff, 'm_eff': m_eff}


def main():
    run_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run')

    if len(sys.argv) > 1:
        kappa_dirs = {}
        for arg in sys.argv[1:]:
            # Try to parse kappa from argument or directory
            kappa_dirs[arg] = (float(arg.split('k')[-1]) / 1000 if 'k' in arg else None, arg)
    else:
        # Auto-detect
        pass

    # Collect results: list of (kappa, pion_result, rho_result, directory)
    results = []

    # User supplies kappa values as second arg or we detect from filenames
    # For now, look for directories and prompt
    print("=" * 65)
    print("  Meson Spectroscopy Analysis - Test 2b")
    print("  m_pi^2 vs 1/kappa (PCAC relation)")
    print("=" * 65)

    # Check for spectroscopy data passed as kappa:directory pairs
    # Or just process what we find
    if len(sys.argv) > 1:
        for i in range(1, len(sys.argv), 2):
            kappa = float(sys.argv[i])
            directory = sys.argv[i+1] if i+1 < len(sys.argv) else None
            if directory and os.path.isdir(directory):
                pion = get_pion_mass(directory)
                rho = get_rho_mass(directory)
                if pion:
                    results.append((kappa, pion, rho, directory))
    else:
        # Try to find spectroscopy directories
        # These are named SU3_heatbath_beta6.00_8x8x8x16 (all look the same)
        # We need kappa mapping. Read from a results file if it exists.
        results_file = os.path.join(run_dir, 'spectroscopy_kappa_map.txt')
        if os.path.exists(results_file):
            with open(results_file) as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        kappa = float(parts[0])
                        directory = parts[1]
                        if os.path.isdir(directory):
                            pion = get_pion_mass(directory)
                            rho = get_rho_mass(directory)
                            if pion:
                                results.append((kappa, pion, rho, directory))

    if not results:
        print("\n  No spectroscopy data found.")
        print("  Usage: python3 spectroscopy.py kappa1 dir1 kappa2 dir2 ...")
        print("  Or create run/spectroscopy_kappa_map.txt with lines: kappa directory")
        sys.exit(1)

    results.sort(key=lambda x: x[0])

    # Print results
    print(f"\n  {'Kappa':>8} {'m_pi':>10} {'m_pi err':>10} {'m_rho':>10} {'m_pi^2':>10} {'1/kappa':>10} {'N_cfg':>6}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")

    kappas = []
    m_pis = []
    m_pi_errs = []
    m_rhos = []

    for kappa, pion, rho, d in results:
        m_rho_val = rho['mass'] if rho else float('nan')
        print(f"  {kappa:8.4f} {pion['mass']:10.6f} {pion['error']:10.6f} "
              f"{m_rho_val:10.6f} {pion['mass']**2:10.6f} {1.0/kappa:10.6f} {pion['n_configs']:6d}")

        kappas.append(kappa)
        m_pis.append(pion['mass'])
        m_pi_errs.append(pion['error'])
        m_rhos.append(m_rho_val)

    # PCAC relation: m_pi^2 ~ A * (1/kappa - 1/kappa_c)
    if len(kappas) >= 3:
        inv_k = np.array([1.0/k for k in kappas])
        m2 = np.array([m**2 for m in m_pis])

        # Linear fit: m_pi^2 = A * (1/kappa) + B
        # kappa_c = -A/B
        A_mat = np.vstack([inv_k, np.ones(len(inv_k))]).T
        params = np.linalg.lstsq(A_mat, m2, rcond=None)[0]
        slope, intercept = params

        inv_kappa_c = -intercept / slope if abs(slope) > 1e-10 else float('nan')
        kappa_c = 1.0 / inv_kappa_c if abs(inv_kappa_c) > 1e-10 else float('nan')

        print(f"\n  PCAC fit: m_pi^2 = {slope:.4f} * (1/kappa) + ({intercept:.4f})")
        print(f"  => 1/kappa_c = {inv_kappa_c:.4f},  kappa_c = {kappa_c:.4f}")
        print(f"  Expected kappa_c ~ 0.157 at beta=6.0")

    # Plot
    if HAS_PLOT and len(results) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Panel 1: Effective mass plateaus
        ax1 = axes[0]
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
        for i, (kappa, pion, rho, d) in enumerate(results):
            c = colors[i % len(colors)]
            ax1.errorbar(pion['t_eff'], pion['m_eff'], yerr=pion['m_eff_err'],
                         fmt='o-', color=c, markersize=5, capsize=3,
                         label=f'$\\kappa={kappa}$')
        ax1.set_xlabel('$t/a$')
        ax1.set_ylabel('$m_{\\mathrm{eff}} \\cdot a$')
        ax1.set_title('Pion Effective Mass')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Panel 2: m_pi^2 vs 1/kappa (PCAC)
        ax2 = axes[1]
        inv_k = [1.0/k for k in kappas]
        m2 = [m**2 for m in m_pis]
        m2_err = [2*m*e for m, e in zip(m_pis, m_pi_errs)]
        ax2.errorbar(inv_k, m2, yerr=m2_err, fmt='o', color='#2196F3',
                     markersize=8, capsize=4)

        if len(kappas) >= 3:
            inv_k_fine = np.linspace(min(inv_k) - 0.2, max(inv_k) + 0.2, 100)
            m2_fit = slope * inv_k_fine + intercept
            ax2.plot(inv_k_fine, m2_fit, '--', color='gray', linewidth=1.5,
                     label=f'PCAC fit ($\\kappa_c={kappa_c:.3f}$)')
            ax2.axhline(0, color='black', linewidth=0.5)
            ax2.axvline(1.0/kappa_c, color='red', linestyle=':', alpha=0.5,
                        label=f'$1/\\kappa_c$')
            ax2.legend(fontsize=9)

        ax2.set_xlabel('$1/\\kappa$')
        ax2.set_ylabel('$(m_\\pi a)^2$')
        ax2.set_title('PCAC Relation')
        ax2.grid(True, alpha=0.3)

        # Panel 3: m_pi and m_rho vs kappa
        ax3 = axes[2]
        ax3.errorbar(kappas, m_pis, yerr=m_pi_errs, fmt='o-', color='#2196F3',
                     markersize=7, capsize=4, label='$m_\\pi$')
        valid_rho = [(k, m) for k, m in zip(kappas, m_rhos) if not np.isnan(m)]
        if valid_rho:
            ax3.plot([v[0] for v in valid_rho], [v[1] for v in valid_rho],
                     's--', color='#F44336', markersize=7, label='$m_\\rho$')
        ax3.set_xlabel('$\\kappa$')
        ax3.set_ylabel('$m \\cdot a$')
        ax3.set_title('Meson Masses vs $\\kappa$')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        outfile = os.path.join(run_dir, 'spectroscopy_results.png')
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f"\n  Plot saved to: {outfile}")
        plt.close()


if __name__ == '__main__':
    main()
