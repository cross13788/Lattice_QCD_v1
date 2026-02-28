#!/usr/bin/env python3
"""
Effective mass analysis for Lattice QCD meson correlators.

Reads correlator data C(t), computes effective mass via:
  - Log:  m_eff(t) = log(C(t)/C(t+1))
  - Cosh: solves C(t)/C(t+1) = cosh(m*(T/2-t)) / cosh(m*(T/2-t-1))

Identifies plateau region and extracts mass with jackknife errors.

Usage:
  python3 effective_mass.py <correlator_dir> [--channel pion]
  python3 effective_mass.py   (auto-detect from run/)
"""

import os
import sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jackknife_analysis import jackknife, jackknife_func

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


CHANNEL_NAMES = {
    'pion': 'correlator_pion.dat',
    'rho_x': 'correlator_rho_x.dat',
    'rho_y': 'correlator_rho_y.dat',
    'rho_z': 'correlator_rho_z.dat',
    'scalar': 'correlator_scalar.dat',
    'a1_x': 'correlator_a1_x.dat',
    'a1_y': 'correlator_a1_y.dat',
    'a1_z': 'correlator_a1_z.dat',
}


def read_correlators(filename):
    """
    Read correlator file.
    Returns dict: config_num -> (t_array, C_array, meff_array)
    """
    configs = defaultdict(lambda: ([], [], []))
    with open(filename) as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            cfg = int(parts[0])
            t = int(parts[1])
            C = float(parts[2])
            meff = float(parts[3])
            configs[cfg][0].append(t)
            configs[cfg][1].append(C)
            configs[cfg][2].append(meff)

    result = {}
    for cfg, (ts, Cs, ms) in configs.items():
        result[cfg] = (np.array(ts), np.array(Cs), np.array(ms))
    return result


def cosh_effective_mass(C, nT):
    """
    Compute cosh effective mass from correlator.
    Solves: C(t)/C(t+1) = cosh(m*(T/2-t)) / cosh(m*(T/2-t-1))
    using Newton's method.
    """
    T2 = nT / 2.0
    t_eff, m_eff = [], []

    for t in range(1, nT // 2 - 1):
        if C[t] <= 0 or C[t+1] <= 0:
            continue
        ratio = C[t] / C[t+1]
        if ratio <= 0:
            continue

        # Initial guess from log
        m = np.log(ratio) if ratio > 1 else 0.1

        # Newton's method
        converged = False
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
                dm = f / df
                m -= dm
                if abs(dm) < 1e-12:
                    converged = True
                    break
            except (OverflowError, FloatingPointError):
                break

        if converged and m > 0:
            t_eff.append(t)
            m_eff.append(m)

    return np.array(t_eff), np.array(m_eff)


def extract_plateau_mass(t_arr, m_arr, t_min=None, t_max=None):
    """
    Extract mass from plateau region of effective mass.
    If t_min/t_max not specified, uses middle third of data.
    """
    if len(t_arr) < 3:
        if len(t_arr) > 0:
            return np.mean(m_arr), 0.0
        return 0.0, 0.0

    if t_min is None:
        t_min = t_arr[len(t_arr)//3]
    if t_max is None:
        t_max = t_arr[2*len(t_arr)//3]

    mask = (t_arr >= t_min) & (t_arr <= t_max)
    plateau_data = m_arr[mask]

    if len(plateau_data) < 2:
        return np.mean(m_arr), 0.0

    return jackknife(plateau_data)


def analyze_directory(directory, channel='pion'):
    """Analyze correlators from a single output directory."""
    filename = os.path.join(directory, CHANNEL_NAMES.get(channel, f'correlator_{channel}.dat'))
    if not os.path.exists(filename):
        print(f"  Warning: {filename} not found")
        return None

    configs = read_correlators(filename)
    if not configs:
        print(f"  Warning: No data in {filename}")
        return None

    # Get nT from data
    first_cfg = list(configs.values())[0]
    nT = len(first_cfg[0])

    # Average correlator over configs
    n_configs = len(configs)
    C_avg = np.zeros(nT)
    C_all = np.zeros((n_configs, nT))

    for i, (cfg_num, (ts, Cs, ms)) in enumerate(sorted(configs.items())):
        C_all[i, :] = Cs
        C_avg += Cs
    C_avg /= n_configs

    # Cosh effective mass from averaged correlator
    t_eff, m_eff = cosh_effective_mass(C_avg, nT)

    # Jackknife effective mass at each t
    m_eff_jack = np.zeros(len(t_eff))
    m_eff_err = np.zeros(len(t_eff))

    for j, t in enumerate(t_eff):
        def extract_mass_at_t(corr_at_t_and_tp1):
            ct = corr_at_t_and_tp1[0]
            ctp1 = corr_at_t_and_tp1[1]
            if ct <= 0 or ctp1 <= 0:
                return 0.0
            ratio = ct / ctp1
            T2 = nT / 2.0
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
            return m

        # Jackknife over configs
        masses = []
        for i in range(n_configs):
            ct = C_all[i, t]
            ctp1 = C_all[i, t+1]
            if ct > 0 and ctp1 > 0:
                masses.append(extract_mass_at_t(np.array([ct, ctp1])))

        if len(masses) >= 2:
            m_eff_jack[j], m_eff_err[j] = jackknife(np.array(masses))
        else:
            m_eff_jack[j] = m_eff[j]
            m_eff_err[j] = 0.0

    return {
        'nT': nT,
        'n_configs': n_configs,
        'C_avg': C_avg,
        't_eff': t_eff,
        'm_eff': m_eff_jack,
        'm_eff_err': m_eff_err,
    }


def main():
    import glob

    run_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run')
    channel = 'pion'

    if len(sys.argv) > 1:
        dirs = [sys.argv[1]]
    else:
        # Auto-detect directories with correlator data
        dirs = sorted(glob.glob(os.path.join(run_dir, 'SU3_heatbath_beta*')))
        dirs = [d for d in dirs if os.path.exists(
            os.path.join(d, CHANNEL_NAMES.get(channel, f'correlator_{channel}.dat')))]

    if not dirs:
        print("No correlator data found.")
        sys.exit(1)

    print("=" * 60)
    print(f"  Effective Mass Analysis - Channel: {channel}")
    print("=" * 60)

    all_results = []

    for d in dirs:
        basename = os.path.basename(d)
        print(f"\n  Directory: {basename}")

        result = analyze_directory(d, channel)
        if result is None:
            continue

        # Extract plateau mass
        m, m_err = extract_plateau_mass(result['t_eff'], result['m_eff'])

        print(f"    N_configs = {result['n_configs']}")
        print(f"    nT = {result['nT']}")
        print(f"    Effective mass:")
        for j, t in enumerate(result['t_eff']):
            print(f"      t={t:2d}: m_eff = {result['m_eff'][j]:.6f} +/- {result['m_eff_err'][j]:.6f}")
        print(f"    Plateau mass: {m:.6f} +/- {m_err:.6f}")

        all_results.append((basename, result, m, m_err))

    # Plot
    if HAS_PLOT and all_results:
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']
        for i, (name, result, m, m_err) in enumerate(all_results):
            color = colors[i % len(colors)]
            ax.errorbar(result['t_eff'], result['m_eff'],
                        yerr=result['m_eff_err'],
                        fmt='o-', color=color, markersize=6, capsize=3,
                        linewidth=1.5, label=name[:30])

        ax.set_xlabel('$t/a$')
        ax.set_ylabel('$m_{\\mathrm{eff}}(t) \\cdot a$')
        ax.set_title(f'Effective Mass - {channel}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        outfile = os.path.join(run_dir, f'effective_mass_{channel}.png')
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f"\n  Plot saved to: {outfile}")
        plt.close()


if __name__ == '__main__':
    main()
