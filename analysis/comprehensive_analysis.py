#!/usr/bin/env python3
"""
Comprehensive analysis and visualization of Lattice QCD validation tests.

Generates a multi-panel figure summarizing key physics results:
  1. Plaquette vs beta
  2. Polyakov loop: confined vs deconfined
  3. Static potential V(R)
  4. Pion effective mass
  5. HMC acceptance rate vs step size
  6. Wilson loop area law

Usage:
  python3 comprehensive_analysis.py
"""

import os
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("Error: matplotlib required. Install with: pip install matplotlib")
    sys.exit(1)

RUN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run')
os.chdir(RUN_DIR)


def jackknife_mean_error(data):
    n = len(data)
    if n < 2:
        return np.mean(data), 0.0
    mean = np.mean(data)
    jack_means = np.array([(np.sum(data) - data[i]) / (n - 1) for i in range(n)])
    jack_var = (n - 1) / n * np.sum((jack_means - mean)**2)
    return mean, np.sqrt(jack_var)


def read_data(filename, comment='#'):
    data = []
    with open(filename) as f:
        for line in f:
            if line.startswith(comment) or line.strip() == '':
                continue
            data.append([float(x) for x in line.split()])
    return np.array(data)


def plot_plaquette_vs_beta(ax):
    """Panel 1: Plaquette vs beta."""
    import glob
    dirs = sorted(glob.glob('SU3_heatbath_beta*_8x8x8x8'))
    if not dirs:
        ax.text(0.5, 0.5, 'No 8^4 data', transform=ax.transAxes, ha='center')
        return

    reference = {5.6: 0.548, 5.8: 0.571, 6.0: 0.5935, 6.2: 0.613}
    betas, means, errs = [], [], []

    for d in dirs:
        pf = os.path.join(d, 'plaquette.dat')
        if not os.path.exists(pf):
            continue
        basename = os.path.basename(d)
        beta = float([p for p in basename.split('_') if p.startswith('beta')][0][4:])
        data = read_data(pf)
        m, e = jackknife_mean_error(data[:, 1])
        betas.append(beta)
        means.append(m)
        errs.append(e)

    ax.errorbar(betas, means, yerr=errs, fmt='o-', color='#2196F3',
                markersize=7, capsize=4, linewidth=2, label='This work ($8^4$)')
    ref_b = sorted(reference.keys())
    ref_v = [reference[b] for b in ref_b]
    ax.plot(ref_b, ref_v, 's--', color='#F44336', markersize=7,
            alpha=0.8, linewidth=1.5, label='Literature')
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$\langle P \rangle$')
    ax.set_title('Plaquette vs $\\beta$')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_polyakov_comparison(ax):
    """Panel 2: Polyakov loop confined vs deconfined."""
    confined_file = 'SU3_heatbath_beta6.00_8x8x8x8/polyakov.dat'
    deconfined_file = 'SU3_heatbath_beta6.00_8x8x8x4/polyakov.dat'

    if not os.path.exists(confined_file) or not os.path.exists(deconfined_file):
        ax.text(0.5, 0.5, 'No Polyakov data', transform=ax.transAxes, ha='center')
        return

    dc = read_data(confined_file)
    dd = read_data(deconfined_file)

    # Plot histograms of |L|
    ax.hist(np.abs(dc[:, 1]), bins=20, alpha=0.6, color='#2196F3',
            label=f'$8^3\\times 8$ (confined)', density=True, edgecolor='black', linewidth=0.5)
    ax.hist(np.abs(dd[:, 1]), bins=20, alpha=0.6, color='#F44336',
            label=f'$8^3\\times 4$ (deconfined)', density=True, edgecolor='black', linewidth=0.5)

    mc, ec = jackknife_mean_error(np.abs(dc[:, 1]))
    md, ed = jackknife_mean_error(np.abs(dd[:, 1]))
    ax.axvline(mc, color='#2196F3', linestyle='--', linewidth=1.5)
    ax.axvline(md, color='#F44336', linestyle='--', linewidth=1.5)

    ax.set_xlabel(r'$|\langle L \rangle|$')
    ax.set_ylabel('Density')
    ax.set_title(f'Polyakov Loop ($\\beta=6.0$)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_static_potential(ax):
    """Panel 3: Static potential V(R) from Wilson loops."""
    wloop_file = 'SU3_heatbath_beta6.00_8x8x8x16/wilson_loops.dat'
    if not os.path.exists(wloop_file):
        ax.text(0.5, 0.5, 'No Wilson loop data', transform=ax.transAxes, ha='center')
        return

    data = read_data(wloop_file)
    # Average over configs
    from collections import defaultdict
    wloops = defaultdict(list)
    for row in data:
        cfg, R, T, W = int(row[0]), int(row[1]), int(row[2]), row[3]
        wloops[(R, T)].append(W)

    Wavg = {}
    for key, vals in wloops.items():
        Wavg[key] = np.mean(vals)

    # Extract V(R) at different T
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    for Ti, T_pair in enumerate([(2, 3), (3, 4), (4, 5)]):
        T1, T2 = T_pair
        Rs, Vs = [], []
        for R in range(1, 5):
            if (R, T1) in Wavg and (R, T2) in Wavg:
                w1, w2 = Wavg[(R, T1)], Wavg[(R, T2)]
                if w1 > 0 and w2 > 0:
                    Rs.append(R)
                    Vs.append(-np.log(w2 / w1))
        if Rs:
            ax.plot(Rs, Vs, 'o-', color=colors[Ti], markersize=6,
                    label=f'$T={T1}/{T2}$', linewidth=1.5)

    # Add Cornell fit
    Rs_fit, Vs_fit = [], []
    for R in range(1, 5):
        if (R, 3) in Wavg and (R, 4) in Wavg:
            w1, w2 = Wavg[(R, 3)], Wavg[(R, 4)]
            if w1 > 0 and w2 > 0:
                Rs_fit.append(R)
                Vs_fit.append(-np.log(w2 / w1))
    if len(Rs_fit) >= 3:
        R_arr = np.array(Rs_fit, dtype=float)
        V_arr = np.array(Vs_fit)
        # Fit V = -alpha/R + sigma*R + c
        A = np.vstack([-1.0/R_arr, R_arr, np.ones(len(R_arr))]).T
        try:
            params = np.linalg.lstsq(A, V_arr, rcond=None)[0]
            R_fine = np.linspace(0.8, 4.5, 100)
            V_fine = -params[0]/R_fine + params[1]*R_fine + params[2]
            ax.plot(R_fine, V_fine, '--', color='gray', linewidth=1,
                    label=f'Cornell fit')
        except:
            pass

    ax.set_xlabel('$R/a$')
    ax.set_ylabel('$V(R) \\cdot a$')
    ax.set_title('Static Potential ($\\beta=6.0$)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_pion_effective_mass(ax):
    """Panel 4: Pion effective mass from free-field test."""
    pion_file = 'SU3_heatbath_beta6.00_8x8x8x16/correlator_pion.dat'
    if not os.path.exists(pion_file):
        ax.text(0.5, 0.5, 'No pion data', transform=ax.transAxes, ha='center')
        return

    data = read_data(pion_file)
    t = data[:, 1].astype(int)
    C = data[:, 2]
    nT = len(C)

    # Cosh effective mass
    t_eff, m_eff = [], []
    for i in range(1, nT//2 - 1):
        if C[i] > 0 and C[i+1] > 0:
            ratio = C[i] / C[i+1]
            T2 = nT / 2.0
            m = np.log(ratio)
            # Newton's method for cosh version
            for _ in range(100):
                num = np.cosh(m * (T2 - i))
                den = np.cosh(m * (T2 - i - 1))
                f = num / den - ratio
                d_num = (T2 - i) * np.sinh(m * (T2 - i))
                d_den = (T2 - i - 1) * np.sinh(m * (T2 - i - 1))
                df = (d_num * den - num * d_den) / (den * den)
                if abs(df) > 1e-15:
                    m -= f / df
            t_eff.append(i)
            m_eff.append(m)

    ax.plot(t_eff, m_eff, 'o-', color='#2196F3', markersize=7, linewidth=1.5)

    # Analytic free-field value
    kappa = 0.12
    m0 = 1.0 / (2.0 * kappa) - 4.0
    m_quark = np.arccosh(1.0 + m0)
    m_pion_analytic = 2.0 * m_quark
    ax.axhline(m_pion_analytic, color='#F44336', linestyle='--', linewidth=1.5,
               label=f'Analytic $2m_q = {m_pion_analytic:.3f}$')

    ax.set_xlabel('$t/a$')
    ax.set_ylabel('$m_{\\mathrm{eff}}(t) \\cdot a$')
    ax.set_title(f'Pion Effective Mass (free field, $\\kappa={kappa}$)')
    ax.legend(fontsize=9)
    ax.set_ylim(0.5, 2.5)
    ax.grid(True, alpha=0.3)


def plot_hmc_acceptance(ax):
    """Panel 5: HMC acceptance rate vs step size."""
    results_file = 'hmc_acceptance_scan_results.txt'
    if not os.path.exists(results_file):
        ax.text(0.5, 0.5, 'No HMC scan data', transform=ax.transAxes, ha='center')
        return

    data = read_data(results_file)
    eps = data[:, 1]
    acc = data[:, 2]

    ax.plot(eps, acc, 'o-', color='#2196F3', markersize=8, linewidth=2)
    ax.axhspan(70, 80, alpha=0.15, color='green', label='Target (70-80%)')
    ax.set_xlabel('MD step size $\\epsilon$')
    ax.set_ylabel('Acceptance rate (%)')
    ax.set_title('HMC Acceptance vs Step Size')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_wilson_loop_area_law(ax):
    """Panel 6: Wilson loop W(R,T) showing area law decay."""
    wloop_file = 'SU3_heatbath_beta6.00_8x8x8x16/wilson_loops.dat'
    if not os.path.exists(wloop_file):
        ax.text(0.5, 0.5, 'No Wilson loop data', transform=ax.transAxes, ha='center')
        return

    data = read_data(wloop_file)
    from collections import defaultdict
    wloops = defaultdict(list)
    for row in data:
        cfg, R, T, W = int(row[0]), int(row[1]), int(row[2]), row[3]
        wloops[(R, T)].append(W)

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
    for Ri, R in enumerate([1, 2, 3, 4]):
        Ts, Ws = [], []
        for T in range(1, 7):
            if (R, T) in wloops:
                vals = np.array(wloops[(R, T)])
                Ts.append(T)
                Ws.append(np.mean(vals))
        if Ts:
            ax.semilogy(Ts, Ws, 'o-', color=colors[Ri], markersize=6,
                        linewidth=1.5, label=f'$R={R}$')

    ax.set_xlabel('$T/a$')
    ax.set_ylabel('$\\langle W(R,T) \\rangle$')
    ax.set_title('Wilson Loops (area law)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def main():
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle('Lattice QCD Validation Tests — SU(3) Wilson Gauge',
                 fontsize=16, fontweight='bold', y=0.98)

    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3,
                  top=0.92, bottom=0.08, left=0.06, right=0.97)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    plot_plaquette_vs_beta(ax1)
    plot_polyakov_comparison(ax2)
    plot_static_potential(ax3)
    plot_pion_effective_mass(ax4)
    plot_hmc_acceptance(ax5)
    plot_wilson_loop_area_law(ax6)

    outfile = os.path.join(RUN_DIR, 'lqcd_validation_summary.png')
    plt.savefig(outfile, dpi=150)
    print(f"Saved: {outfile}")
    plt.close()


if __name__ == '__main__':
    main()
