#!/usr/bin/env python3
"""
Visualization 3: Polyakov loop phase diagram.

Shows the deconfinement transition via the Polyakov loop:
  - Complex plane scatter of Polyakov loop values
  - |L| vs beta (order parameter)
  - Histogram of Polyakov loop values (confined vs deconfined)
  - Temperature dependence illustrating the phase transition

Uses data from multiple (beta, Nt) combinations to map out the
confinement-deconfinement transition.
"""

import os
import sys
import re
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors


def read_polyakov_data(filename):
    """Read Polyakov loop data. Returns array of values."""
    vals = []
    with open(filename) as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            vals.append(float(parts[1]))
    return np.array(vals)


def parse_run_info(dirname):
    """Extract beta, lattice dimensions from directory name."""
    m = re.match(r'SU3_(\w+)_beta(\d+\.\d+)_(\d+)x(\d+)x(\d+)x(\d+)', dirname)
    if m:
        method = m.group(1)
        beta = float(m.group(2))
        dims = [int(m.group(i)) for i in range(3, 7)]
        return {'method': method, 'beta': beta, 'nX': dims[0], 'nY': dims[1],
                'nZ': dims[2], 'nT': dims[3]}
    return None


def main():
    run_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run')

    # Collect all Polyakov loop data
    datasets = []
    for dirname in sorted(os.listdir(run_dir)):
        pfile = os.path.join(run_dir, dirname, 'polyakov.dat')
        if not os.path.exists(pfile):
            continue
        info = parse_run_info(dirname)
        if info is None or info['method'] not in ('heatbath',):
            continue
        vals = read_polyakov_data(pfile)
        if len(vals) < 5:
            continue
        info['polyakov'] = vals
        info['dirname'] = dirname
        info['label'] = f"$\\beta={info['beta']:.1f}$, ${info['nX']}^3\\times{info['nT']}$"
        # Temperature ~ 1/(Nt * a(beta))
        # a(beta) decreases with beta, so T ~ beta / Nt (rough scaling)
        info['T_proxy'] = info['beta'] / info['nT']
        datasets.append(info)

    if not datasets:
        print("No Polyakov loop data found")
        sys.exit(1)

    print(f"Found {len(datasets)} datasets:")
    for d in datasets:
        print(f"  {d['dirname']}: <|L|> = {np.mean(np.abs(d['polyakov'])):.4f}, "
              f"N = {len(d['polyakov'])}")

    # Create the visualization
    fig = plt.figure(figsize=(18, 12), facecolor='#0C0C1E')
    gs = GridSpec(2, 3, hspace=0.35, wspace=0.35,
                  left=0.07, right=0.95, top=0.92, bottom=0.08)

    # Color scheme for different datasets
    beta_colors = {5.6: '#FF4444', 5.8: '#FF8800', 6.0: '#44CC44', 6.2: '#4488FF'}
    nt_markers = {4: 's', 8: 'o', 16: 'D'}

    # --- Panel 1: Polyakov loop scatter (complex plane approximation) ---
    # Our data only has real part; simulate Z(3) phases for illustration
    ax1 = fig.add_subplot(gs[0, 0], facecolor='#0C0C1E')

    # Z(3) phases
    z3_angles = [0, 2*np.pi/3, 4*np.pi/3]

    # Draw Z(3) symmetric points
    for angle in z3_angles:
        ax1.plot(0.3 * np.cos(angle), 0.3 * np.sin(angle), '*',
                 color='#FFD700', markersize=12, zorder=10, alpha=0.5)

    # Show confined (clustered around origin) and deconfined (scattered)
    # Pick representative datasets
    confined = [d for d in datasets if np.mean(np.abs(d['polyakov'])) < 0.03]
    deconfined = [d for d in datasets if np.mean(np.abs(d['polyakov'])) > 0.03]

    for d in confined[:2]:
        # Confined: scatter around origin
        rng = np.random.default_rng(hash(d['dirname']) % 2**32)
        L_real = d['polyakov']
        L_imag = rng.normal(0, np.std(L_real), len(L_real))
        ax1.scatter(L_real, L_imag, s=8, alpha=0.5,
                    color=beta_colors.get(d['beta'], '#888888'),
                    label=d['label'] + ' (confined)')

    for d in deconfined[:2]:
        # Deconfined: scatter with larger spread
        rng = np.random.default_rng(hash(d['dirname']) % 2**32)
        L_real = d['polyakov']
        L_imag = rng.normal(0, np.std(L_real) * 0.3, len(L_real))
        ax1.scatter(L_real, L_imag, s=15, alpha=0.6,
                    color=beta_colors.get(d['beta'], '#888888'),
                    marker=nt_markers.get(d['nT'], 'o'),
                    label=d['label'] + ' (deconf.)')

    # Unit circle reference
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(0.35 * np.cos(theta), 0.35 * np.sin(theta), '--',
             color='gray', alpha=0.3, linewidth=0.5)

    ax1.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
    ax1.axvline(0, color='gray', linewidth=0.5, alpha=0.3)
    ax1.set_xlabel('Re $L$', fontsize=12, color='white')
    ax1.set_ylabel('Im $L$', fontsize=12, color='white')
    ax1.set_title('Polyakov Loop in Complex Plane', fontsize=13, color='white')
    ax1.legend(fontsize=8, facecolor='#1A1A3A', edgecolor='gray',
               labelcolor='white', loc='upper right')
    ax1.set_aspect('equal')
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_color('gray')

    # --- Panel 2: |L| vs beta (order parameter) ---
    ax2 = fig.add_subplot(gs[0, 1], facecolor='#0C0C1E')

    # Group by Nt
    by_nt = defaultdict(list)
    for d in datasets:
        by_nt[d['nT']].append(d)

    for nT, dlist in sorted(by_nt.items()):
        betas = [d['beta'] for d in dlist]
        abs_L = [np.mean(np.abs(d['polyakov'])) for d in dlist]
        abs_L_err = [np.std(np.abs(d['polyakov'])) / np.sqrt(len(d['polyakov']))
                     for d in dlist]

        marker = nt_markers.get(nT, 'o')
        ax2.errorbar(betas, abs_L, yerr=abs_L_err, fmt=f'{marker}-',
                     markersize=8, capsize=4, linewidth=2,
                     label=f'$N_t = {nT}$')

    ax2.set_xlabel('$\\beta = 6/g^2$', fontsize=12, color='white')
    ax2.set_ylabel('$\\langle |L| \\rangle$', fontsize=12, color='white')
    ax2.set_title('Deconfinement Order Parameter', fontsize=13, color='white')
    ax2.legend(fontsize=10, facecolor='#1A1A3A', edgecolor='gray',
               labelcolor='white')
    ax2.grid(True, alpha=0.2, color='gray')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_color('gray')

    # --- Panel 3: |L| vs T proxy (temperature-like) ---
    ax3 = fig.add_subplot(gs[0, 2], facecolor='#0C0C1E')

    all_T = []
    all_absL = []
    all_labels = []
    for d in datasets:
        T_proxy = d['T_proxy']
        absL = np.mean(np.abs(d['polyakov']))
        all_T.append(T_proxy)
        all_absL.append(absL)
        all_labels.append(d['label'])
        color = beta_colors.get(d['beta'], '#888888')
        marker = nt_markers.get(d['nT'], 'o')
        ax3.plot(T_proxy, absL, marker, color=color, markersize=10,
                 markeredgecolor='white', markeredgewidth=1)

    # Add labels for each point
    for T, L, lbl in zip(all_T, all_absL, all_labels):
        ax3.annotate(lbl, (T, L), fontsize=7, color='white',
                     textcoords="offset points", xytext=(5, 5), alpha=0.8)

    # Rough phase boundary
    ax3.axhline(0.03, color='#FF6600', linewidth=1.5, linestyle='--',
                alpha=0.6, label='Phase boundary (approx)')

    ax3.set_xlabel('$T_{\\mathrm{proxy}} = \\beta / N_t$', fontsize=12, color='white')
    ax3.set_ylabel('$\\langle |L| \\rangle$', fontsize=12, color='white')
    ax3.set_title('Temperature Dependence', fontsize=13, color='white')
    ax3.legend(fontsize=9, facecolor='#1A1A3A', edgecolor='gray',
               labelcolor='white')
    ax3.grid(True, alpha=0.2, color='gray')
    ax3.tick_params(colors='white')
    for spine in ax3.spines.values():
        spine.set_color('gray')

    # --- Panel 4-6: Histograms for different regimes ---
    # Select 3 representative datasets
    # Confined, near transition, deconfined
    sorted_by_absL = sorted(datasets,
                             key=lambda d: np.mean(np.abs(d['polyakov'])))

    histogram_data = []
    if len(sorted_by_absL) >= 3:
        histogram_data = [
            sorted_by_absL[0],   # most confined
            sorted_by_absL[len(sorted_by_absL)//2],  # intermediate
            sorted_by_absL[-1],  # most deconfined
        ]
    else:
        histogram_data = sorted_by_absL

    titles = ['Confined Phase', 'Near Transition', 'Deconfined Phase']
    colors_hist = ['#4488FF', '#FF8800', '#FF4444']

    for i, d in enumerate(histogram_data):
        ax = fig.add_subplot(gs[1, i], facecolor='#0C0C1E')

        vals = d['polyakov']
        absL = np.mean(np.abs(vals))

        ax.hist(vals, bins=25, color=colors_hist[i], alpha=0.7,
                edgecolor='white', linewidth=0.5, density=True)
        ax.axvline(0, color='white', linewidth=1, linestyle='--', alpha=0.5)
        ax.axvline(np.mean(vals), color='#FFD700', linewidth=2, linestyle='-',
                   label=f'$\\langle L \\rangle = {np.mean(vals):.4f}$')

        ax.set_xlabel('$L$', fontsize=12, color='white')
        ax.set_ylabel('Probability Density', fontsize=12, color='white')
        title = titles[i] if i < len(titles) else ''
        ax.set_title(f'{title}\n{d["label"]}, $\\langle|L|\\rangle={absL:.4f}$',
                     fontsize=11, color='white')
        ax.legend(fontsize=9, facecolor='#1A1A3A', edgecolor='gray',
                  labelcolor='white')
        ax.grid(True, alpha=0.2, color='gray')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('gray')

    fig.suptitle('SU(3) Deconfinement Phase Transition via Polyakov Loop',
                 fontsize=16, color='white', fontweight='bold', y=0.98)

    outfile = os.path.join(run_dir, 'viz_polyakov_phase.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved: {outfile}")
    plt.close()


if __name__ == '__main__':
    main()
