#!/usr/bin/env python3
"""
Visualization: HMC Thermalization - Hot vs Cold Start.

Shows plaquette evolution from hot (random) and cold (identity) starts
converging toward the same equilibrium value from opposite directions.
This is a key validation test for any Monte Carlo code.
"""

import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors


def read_hmc_plaquette(filename):
    """Read HMC plaquette.dat. Returns (traj, plaq, deltaH, accepted)."""
    traj, plaq, dH, acc = [], [], [], []
    with open(filename) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            traj.append(int(parts[0]))
            plaq.append(float(parts[1]))
            if len(parts) >= 5:
                dH.append(float(parts[3]))
                acc.append(int(parts[4]))
    return np.array(traj), np.array(plaq), np.array(dH), np.array(acc)


def main():
    run_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run')

    # Load hot start data
    hot_file = os.path.join(run_dir, 'HMC_therm_hot_beta5.60_8x8x8x8', 'plaquette.dat')
    cold_file = os.path.join(run_dir, 'HMC_therm_cold_v2_beta5.60_8x8x8x8', 'plaquette.dat')

    if not os.path.exists(hot_file):
        print(f"Hot start data not found: {hot_file}")
        sys.exit(1)

    hot_traj, hot_plaq, hot_dH, hot_acc = read_hmc_plaquette(hot_file)

    has_cold = os.path.exists(cold_file)
    if has_cold:
        cold_traj, cold_plaq, cold_dH, cold_acc = read_hmc_plaquette(cold_file)

    # Estimate equilibrium from hot start (last third)
    eq_start = len(hot_plaq) * 2 // 3
    P_eq = np.mean(hot_plaq[eq_start:])
    P_eq_err = np.std(hot_plaq[eq_start:])

    fig = plt.figure(figsize=(20, 16), facecolor='#080818')

    gs = GridSpec(3, 2, hspace=0.32, wspace=0.28,
                  left=0.07, right=0.95, top=0.93, bottom=0.06)

    # Color scheme
    HOT_COLOR = '#FF4444'
    COLD_COLOR = '#4488FF'
    EQ_COLOR = '#44DD44'

    # ===== Panel 1: Main thermalization plot =====
    ax1 = fig.add_subplot(gs[0, :], facecolor='#080818')

    ax1.plot(hot_traj, hot_plaq, '-', color=HOT_COLOR, linewidth=1.2, alpha=0.7,
             label='Hot start ($P_0 \\approx 0$)')

    # Running average for hot
    hot_running = np.cumsum(hot_plaq) / np.arange(1, len(hot_plaq) + 1)
    ax1.plot(hot_traj, hot_running, '-', color=HOT_COLOR, linewidth=2.5, alpha=0.9)

    if has_cold:
        ax1.plot(cold_traj, cold_plaq, '-', color=COLD_COLOR, linewidth=1.2, alpha=0.7,
                 label='Cold start ($P_0 = 1$, after 50 HB sweeps)')

        cold_running = np.cumsum(cold_plaq) / np.arange(1, len(cold_plaq) + 1)
        ax1.plot(cold_traj, cold_running, '-', color=COLD_COLOR, linewidth=2.5, alpha=0.9)

    # Equilibrium band
    ax1.axhspan(P_eq - 2*P_eq_err, P_eq + 2*P_eq_err, color=EQ_COLOR, alpha=0.15)
    ax1.axhline(P_eq, color=EQ_COLOR, linewidth=2, linestyle='--', alpha=0.8,
                label=f'Equilibrium $\\langle P \\rangle = {P_eq:.4f}$')

    ax1.set_xlabel('HMC Trajectory', fontsize=13, color='white')
    ax1.set_ylabel('$\\langle P \\rangle$', fontsize=13, color='white')
    ax1.set_title('HMC Thermalization: Hot vs Cold Start ($\\beta=5.6$, $8^4$)',
                  fontsize=15, color='white', fontweight='bold')
    ax1.legend(fontsize=11, facecolor='#1A1A3A', edgecolor='gray',
               labelcolor='white', loc='center right')
    ax1.grid(True, alpha=0.15, color='gray')
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_color('gray')

    # ===== Panel 2: deltaH distribution (hot) =====
    ax2 = fig.add_subplot(gs[1, 0], facecolor='#080818')

    if len(hot_dH) > 0:
        # Split into early (thermalizing) and late (equilibrium)
        mid = len(hot_dH) // 2
        ax2.hist(hot_dH[:mid], bins=40, color=HOT_COLOR, alpha=0.5, density=True,
                 edgecolor='white', linewidth=0.3, label=f'Traj 1-{mid} (thermalizing)')
        ax2.hist(hot_dH[mid:], bins=40, color='#FF8844', alpha=0.5, density=True,
                 edgecolor='white', linewidth=0.3, label=f'Traj {mid+1}-{len(hot_dH)} (equilibrium)')

        # Overlay Gaussian for equilibrium phase
        eq_dH = hot_dH[mid:]
        x_range = np.linspace(eq_dH.min(), eq_dH.max(), 200)
        mu, sig = np.mean(eq_dH), np.std(eq_dH)
        gauss = np.exp(-(x_range - mu)**2 / (2*sig**2)) / (sig * np.sqrt(2*np.pi))
        ax2.plot(x_range, gauss, '--', color='white', linewidth=1.5, alpha=0.6,
                 label=f'Gaussian ($\\mu={mu:.2f}$, $\\sigma={sig:.2f}$)')

    ax2.axvline(0, color='white', linewidth=0.5, linestyle=':', alpha=0.3)
    ax2.set_xlabel('$\\Delta H$', fontsize=12, color='white')
    ax2.set_ylabel('Probability Density', fontsize=12, color='white')
    ax2.set_title('$\\Delta H$ Distribution (Hot Start)', fontsize=13, color='white')
    ax2.legend(fontsize=9, facecolor='#1A1A3A', edgecolor='gray', labelcolor='white')
    ax2.grid(True, alpha=0.15, color='gray')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_color('gray')

    # ===== Panel 3: Acceptance rate evolution =====
    ax3 = fig.add_subplot(gs[1, 1], facecolor='#080818')

    if len(hot_acc) > 0:
        # Cumulative acceptance rate
        hot_cum_acc = np.cumsum(hot_acc) / np.arange(1, len(hot_acc) + 1) * 100
        ax3.plot(hot_traj, hot_cum_acc, '-', color=HOT_COLOR, linewidth=2,
                 label=f'Hot (final: {hot_cum_acc[-1]:.0f}%)')

    if has_cold and len(cold_acc) > 0:
        cold_cum_acc = np.cumsum(cold_acc) / np.arange(1, len(cold_acc) + 1) * 100
        ax3.plot(cold_traj, cold_cum_acc, '-', color=COLD_COLOR, linewidth=2,
                 label=f'Cold v2 (final: {cold_cum_acc[-1]:.0f}%)')

    ax3.axhline(75, color='#FFD700', linewidth=1, linestyle=':', alpha=0.5,
                label='Target range (75-95%)')
    ax3.axhline(95, color='#FFD700', linewidth=1, linestyle=':', alpha=0.5)
    ax3.axhspan(75, 95, color='#FFD700', alpha=0.05)

    ax3.set_xlabel('HMC Trajectory', fontsize=12, color='white')
    ax3.set_ylabel('Cumulative Acceptance Rate (%)', fontsize=12, color='white')
    ax3.set_title('HMC Acceptance Rate', fontsize=13, color='white')
    ax3.set_ylim(0, 105)
    ax3.legend(fontsize=10, facecolor='#1A1A3A', edgecolor='gray', labelcolor='white')
    ax3.grid(True, alpha=0.15, color='gray')
    ax3.tick_params(colors='white')
    for spine in ax3.spines.values():
        spine.set_color('gray')

    # ===== Panel 4: |deltaH| vs trajectory (integration quality) =====
    ax4 = fig.add_subplot(gs[2, 0], facecolor='#080818')

    if len(hot_dH) > 0:
        ax4.semilogy(hot_traj, np.abs(hot_dH), '.', color=HOT_COLOR, markersize=3,
                     alpha=0.5, label='Hot start')
    if has_cold and len(cold_dH) > 0:
        ax4.semilogy(cold_traj, np.abs(cold_dH), '.', color=COLD_COLOR, markersize=3,
                     alpha=0.5, label='Cold start v2')

    ax4.axhline(1.0, color='#FFD700', linewidth=1, linestyle='--', alpha=0.5,
                label='$|\\Delta H| = 1$ (acceptance ~ 37%)')
    ax4.set_xlabel('HMC Trajectory', fontsize=12, color='white')
    ax4.set_ylabel('$|\\Delta H|$', fontsize=12, color='white')
    ax4.set_title('Leapfrog Integration Quality', fontsize=13, color='white')
    ax4.legend(fontsize=9, facecolor='#1A1A3A', edgecolor='gray', labelcolor='white')
    ax4.grid(True, alpha=0.15, color='gray')
    ax4.tick_params(colors='white')
    for spine in ax4.spines.values():
        spine.set_color('gray')

    # ===== Panel 5: Plaquette distribution comparison =====
    ax5 = fig.add_subplot(gs[2, 1], facecolor='#080818')

    # Use last third of each run (closest to equilibrium)
    hot_eq = hot_plaq[len(hot_plaq)*2//3:]
    ax5.hist(hot_eq, bins=25, color=HOT_COLOR, alpha=0.6, density=True,
             edgecolor='white', linewidth=0.3,
             label=f'Hot (last {len(hot_eq)} traj)')

    if has_cold:
        cold_eq = cold_plaq[len(cold_plaq)*2//3:]
        ax5.hist(cold_eq, bins=25, color=COLD_COLOR, alpha=0.6, density=True,
                 edgecolor='white', linewidth=0.3,
                 label=f'Cold (last {len(cold_eq)} traj)')

    ax5.axvline(P_eq, color=EQ_COLOR, linewidth=2, linestyle='--', alpha=0.8)
    ax5.set_xlabel('$\\langle P \\rangle$', fontsize=12, color='white')
    ax5.set_ylabel('Probability Density', fontsize=12, color='white')
    ax5.set_title('Equilibrium Plaquette Distribution', fontsize=13, color='white')
    ax5.legend(fontsize=10, facecolor='#1A1A3A', edgecolor='gray', labelcolor='white')
    ax5.grid(True, alpha=0.15, color='gray')
    ax5.tick_params(colors='white')
    for spine in ax5.spines.values():
        spine.set_color('gray')

    fig.suptitle('Hybrid Monte Carlo Validation: Thermalization from Opposite Starts',
                 fontsize=17, color='white', fontweight='bold', y=0.98)

    outfile = os.path.join(run_dir, 'viz_hmc_thermalization.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved: {outfile}")
    plt.close()


if __name__ == '__main__':
    main()
