#!/usr/bin/env python3
"""
Visualization 2: Chromoelectric flux tube between quark-antiquark pair.

Uses Wilson loop data W(R,T) to extract the static potential V(R) and
visualize the flux tube connecting a quark and antiquark. Shows:
  - The string-like confining flux tube in 2D cross-section
  - V(R) with Cornell fit overlaid
  - Action density profile of the flux tube
"""

import os
import sys
import numpy as np
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec


def read_wilson_loops(filename):
    """Read W(R,T) data. Returns dict: (R,T) -> list of values per config."""
    data = defaultdict(list)
    with open(filename) as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            cfg = int(parts[0])
            R = int(parts[1])
            T = int(parts[2])
            W = float(parts[3])
            data[(R, T)].append(W)
    return data


def extract_potential(wilson_data, T_extract=5):
    """Extract V(R) = -log(W(R,T)/W(R,T-1))."""
    Rs = sorted(set(R for (R, T) in wilson_data.keys()))
    V = {}
    for R in Rs:
        if (R, T_extract) in wilson_data and (R, T_extract-1) in wilson_data:
            W_T = np.mean(wilson_data[(R, T_extract)])
            W_Tm1 = np.mean(wilson_data[(R, T_extract-1)])
            if W_T > 0 and W_Tm1 > 0:
                V[R] = -np.log(W_T / W_Tm1)
    return V


def cornell_fit(Rs, Vs):
    """Fit V(R) = V0 + sigma*R - alpha/R."""
    from scipy.optimize import curve_fit
    def cornell(R, V0, sigma, alpha):
        return V0 + sigma * R - alpha / R
    try:
        popt, pcov = curve_fit(cornell, Rs, Vs, p0=[0.5, 0.05, 0.3], maxfev=5000)
        return popt
    except:
        return None


def draw_flux_tube(ax, R_sep=4, sigma=0.05, tube_width=1.2):
    """
    Draw a 2D visualization of the chromoelectric flux tube
    between a quark and antiquark separated by distance R_sep.
    Uses a Gaussian profile for the flux tube.
    """
    nx, ny = 200, 100
    x = np.linspace(-1, R_sep + 1, nx)
    y = np.linspace(-2.5, 2.5, ny)
    X, Y = np.meshgrid(x, y)

    # Flux tube: Gaussian transverse profile along the line connecting q and qbar
    # Action density ~ exp(-y^2 / (2*w^2)) in the region between quarks
    # with Coulomb-like enhancement near the charges

    field = np.zeros_like(X)

    # String part: uniform tube between quarks
    mask_between = (X >= 0) & (X <= R_sep)
    field += np.exp(-Y**2 / (2 * tube_width**2)) * mask_between

    # Coulomb part near quarks: 1/r^2 field
    r_q = np.sqrt(X**2 + Y**2 + 0.3**2)
    r_qbar = np.sqrt((X - R_sep)**2 + Y**2 + 0.3**2)
    coulomb = 0.4 / r_q**2 + 0.4 / r_qbar**2
    field += coulomb * 0.5

    # Normalize
    field = field / field.max()

    # Custom colormap: dark blue -> blue -> cyan -> white (like energy density)
    colors_list = ['#000020', '#000060', '#0040A0', '#0080FF', '#00C0FF',
                   '#60FFE0', '#FFFF80', '#FF8000', '#FF0000', '#FFFFFF']
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('flux', colors_list, N=n_bins)

    im = ax.pcolormesh(X, Y, field, cmap=cmap, shading='gouraud', vmin=0, vmax=1)

    # Draw quark and antiquark
    quark_color = '#FF3333'
    antiquark_color = '#3333FF'

    # Quark markers
    ax.plot(0, 0, 'o', color=quark_color, markersize=18, markeredgecolor='white',
            markeredgewidth=2, zorder=5)
    ax.text(0, -1.8, '$q$', fontsize=16, color='white', ha='center',
            fontweight='bold')

    ax.plot(R_sep, 0, 'o', color=antiquark_color, markersize=18,
            markeredgecolor='white', markeredgewidth=2, zorder=5)
    ax.text(R_sep, -1.8, '$\\bar{q}$', fontsize=16, color='white', ha='center',
            fontweight='bold')

    ax.set_xlim(-1, R_sep + 1)
    ax.set_ylim(-2.5, 2.5)
    ax.set_facecolor('#000010')
    ax.set_aspect('equal')
    ax.set_xlabel('$R/a$', fontsize=12, color='white')
    ax.set_ylabel('Transverse $y/a$', fontsize=12, color='white')
    ax.tick_params(colors='white')

    return im


def main():
    run_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run')

    # Read Wilson loop data
    wl_file = os.path.join(run_dir, 'SU3_heatbath_beta6.00_8x8x8x16', 'wilson_loops.dat')
    if not os.path.exists(wl_file):
        print(f"Wilson loop data not found: {wl_file}")
        sys.exit(1)

    wilson_data = read_wilson_loops(wl_file)

    # Extract potential at different T values
    potentials = {}
    for T in [3, 4, 5, 6]:
        V = extract_potential(wilson_data, T_extract=T)
        if V:
            potentials[T] = V

    fig = plt.figure(figsize=(18, 10), facecolor='#0A0A2A')

    gs = GridSpec(2, 2, hspace=0.35, wspace=0.3,
                  left=0.08, right=0.95, top=0.92, bottom=0.08)

    # --- Panel 1: Flux tube visualization (large, top spanning both columns) ---
    ax_tube = fig.add_subplot(gs[0, :])
    im = draw_flux_tube(ax_tube, R_sep=4, sigma=0.05, tube_width=1.0)
    ax_tube.set_title('Chromoelectric Flux Tube Between Static Quarks',
                      fontsize=15, color='white', pad=12, fontweight='bold')

    # Colorbar for flux tube
    cbar = fig.colorbar(im, ax=ax_tube, shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label('Action Density (normalized)', fontsize=11, color='white')
    cbar.ax.tick_params(colors='white')

    # --- Panel 2: Static potential V(R) ---
    ax_pot = fig.add_subplot(gs[1, 0])
    ax_pot.set_facecolor('#0A0A2A')

    colors_T = {3: '#FF6666', 4: '#FFAA33', 5: '#66FF66', 6: '#6666FF'}

    best_T = max(potentials.keys()) if potentials else 5
    for T, V_dict in sorted(potentials.items()):
        Rs = sorted(V_dict.keys())
        Vs = [V_dict[R] for R in Rs]
        marker = 'o' if T == best_T else 's'
        lw = 2.5 if T == best_T else 1.0
        ax_pot.plot(Rs, Vs, f'{marker}-', color=colors_T.get(T, 'white'),
                    markersize=7, linewidth=lw, label=f'T={T}', alpha=0.9)

    # Cornell fit on best T
    if best_T in potentials:
        V_best = potentials[best_T]
        Rs_fit = np.array(sorted(V_best.keys()))
        Vs_fit = np.array([V_best[R] for R in Rs_fit])

        try:
            params = cornell_fit(Rs_fit, Vs_fit)
            if params is not None:
                V0, sigma, alpha = params
                R_fine = np.linspace(0.5, max(Rs_fit) + 0.5, 100)
                V_cornell = V0 + sigma * R_fine - alpha / R_fine
                ax_pot.plot(R_fine, V_cornell, '--', color='white', linewidth=1.5,
                            alpha=0.7, label=f'Cornell: $\\sigma a^2={sigma:.3f}$')
        except ImportError:
            pass

    ax_pot.set_xlabel('$R/a$', fontsize=12, color='white')
    ax_pot.set_ylabel('$V(R) \\cdot a$', fontsize=12, color='white')
    ax_pot.set_title('Static Quark Potential', fontsize=13, color='white')
    ax_pot.legend(fontsize=9, facecolor='#1A1A3A', edgecolor='gray',
                  labelcolor='white')
    ax_pot.grid(True, alpha=0.2, color='gray')
    ax_pot.tick_params(colors='white')
    for spine in ax_pot.spines.values():
        spine.set_color('gray')

    # --- Panel 3: Wilson loop area law ---
    ax_area = fig.add_subplot(gs[1, 1])
    ax_area.set_facecolor('#0A0A2A')

    # W(R,T) ~ exp(-V(R)*T) ~ exp(-sigma*R*T) for large R,T
    # Plot log(W) vs area R*T
    areas = []
    logW = []
    for (R, T), vals in wilson_data.items():
        avg_W = np.mean(vals)
        if avg_W > 1e-10:
            areas.append(R * T)
            logW.append(np.log(avg_W))

    if areas:
        ax_area.scatter(areas, logW, c='#00CCFF', s=25, alpha=0.7, zorder=3)

        # Linear fit for area law: log(W) = -sigma * A + const
        A = np.array(areas)
        lW = np.array(logW)
        coeffs = np.polyfit(A, lW, 1)
        A_fine = np.linspace(min(A), max(A), 100)
        ax_area.plot(A_fine, np.polyval(coeffs, A_fine), '--', color='#FF6666',
                     linewidth=2, label=f'$\\sigma_{{\\mathrm{{string}}}} a^2 = {-coeffs[0]:.4f}$')

    ax_area.set_xlabel('Area $R \\times T$ $(a^2)$', fontsize=12, color='white')
    ax_area.set_ylabel('$\\ln\\langle W(R,T)\\rangle$', fontsize=12, color='white')
    ax_area.set_title('Wilson Loop Area Law (Confinement)', fontsize=13, color='white')
    ax_area.legend(fontsize=10, facecolor='#1A1A3A', edgecolor='gray',
                   labelcolor='white')
    ax_area.grid(True, alpha=0.2, color='gray')
    ax_area.tick_params(colors='white')
    for spine in ax_area.spines.values():
        spine.set_color('gray')

    fig.suptitle('SU(3) Confinement: Flux Tube & Static Potential ($\\beta=6.0$, $8^3\\times 16$)',
                 fontsize=16, color='white', fontweight='bold', y=0.98)

    outfile = os.path.join(run_dir, 'viz_flux_tube.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved: {outfile}")
    plt.close()


if __name__ == '__main__':
    main()
