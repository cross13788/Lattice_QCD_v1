#!/usr/bin/env python3
"""
Visualization 1: 3D Lattice slice with plaquette-colored faces.

Renders a 3D view of a lattice time-slice showing plaquettes as
colored squares. Color maps plaquette value from ordered (blue, P~1)
to disordered (red, P~0). Simulates a plaquette field using the
measured distribution from our Monte Carlo data.

Creates a publication-quality figure showing the lattice structure
that is immediately recognizable as "QCD in a box."
"""

import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors


def generate_plaquette_field(L, avg_plaq, sigma=0.05):
    """
    Generate a synthetic plaquette field with the correct average
    and fluctuations for a given lattice size and coupling.
    Each plaquette is in [0,1], with distribution peaked at avg_plaq.
    """
    rng = np.random.default_rng(42)
    # Beta distribution parameterized by mean and variance
    mean = avg_plaq
    var = sigma**2
    # Beta distribution: a = mean * ((mean*(1-mean)/var) - 1)
    factor = mean * (1 - mean) / var - 1
    if factor <= 0:
        factor = 10.0
    a = mean * factor
    b = (1 - mean) * factor

    # 3 orientations of plaquettes at each site of a spatial slice
    plaq = rng.beta(a, b, size=(L, L, L, 3))
    return plaq


def draw_lattice_slice(ax, plaq_field, L, orientation=0, slice_idx=None,
                       cmap='RdYlBu', alpha=0.85):
    """
    Draw plaquettes as colored squares on a 3D lattice.
    orientation: 0=xy, 1=xz, 2=yz planes
    """
    if slice_idx is None:
        slice_idx = L // 2

    vmin, vmax = 0.4, 0.8
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cm = plt.get_cmap(cmap)

    faces = []
    colors = []

    for i in range(L):
        for j in range(L):
            for k in range(L):
                val = plaq_field[i, j, k, orientation]
                color = cm(norm(val))

                # Draw face depending on orientation
                if orientation == 0:  # xy plane
                    if k != slice_idx:
                        continue
                    verts = [
                        [i, j, k],
                        [i+1, j, k],
                        [i+1, j+1, k],
                        [i, j+1, k],
                    ]
                elif orientation == 1:  # xz plane
                    if j != slice_idx:
                        continue
                    verts = [
                        [i, j, k],
                        [i+1, j, k],
                        [i+1, j, k+1],
                        [i, j, k+1],
                    ]
                else:  # yz plane
                    if i != slice_idx:
                        continue
                    verts = [
                        [i, j, k],
                        [i, j+1, k],
                        [i, j+1, k+1],
                        [i, j, k+1],
                    ]

                faces.append(verts)
                colors.append(color)

    poly = Poly3DCollection(faces, alpha=alpha, linewidths=0.3, edgecolors='gray')
    poly.set_facecolors(colors)
    ax.add_collection3d(poly)

    return norm, cm


def main():
    run_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run')

    # Read actual plaquette averages from our data
    plaq_data = {}
    for dirname in os.listdir(run_dir):
        if not dirname.startswith('SU3_heatbath_beta'):
            continue
        pfile = os.path.join(run_dir, dirname, 'plaquette.dat')
        if not os.path.exists(pfile):
            continue
        vals = []
        with open(pfile) as f:
            for line in f:
                if line.startswith('#') or line.strip() == '':
                    continue
                parts = line.split()
                vals.append(float(parts[1]))
        if vals:
            # Extract beta from dirname
            import re
            m = re.search(r'beta(\d+\.\d+)', dirname)
            if m:
                beta = float(m.group(1))
                lattice = dirname.split('_')[-1]
                plaq_data[(beta, lattice)] = np.mean(vals)

    L = 8  # Lattice size for visualization

    # Use beta=6.0 (ordered) and beta=5.6 (more disordered) data
    avg_plaq_ordered = plaq_data.get((6.0, '8x8x8x8'), 0.593)
    avg_plaq_disordered = plaq_data.get((5.6, '8x8x8x8'), 0.519)

    fig = plt.figure(figsize=(18, 8))

    # --- Panel 1: Ordered (weak coupling, beta=6.0) ---
    ax1 = fig.add_subplot(121, projection='3d')

    plaq_ordered = generate_plaquette_field(L, avg_plaq_ordered, sigma=0.03)

    # Draw all 3 orientations of plaquettes on selected slices
    for orient in range(3):
        norm, cm = draw_lattice_slice(ax1, plaq_ordered, L,
                                       orientation=orient, slice_idx=L//2,
                                       alpha=0.7)

    ax1.set_xlim(0, L)
    ax1.set_ylim(0, L)
    ax1.set_zlim(0, L)
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.set_zlabel('z', fontsize=11)
    ax1.set_title(f'Weak Coupling ($\\beta=6.0$, $\\langle P\\rangle={avg_plaq_ordered:.3f}$)',
                  fontsize=13, pad=15)
    ax1.view_init(elev=25, azim=45)

    # --- Panel 2: Disordered (strong coupling, beta=5.6) ---
    ax2 = fig.add_subplot(122, projection='3d')

    plaq_disordered = generate_plaquette_field(L, avg_plaq_disordered, sigma=0.06)

    for orient in range(3):
        norm, cm = draw_lattice_slice(ax2, plaq_disordered, L,
                                       orientation=orient, slice_idx=L//2,
                                       alpha=0.7)

    ax2.set_xlim(0, L)
    ax2.set_ylim(0, L)
    ax2.set_zlim(0, L)
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('y', fontsize=11)
    ax2.set_zlabel('z', fontsize=11)
    ax2.set_title(f'Strong Coupling ($\\beta=5.6$, $\\langle P\\rangle={avg_plaq_disordered:.3f}$)',
                  fontsize=13, pad=15)
    ax2.view_init(elev=25, azim=45)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlBu', norm=mcolors.Normalize(vmin=0.4, vmax=0.8))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.6, aspect=20, pad=0.08)
    cbar.set_label('Plaquette Value $P_{\\mu\\nu}(x)$', fontsize=12)

    fig.suptitle('SU(3) Lattice QCD: Plaquette Field on Spatial Slices ($8^3$ lattice)',
                 fontsize=15, fontweight='bold', y=0.98)

    outfile = os.path.join(run_dir, 'viz_lattice_3d.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {outfile}")
    plt.close()


if __name__ == '__main__':
    main()
