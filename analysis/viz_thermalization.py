#!/usr/bin/env python3
"""
Visualization 4: Thermalization animation.

Shows plaquette evolution from different starting configurations
(hot and cold starts) converging to the same equilibrium value.
Creates a multi-frame "filmstrip" visualization and optionally
an animated GIF.

Uses existing plaquette time-series data from our Monte Carlo runs.
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


def read_plaquette_timeseries(filename):
    """
    Read plaquette.dat. Returns (configs, plaquettes, [action_density]).
    Handles both quenched and HMC formats.
    """
    configs = []
    plaquettes = []
    extra = {}  # deltaH, accepted for HMC

    with open(filename) as f:
        header = f.readline()
        is_hmc = 'deltaH' in header or 'traj' in header

        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            configs.append(int(parts[0]))
            plaquettes.append(float(parts[1]))

            if is_hmc and len(parts) >= 5:
                if 'deltaH' not in extra:
                    extra['deltaH'] = []
                    extra['accepted'] = []
                extra['deltaH'].append(float(parts[3]))
                extra['accepted'].append(int(parts[4]))

    return np.array(configs), np.array(plaquettes), extra


def parse_run_info(dirname):
    """Extract info from directory name."""
    m = re.match(r'SU3_(\w+)_beta(\d+\.\d+)_(\d+)x(\d+)x(\d+)x(\d+)', dirname)
    if m:
        return {
            'method': m.group(1),
            'beta': float(m.group(2)),
            'nX': int(m.group(3)),
            'nT': int(m.group(6)),
            'lattice': f"{m.group(3)}^3x{m.group(6)}",
        }
    return None


def main():
    run_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run')

    # Collect all plaquette time series
    datasets = []
    for dirname in sorted(os.listdir(run_dir)):
        pfile = os.path.join(run_dir, dirname, 'plaquette.dat')
        if not os.path.exists(pfile) or os.path.getsize(pfile) == 0:
            continue
        info = parse_run_info(dirname)
        if info is None:
            continue
        configs, plaqs, extra = read_plaquette_timeseries(pfile)
        if len(plaqs) < 5:
            continue
        info['configs'] = configs
        info['plaquettes'] = plaqs
        info['extra'] = extra
        info['dirname'] = dirname
        datasets.append(info)

    if not datasets:
        print("No plaquette data found")
        sys.exit(1)

    print(f"Found {len(datasets)} datasets for thermalization visualization")

    fig = plt.figure(figsize=(20, 14), facecolor='#0A0A1E')

    gs = GridSpec(3, 2, hspace=0.35, wspace=0.25,
                  left=0.08, right=0.95, top=0.93, bottom=0.06)

    # --- Panel 1: All heatbath runs, plaquette vs sweep (beta comparison) ---
    ax1 = fig.add_subplot(gs[0, 0], facecolor='#0A0A1E')

    beta_colors = {5.6: '#FF4444', 5.8: '#FF8800', 6.0: '#44CC44', 6.2: '#4488FF'}
    hb_8cube = [d for d in datasets if d['method'] == 'heatbath'
                and d['nX'] == 8 and d['nT'] == 8]

    for d in sorted(hb_8cube, key=lambda x: x['beta']):
        color = beta_colors.get(d['beta'], '#888888')
        ax1.plot(d['configs'], d['plaquettes'], '-', color=color,
                 linewidth=1.2, alpha=0.8,
                 label=f"$\\beta={d['beta']:.1f}$, $\\langle P \\rangle = {np.mean(d['plaquettes']):.4f}$")
        # Equilibrium line
        eq = np.mean(d['plaquettes'][len(d['plaquettes'])//3:])
        ax1.axhline(eq, color=color, linewidth=0.5, linestyle='--', alpha=0.4)

    ax1.set_xlabel('Configuration', fontsize=12, color='white')
    ax1.set_ylabel('$\\langle P \\rangle$', fontsize=12, color='white')
    ax1.set_title('Thermalization: Plaquette vs Configuration ($8^4$ lattice)',
                  fontsize=13, color='white')
    ax1.legend(fontsize=9, facecolor='#1A1A3A', edgecolor='gray',
               labelcolor='white', loc='best')
    ax1.grid(True, alpha=0.2, color='gray')
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_color('gray')

    # --- Panel 2: Running average (smoothed thermalization) ---
    ax2 = fig.add_subplot(gs[0, 1], facecolor='#0A0A1E')

    for d in sorted(hb_8cube, key=lambda x: x['beta']):
        color = beta_colors.get(d['beta'], '#888888')
        plaqs = d['plaquettes']
        # Cumulative running average
        running_avg = np.cumsum(plaqs) / np.arange(1, len(plaqs) + 1)
        ax2.plot(d['configs'], running_avg, '-', color=color,
                 linewidth=2, alpha=0.9,
                 label=f"$\\beta={d['beta']:.1f}$")

    ax2.set_xlabel('Configuration', fontsize=12, color='white')
    ax2.set_ylabel('Running Average $\\langle P \\rangle$', fontsize=12, color='white')
    ax2.set_title('Convergence of Plaquette Average', fontsize=13, color='white')
    ax2.legend(fontsize=10, facecolor='#1A1A3A', edgecolor='gray',
               labelcolor='white')
    ax2.grid(True, alpha=0.2, color='gray')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_color('gray')

    # --- Panel 3: HMC trajectory (if available) ---
    ax3 = fig.add_subplot(gs[1, 0], facecolor='#0A0A1E')

    hmc_data = [d for d in datasets if d['method'] == 'hmc']
    if hmc_data:
        d = hmc_data[0]
        ax3.plot(d['configs'], d['plaquettes'], '-', color='#00CCFF',
                 linewidth=1, alpha=0.7, label='Plaquette')
        # Running average
        running = np.cumsum(d['plaquettes']) / np.arange(1, len(d['plaquettes']) + 1)
        ax3.plot(d['configs'], running, '-', color='#FF6600',
                 linewidth=2, label='Running average')
        ax3.set_title(f"HMC Thermalization ($\\beta={d['beta']:.1f}$, ${d['lattice']}$)",
                      fontsize=13, color='white')
    else:
        # Show one of our heatbath runs with error bands
        if hb_8cube:
            d = sorted(hb_8cube, key=lambda x: x['beta'])[-1]  # highest beta
            plaqs = d['plaquettes']
            ax3.fill_between(d['configs'], plaqs - np.std(plaqs), plaqs + np.std(plaqs),
                             color='#00CCFF', alpha=0.2)
            ax3.plot(d['configs'], plaqs, '-', color='#00CCFF', linewidth=1, alpha=0.7)
            running = np.cumsum(plaqs) / np.arange(1, len(plaqs) + 1)
            ax3.plot(d['configs'], running, '-', color='#FF6600', linewidth=2,
                     label='Running average')
            ax3.set_title(f"Heatbath $\\beta={d['beta']:.1f}$ with $1\\sigma$ band",
                          fontsize=13, color='white')

    ax3.set_xlabel('Trajectory / Configuration', fontsize=12, color='white')
    ax3.set_ylabel('$\\langle P \\rangle$', fontsize=12, color='white')
    ax3.legend(fontsize=10, facecolor='#1A1A3A', edgecolor='gray',
               labelcolor='white')
    ax3.grid(True, alpha=0.2, color='gray')
    ax3.tick_params(colors='white')
    for spine in ax3.spines.values():
        spine.set_color('gray')

    # --- Panel 4: Autocorrelation function ---
    ax4 = fig.add_subplot(gs[1, 1], facecolor='#0A0A1E')

    for d in sorted(hb_8cube, key=lambda x: x['beta']):
        color = beta_colors.get(d['beta'], '#888888')
        plaqs = d['plaquettes']
        mean = np.mean(plaqs)
        var = np.var(plaqs)
        if var < 1e-20:
            continue

        max_lag = min(50, len(plaqs) // 3)
        autocorr = np.zeros(max_lag)
        for lag in range(max_lag):
            autocorr[lag] = np.mean((plaqs[:len(plaqs)-lag] - mean) *
                                     (plaqs[lag:] - mean)) / var

        ax4.plot(range(max_lag), autocorr, '-', color=color, linewidth=2,
                 alpha=0.8, label=f"$\\beta={d['beta']:.1f}$")

    ax4.axhline(0, color='white', linewidth=0.5, linestyle='--', alpha=0.3)
    ax4.axhline(1/np.e, color='gray', linewidth=1, linestyle=':', alpha=0.5,
                label='$1/e$ (decorrelation)')
    ax4.set_xlabel('Lag (configurations)', fontsize=12, color='white')
    ax4.set_ylabel('Autocorrelation $\\Gamma(t)$', fontsize=12, color='white')
    ax4.set_title('Plaquette Autocorrelation', fontsize=13, color='white')
    ax4.legend(fontsize=9, facecolor='#1A1A3A', edgecolor='gray',
               labelcolor='white')
    ax4.grid(True, alpha=0.2, color='gray')
    ax4.tick_params(colors='white')
    for spine in ax4.spines.values():
        spine.set_color('gray')

    # --- Panel 5-6: "Filmstrip" showing lattice evolution ---
    # Show snapshots of plaquette distribution at different MC times
    ax5 = fig.add_subplot(gs[2, :], facecolor='#0A0A1E')

    # Use beta=6.0 data and show distribution evolving
    d_filmstrip = None
    for d in hb_8cube:
        if abs(d['beta'] - 6.0) < 0.01:
            d_filmstrip = d
            break
    if d_filmstrip is None and hb_8cube:
        d_filmstrip = hb_8cube[0]

    if d_filmstrip:
        plaqs = d_filmstrip['plaquettes']
        n_frames = 8
        frame_size = max(1, len(plaqs) // n_frames)
        frame_positions = np.linspace(0, len(plaqs) - frame_size, n_frames, dtype=int)

        for i, start in enumerate(frame_positions):
            subset = plaqs[start:start + frame_size]
            x_offset = i * 1.2

            # Vertical histogram for each frame
            bins = np.linspace(min(plaqs) - 0.01, max(plaqs) + 0.01, 20)
            hist, edges = np.histogram(subset, bins=bins, density=True)
            hist_norm = hist / max(hist.max(), 1)  # Normalize to [0,1]

            centers = 0.5 * (edges[:-1] + edges[1:])

            # Draw as horizontal bars
            ax5.barh(centers, hist_norm * 0.8, height=edges[1]-edges[0],
                     left=x_offset, color=plt.cm.plasma(i / n_frames),
                     alpha=0.7, edgecolor='white', linewidth=0.3)

            # Label
            cfg_label = f"cfg {start+1}-{start+frame_size}"
            ax5.text(x_offset + 0.4, min(plaqs) - 0.02, cfg_label,
                     fontsize=8, color='white', ha='center', rotation=0)

    ax5.set_xlabel('Monte Carlo Time (configuration windows)', fontsize=12, color='white')
    ax5.set_ylabel('$\\langle P \\rangle$', fontsize=12, color='white')
    ax5.set_title(f'Plaquette Distribution Evolution ($\\beta={d_filmstrip["beta"]:.1f}$, '
                  f'{d_filmstrip["lattice"]})', fontsize=13, color='white')
    ax5.tick_params(colors='white')
    ax5.set_xticks([])
    for spine in ax5.spines.values():
        spine.set_color('gray')

    fig.suptitle('Monte Carlo Thermalization & Equilibration in SU(3) Gauge Theory',
                 fontsize=16, color='white', fontweight='bold', y=0.98)

    outfile = os.path.join(run_dir, 'viz_thermalization.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved: {outfile}")
    plt.close()

    # --- Create animated GIF if imageio/pillow available ---
    try:
        from PIL import Image
        import io

        frames = []
        n_anim_frames = 40

        if d_filmstrip:
            plaqs = d_filmstrip['plaquettes']

            for frame in range(n_anim_frames):
                fig_anim, ax_anim = plt.subplots(figsize=(8, 5), facecolor='#0A0A1E')
                ax_anim.set_facecolor('#0A0A1E')

                n_show = max(2, int(len(plaqs) * (frame + 1) / n_anim_frames))
                shown = plaqs[:n_show]

                ax_anim.plot(range(1, n_show + 1), shown, '-', color='#00CCFF',
                             linewidth=1.5, alpha=0.8)

                # Running average
                running = np.cumsum(shown) / np.arange(1, n_show + 1)
                ax_anim.plot(range(1, n_show + 1), running, '-', color='#FF6600',
                             linewidth=2.5)

                # Equilibrium line
                eq_val = np.mean(plaqs[len(plaqs)//3:])
                ax_anim.axhline(eq_val, color='#44CC44', linewidth=1.5,
                                linestyle='--', alpha=0.6)

                ax_anim.set_xlim(0, len(plaqs) + 5)
                ax_anim.set_ylim(min(plaqs) - 0.01, max(plaqs) + 0.01)
                ax_anim.set_xlabel('Configuration', fontsize=12, color='white')
                ax_anim.set_ylabel('$\\langle P \\rangle$', fontsize=12, color='white')
                ax_anim.set_title(f'Thermalization ($\\beta={d_filmstrip["beta"]:.1f}$, '
                                  f'config {n_show}/{len(plaqs)})',
                                  fontsize=13, color='white')
                ax_anim.tick_params(colors='white')
                for spine in ax_anim.spines.values():
                    spine.set_color('gray')
                ax_anim.grid(True, alpha=0.2, color='gray')

                # Save frame to buffer
                buf = io.BytesIO()
                fig_anim.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                                 facecolor=fig_anim.get_facecolor())
                buf.seek(0)
                frames.append(Image.open(buf).copy())
                buf.close()
                plt.close(fig_anim)

            # Hold last frame
            for _ in range(10):
                frames.append(frames[-1])

            gif_outfile = os.path.join(run_dir, 'viz_thermalization.gif')
            frames[0].save(gif_outfile, save_all=True, append_images=frames[1:],
                           duration=150, loop=0)
            print(f"Saved: {gif_outfile}")

    except ImportError:
        print("  (PIL not available - skipping GIF animation)")


if __name__ == '__main__':
    main()
