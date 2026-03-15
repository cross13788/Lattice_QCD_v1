#!/usr/bin/env python3
"""
Scale setting from Wilson flow and static potential.

Extracts:
  - t0:  where t^2 <E(t)> = 0.3  (BMW definition)
  - w0:  where t * d/dt [t^2 <E(t)>] = 0.3
  - r0:  Sommer parameter from Cornell fit: r0^2 F(r0) = 1.65

Usage:
  python3 scale_setting.py [flow_data_dir] [potential_data_dir]
  python3 scale_setting.py   (auto-detect from run/)
"""

import os
import sys
import numpy as np
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jackknife_analysis import jackknife

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


def read_flow_data(filename):
    """
    Read wilson_flow.dat file.
    Returns: (t, t2Eplaq, t2Eclov, Eplaq, Eclov)
    """
    t, t2ep, t2ec, ep, ec = [], [], [], [], []
    with open(filename) as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            t.append(float(parts[0]))
            t2ep.append(float(parts[1]))
            t2ec.append(float(parts[2]))
            ep.append(float(parts[3]))
            ec.append(float(parts[4]))
    return np.array(t), np.array(t2ep), np.array(t2ec), np.array(ep), np.array(ec)


def extract_t0(t, t2E, threshold=0.3):
    """
    Extract t0 where t^2 <E(t)> = threshold via linear interpolation.
    Returns t0 or None if not found.
    """
    for i in range(1, len(t)):
        if t2E[i-1] < threshold <= t2E[i]:
            # Linear interpolation
            frac = (threshold - t2E[i-1]) / (t2E[i] - t2E[i-1])
            return t[i-1] + frac * (t[i] - t[i-1])
    return None


def extract_w0(t, t2E, threshold=0.3):
    """
    Extract w0 where W(t) = t * d/dt [t^2 <E(t)>] = threshold.
    w0^2 = t at which W(t) = threshold.
    Returns w0 or None.
    """
    if len(t) < 3:
        return None

    # Compute W(t) = t * d(t^2E)/dt using centered differences
    dt = np.diff(t)
    W = np.zeros(len(t))
    for i in range(1, len(t) - 1):
        W[i] = t[i] * (t2E[i+1] - t2E[i-1]) / (t[i+1] - t[i-1])

    # Find crossing
    for i in range(1, len(t) - 1):
        if W[i-1] < threshold <= W[i]:
            frac = (threshold - W[i-1]) / (W[i] - W[i-1])
            t_cross = t[i-1] + frac * (t[i] - t[i-1])
            return np.sqrt(t_cross)

    return None


def extract_r0_from_cornell(alpha, sigma):
    """
    Extract Sommer parameter r0 from Cornell fit parameters.
    r0^2 * F(r0) = 1.65 where F(r) = dV/dr = alpha/r^2 + sigma
    => r0^2 * (alpha/r0^2 + sigma) = 1.65
    => alpha + sigma * r0^2 = 1.65
    => r0 = sqrt((1.65 - alpha) / sigma)
    """
    if sigma <= 0:
        return None
    val = (1.65 - alpha) / sigma
    if val <= 0:
        return None
    return np.sqrt(val)


def main():
    run_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run')

    print("=" * 65)
    print("  Scale Setting Analysis")
    print("  Wilson flow (t0, w0) and Sommer parameter (r0)")
    print("=" * 65)

    # Find flow data
    if len(sys.argv) > 1:
        dirs = [sys.argv[i] for i in range(1, len(sys.argv))]
    else:
        dirs = sorted(glob.glob(os.path.join(run_dir, 'SU3_*')))

    flow_results = []
    for d in dirs:
        flow_file = os.path.join(d, 'wilson_flow.dat')
        if not os.path.exists(flow_file):
            continue

        basename = os.path.basename(d)
        print(f"\n--- {basename} ---")

        t, t2ep, t2ec, ep, ec = read_flow_data(flow_file)

        # Extract t0 from both plaquette and clover definitions
        t0_plaq = extract_t0(t, t2ep, 0.3)
        t0_clov = extract_t0(t, t2ec, 0.3)

        # Extract w0
        w0_plaq = extract_w0(t, t2ep, 0.3)
        w0_clov = extract_w0(t, t2ec, 0.3)

        print(f"  t0 (plaquette): {t0_plaq:.4f}" if t0_plaq else "  t0 (plaquette): not reached")
        print(f"  t0 (clover):    {t0_clov:.4f}" if t0_clov else "  t0 (clover):    not reached")
        print(f"  w0 (plaquette): {w0_plaq:.4f}" if w0_plaq else "  w0 (plaquette): not reached")
        print(f"  w0 (clover):    {w0_clov:.4f}" if w0_clov else "  w0 (clover):    not reached")

        # Physical scale (approximate)
        if t0_clov:
            # sqrt(8 t0) ~ 0.3 fm at beta=6.0
            sqrt_8t0 = np.sqrt(8.0 * t0_clov)
            a_fm = 0.3 / sqrt_8t0  # very rough
            print(f"  sqrt(8t0) = {sqrt_8t0:.4f} => a ~ {a_fm:.4f} fm (rough)")

        flow_results.append({
            'dir': basename,
            't': t, 't2ep': t2ep, 't2ec': t2ec,
            't0_plaq': t0_plaq, 't0_clov': t0_clov,
            'w0_plaq': w0_plaq, 'w0_clov': w0_clov,
        })

    # Try to extract r0 from static potential
    for d in dirs:
        pot_dirs = sorted(glob.glob(os.path.join(d, 'static_potential.dat')))
        # Also check analysis output
        wloop_file = os.path.join(d, 'wilson_loops.dat')
        if not os.path.exists(wloop_file):
            continue

        try:
            from static_potential import read_wilson_loops, extract_potential, fit_cornell
            wloop_data = read_wilson_loops(wloop_file)
            T_max = max(k[1] for k in wloop_data.keys())

            best_T_pair = None
            for T1 in range(max(2, T_max - 2), 1, -1):
                results = extract_potential(wloop_data, (T1, T1 + 1))
                if len(results) >= 3:
                    best_T_pair = (T1, T1 + 1)
                    break

            if best_T_pair:
                results = extract_potential(wloop_data, best_T_pair)
                Rs = [r[0] for r in results]
                Vs = [r[1] for r in results]
                Verrs = [r[2] for r in results]
                alpha, sigma, c, chi2 = fit_cornell(Rs, Vs, Verrs)

                r0 = extract_r0_from_cornell(alpha, sigma)
                basename = os.path.basename(d)
                print(f"\n  {basename}: Cornell fit alpha={alpha:.4f}, sigma={sigma:.4f}")
                if r0:
                    print(f"    r0/a = {r0:.4f}")
                    print(f"    (Physical: r0 ~ 0.49 fm)")
        except Exception as e:
            pass

    # Plot flow data
    if HAS_PLOT and flow_results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        colors = ['#1565C0', '#2E7D32', '#E65100', '#6A1B9A']
        for i, res in enumerate(flow_results):
            c = colors[i % len(colors)]
            label = res['dir'].split('_')[-1] if '_' in res['dir'] else res['dir']

            axes[0].plot(res['t'], res['t2ep'], '-', color=c, linewidth=1.5,
                         label=f'{label} plaq')
            axes[0].plot(res['t'], res['t2ec'], '--', color=c, linewidth=1.5,
                         label=f'{label} clover')

        axes[0].axhline(0.3, color='red', linestyle=':', alpha=0.7, label='$t^2\\langle E \\rangle = 0.3$')
        axes[0].set_xlabel('Flow time $t/a^2$')
        axes[0].set_ylabel('$t^2 \\langle E(t) \\rangle$')
        axes[0].set_title('Wilson Flow Energy')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # W(t) for w0
        for i, res in enumerate(flow_results):
            c = colors[i % len(colors)]
            t = res['t']
            t2e = res['t2ec']
            if len(t) < 3:
                continue
            W = np.zeros(len(t))
            for j in range(1, len(t) - 1):
                W[j] = t[j] * (t2e[j+1] - t2e[j-1]) / (t[j+1] - t[j-1])
            label = res['dir'].split('_')[-1] if '_' in res['dir'] else res['dir']
            axes[1].plot(t[1:-1], W[1:-1], '-', color=c, linewidth=1.5, label=label)

        axes[1].axhline(0.3, color='red', linestyle=':', alpha=0.7, label='$W(t) = 0.3$')
        axes[1].set_xlabel('Flow time $t/a^2$')
        axes[1].set_ylabel('$W(t) = t \\cdot d/dt [t^2 \\langle E \\rangle]$')
        axes[1].set_title('$w_0$ Determination')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        outfile = os.path.join(run_dir, 'scale_setting.png')
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f"\n  Plot saved to: {outfile}")
        plt.close()

    print()


if __name__ == '__main__':
    main()
