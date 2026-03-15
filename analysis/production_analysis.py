#!/usr/bin/env python3
"""
Master analysis pipeline for Lattice QCD production runs.

Processes output from 16^3x32 (and smaller) runs:
  - Wilson flow scale setting (t0, w0) across multiple beta
  - Plaquette comparison with literature
  - Spectroscopy: Wilson vs clover comparison
  - Static potential and Sommer r0
  - HMC diagnostics (acceptance, deltaH)

Usage:
  python3 production_analysis.py [run_dir]
  python3 production_analysis.py   (auto-detect from ../run/)
"""

import os
import sys
import re
import numpy as np
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jackknife_analysis import jackknife, binned_jackknife
from autocorrelation_analysis import madras_sokal_tau_int, optimal_bin_size

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


# ===================================================================
#  Utility: parse directory names
# ===================================================================

def parse_dir_name(dirname):
    """Extract run parameters from directory name like SU3_heatbath_beta6.00_8x8x8x16."""
    info = {}
    basename = os.path.basename(dirname)
    parts = basename.split('_')
    for p in parts:
        if p.startswith('beta'):
            try:
                info['beta'] = float(p[4:])
            except ValueError:
                pass
    # Lattice size
    m = re.search(r'(\d+)x(\d+)x(\d+)x(\d+)', basename)
    if m:
        info['nX'] = int(m.group(1))
        info['nY'] = int(m.group(2))
        info['nZ'] = int(m.group(3))
        info['nT'] = int(m.group(4))
        info['vol'] = info['nX'] * info['nY'] * info['nZ'] * info['nT']
    # Method
    if 'heatbath' in basename:
        info['method'] = 'heatbath'
    elif 'hmc' in basename:
        info['method'] = 'hmc'
    elif 'metropolis' in basename:
        info['method'] = 'metropolis'
    info['dir'] = dirname
    info['name'] = basename
    return info


# ===================================================================
#  Section 1: Wilson Flow Scale Setting
# ===================================================================

def analyze_wilson_flow(dirs, run_dir):
    """Analyze Wilson flow data across multiple beta values."""
    print("\n" + "=" * 65)
    print("  1. Wilson Flow Scale Setting")
    print("=" * 65)

    flow_results = []
    for d in dirs:
        flow_file = os.path.join(d, 'wilson_flow.dat')
        if not os.path.exists(flow_file):
            continue

        info = parse_dir_name(d)
        t, t2ep, t2ec, ep, ec = [], [], [], [], []
        with open(flow_file) as f:
            for line in f:
                if line.startswith('#') or line.strip() == '':
                    continue
                parts = line.split()
                t.append(float(parts[0]))
                t2ep.append(float(parts[1]))
                t2ec.append(float(parts[2]))
                ep.append(float(parts[3]))
                ec.append(float(parts[4]))
        t = np.array(t)
        t2ep = np.array(t2ep)
        t2ec = np.array(t2ec)

        # Extract t0 (t^2 E = 0.3)
        t0 = None
        for i in range(1, len(t)):
            if t2ec[i-1] < 0.3 <= t2ec[i]:
                frac = (0.3 - t2ec[i-1]) / (t2ec[i] - t2ec[i-1])
                t0 = t[i-1] + frac * (t[i] - t[i-1])
                break

        # Extract w0
        w0 = None
        if len(t) >= 3:
            W = np.zeros(len(t))
            for i in range(1, len(t) - 1):
                W[i] = t[i] * (t2ec[i+1] - t2ec[i-1]) / (t[i+1] - t[i-1])
            for i in range(1, len(t) - 1):
                if W[i-1] < 0.3 <= W[i]:
                    frac = (0.3 - W[i-1]) / (W[i] - W[i-1])
                    tc = t[i-1] + frac * (t[i] - t[i-1])
                    w0 = np.sqrt(tc)
                    break

        info['t0'] = t0
        info['w0'] = w0
        info['flow_t'] = t
        info['flow_t2ec'] = t2ec
        info['flow_t2ep'] = t2ep
        flow_results.append(info)

    if not flow_results:
        print("  No Wilson flow data found.")
        return flow_results

    print(f"\n  {'Directory':<45} {'beta':>6} {'t0/a^2':>10} {'w0/a':>10}")
    print(f"  {'-'*45} {'-'*6} {'-'*10} {'-'*10}")
    for r in sorted(flow_results, key=lambda x: x.get('beta', 0)):
        t0_str = f"{r['t0']:.4f}" if r['t0'] else "---"
        w0_str = f"{r['w0']:.4f}" if r['w0'] else "---"
        print(f"  {r['name']:<45} {r.get('beta', 0):6.2f} {t0_str:>10} {w0_str:>10}")

    return flow_results


# ===================================================================
#  Section 2: Plaquette Summary
# ===================================================================

def analyze_plaquettes(dirs, run_dir):
    """Analyze plaquette data with proper autocorrelation treatment."""
    print("\n" + "=" * 65)
    print("  2. Plaquette Summary")
    print("=" * 65)

    # Known literature values (quenched SU(3))
    literature = {5.6: 0.5512, 5.8: 0.5736, 6.0: 0.5935, 6.2: 0.6101}

    plaq_results = []
    for d in dirs:
        plaq_file = os.path.join(d, 'plaquette.dat')
        if not os.path.exists(plaq_file):
            continue

        info = parse_dir_name(d)
        plaq_vals = []
        with open(plaq_file) as f:
            for line in f:
                if line.startswith('#') or line.strip() == '':
                    continue
                parts = line.split()
                plaq_vals.append(float(parts[1]))

        if len(plaq_vals) < 2:
            continue

        data = np.array(plaq_vals)
        tau, _, _ = madras_sokal_tau_int(data)
        bin_sz = optimal_bin_size(tau)
        mean, err = binned_jackknife(data, bin_sz)

        info['plaq_mean'] = mean
        info['plaq_err'] = err
        info['plaq_n'] = len(data)
        info['tau_int'] = tau
        info['bin_size'] = bin_sz

        beta = info.get('beta', 0)
        lit_val = literature.get(round(beta, 1), None)
        info['plaq_lit'] = lit_val

        plaq_results.append(info)

    if not plaq_results:
        print("  No plaquette data found.")
        return plaq_results

    print(f"\n  {'Directory':<40} {'beta':>5} {'<P>':>12} {'err':>10} {'N':>5} {'tau':>5} {'lit':>10}")
    print(f"  {'-'*40} {'-'*5} {'-'*12} {'-'*10} {'-'*5} {'-'*5} {'-'*10}")
    for r in sorted(plaq_results, key=lambda x: x.get('beta', 0)):
        lit = f"{r['plaq_lit']:.4f}" if r['plaq_lit'] else "---"
        print(f"  {r['name']:<40} {r.get('beta', 0):5.2f} {r['plaq_mean']:12.8f} "
              f"{r['plaq_err']:10.2e} {r['plaq_n']:5d} {r['tau_int']:5.1f} {lit:>10}")

    return plaq_results


# ===================================================================
#  Section 3: Spectroscopy (Wilson vs Clover)
# ===================================================================

def analyze_spectroscopy(dirs, run_dir):
    """Extract meson masses and compare Wilson vs clover."""
    print("\n" + "=" * 65)
    print("  3. Meson Spectroscopy")
    print("=" * 65)

    try:
        from spectroscopy import get_pion_mass, get_rho_mass
    except ImportError:
        print("  Cannot import spectroscopy module.")
        return []

    spectro_results = []
    for d in dirs:
        pion_file = os.path.join(d, 'correlator_pion.dat')
        if not os.path.exists(pion_file):
            continue

        info = parse_dir_name(d)
        try:
            pion = get_pion_mass(d)
            rho = get_rho_mass(d)
        except Exception:
            continue

        if pion is None:
            continue

        info['m_pi'] = pion['mass']
        info['m_pi_err'] = pion['error']
        info['n_configs'] = pion['n_configs']
        info['m_rho'] = rho['mass'] if rho else None
        spectro_results.append(info)

    if not spectro_results:
        print("  No spectroscopy data found.")
        return spectro_results

    print(f"\n  {'Directory':<45} {'m_pi':>10} {'err':>10} {'m_rho':>10} {'N_cfg':>6}")
    print(f"  {'-'*45} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
    for r in sorted(spectro_results, key=lambda x: x['name']):
        rho_str = f"{r['m_rho']:.6f}" if r['m_rho'] else "---"
        print(f"  {r['name']:<45} {r['m_pi']:10.6f} {r['m_pi_err']:10.6f} "
              f"{rho_str:>10} {r['n_configs']:6d}")

    return spectro_results


# ===================================================================
#  Section 4: Static Potential
# ===================================================================

def analyze_static_potential(dirs, run_dir):
    """Extract static potential and Sommer r0."""
    print("\n" + "=" * 65)
    print("  4. Static Potential and Sommer r0")
    print("=" * 65)

    try:
        from static_potential import read_wilson_loops, extract_potential, fit_cornell
    except ImportError:
        print("  Cannot import static_potential module.")
        return []

    pot_results = []
    for d in dirs:
        wloop_file = os.path.join(d, 'wilson_loops.dat')
        if not os.path.exists(wloop_file):
            continue

        info = parse_dir_name(d)
        wloop_data = read_wilson_loops(wloop_file)
        if not wloop_data:
            continue

        T_max = max(k[1] for k in wloop_data.keys())

        # Find best T pair
        best = None
        for T1 in range(max(2, T_max - 2), 1, -1):
            res = extract_potential(wloop_data, (T1, T1 + 1))
            if len(res) >= 3:
                best = (T1, T1 + 1)
                break

        if best is None:
            continue

        results = extract_potential(wloop_data, best)
        Rs = [r[0] for r in results]
        Vs = [r[1] for r in results]
        Verrs = [r[2] for r in results]
        alpha, sigma, c, chi2 = fit_cornell(Rs, Vs, Verrs)

        # Sommer r0
        r0 = None
        if sigma > 0:
            val = (1.65 - alpha) / sigma
            if val > 0:
                r0 = np.sqrt(val)

        info['alpha'] = alpha
        info['sigma'] = sigma
        info['r0'] = r0
        info['chi2'] = chi2
        info['Rs'] = Rs
        info['Vs'] = Vs
        info['Verrs'] = Verrs
        pot_results.append(info)

    if not pot_results:
        print("  No Wilson loop data found.")
        return pot_results

    print(f"\n  {'Directory':<40} {'beta':>5} {'alpha':>8} {'sigma':>8} {'r0/a':>8} {'chi2':>8}")
    print(f"  {'-'*40} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in sorted(pot_results, key=lambda x: x.get('beta', 0)):
        r0_str = f"{r['r0']:.4f}" if r['r0'] else "---"
        print(f"  {r['name']:<40} {r.get('beta', 0):5.2f} {r['alpha']:8.4f} "
              f"{r['sigma']:8.4f} {r0_str:>8} {r['chi2']:8.4f}")

    return pot_results


# ===================================================================
#  Section 5: HMC Diagnostics
# ===================================================================

def analyze_hmc(dirs, run_dir):
    """Analyze HMC acceptance rates and energy conservation."""
    print("\n" + "=" * 65)
    print("  5. HMC Diagnostics")
    print("=" * 65)

    hmc_results = []
    for d in dirs:
        info = parse_dir_name(d)
        if info.get('method') != 'hmc':
            continue

        plaq_file = os.path.join(d, 'plaquette.dat')
        if not os.path.exists(plaq_file):
            continue

        # HMC plaquette.dat has columns: traj  plaq  action_density  deltaH  accepted
        deltaH_vals = []
        accepted_vals = []
        plaq_vals = []
        with open(plaq_file) as f:
            for line in f:
                if line.startswith('#') or line.strip() == '':
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    plaq_vals.append(float(parts[1]))
                    deltaH_vals.append(float(parts[3]))
                    accepted_vals.append(int(parts[4]))
                elif len(parts) >= 2:
                    plaq_vals.append(float(parts[1]))

        if not deltaH_vals:
            continue

        dH = np.array(deltaH_vals)
        acc = np.array(accepted_vals)

        info['n_traj'] = len(dH)
        info['acc_rate'] = 100.0 * np.mean(acc)
        info['mean_abs_dH'] = np.mean(np.abs(dH))
        info['mean_dH2'] = np.mean(dH**2)
        info['plaq_mean'] = np.mean(plaq_vals) if plaq_vals else 0
        hmc_results.append(info)

    if not hmc_results:
        print("  No HMC data found.")
        return hmc_results

    print(f"\n  {'Directory':<40} {'N_traj':>7} {'Acc%':>7} {'<|dH|>':>12} {'<dH^2>':>12} {'<P>':>10}")
    print(f"  {'-'*40} {'-'*7} {'-'*7} {'-'*12} {'-'*12} {'-'*10}")
    for r in sorted(hmc_results, key=lambda x: x['name']):
        print(f"  {r['name']:<40} {r['n_traj']:7d} {r['acc_rate']:6.1f}% "
              f"{r['mean_abs_dH']:12.4e} {r['mean_dH2']:12.4e} {r['plaq_mean']:10.6f}")

    return hmc_results


# ===================================================================
#  Section 6: Publication Figures
# ===================================================================

def make_figures(flow_results, plaq_results, spectro_results, pot_results, run_dir):
    """Generate multi-panel summary figure."""
    if not HAS_PLOT:
        print("\n  matplotlib not available — skipping figures.")
        return

    n_panels = sum([
        len(flow_results) > 0,
        len(plaq_results) > 0,
        len(spectro_results) > 0,
        len(pot_results) > 0,
    ])
    if n_panels == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = ['#1565C0', '#2E7D32', '#E65100', '#6A1B9A', '#C62828', '#00838F']
    panel = 0

    # Panel 1: Wilson flow
    if flow_results:
        ax = axes[panel]
        for i, r in enumerate(sorted(flow_results, key=lambda x: x.get('beta', 0))):
            c = colors[i % len(colors)]
            label = f"$\\beta={r.get('beta', '?')}$"
            ax.plot(r['flow_t'], r['flow_t2ec'], '-', color=c, linewidth=1.5, label=label)
        ax.axhline(0.3, color='red', linestyle=':', alpha=0.7, label='$t^2\\langle E\\rangle = 0.3$')
        ax.set_xlabel('Flow time $t/a^2$')
        ax.set_ylabel('$t^2 \\langle E(t) \\rangle$')
        ax.set_title('Wilson Flow (Clover)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        panel += 1

    # Panel 2: Plaquette vs beta
    if plaq_results:
        ax = axes[panel]
        betas = [r.get('beta', 0) for r in plaq_results]
        plaqs = [r['plaq_mean'] for r in plaq_results]
        errs = [r['plaq_err'] for r in plaq_results]
        ax.errorbar(betas, plaqs, yerr=errs, fmt='o', color='#1565C0',
                    markersize=7, capsize=4, label='This work')

        # Literature
        lit_b = sorted(set(round(b, 1) for b in betas))
        literature = {5.6: 0.5512, 5.8: 0.5736, 6.0: 0.5935, 6.2: 0.6101}
        lit_x = [b for b in lit_b if b in literature]
        lit_y = [literature[b] for b in lit_x]
        if lit_x:
            ax.plot(lit_x, lit_y, 's', color='#F44336', markersize=8, label='Literature')

        ax.set_xlabel('$\\beta$')
        ax.set_ylabel('$\\langle P \\rangle$')
        ax.set_title('Average Plaquette')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        panel += 1

    # Panel 3: Meson masses
    if spectro_results:
        ax = axes[panel]
        for i, r in enumerate(sorted(spectro_results, key=lambda x: x['m_pi'])):
            c = colors[i % len(colors)]
            label = r['name'].split('_')[-1] if '_' in r['name'] else r['name']
            ax.errorbar(i, r['m_pi'], yerr=r['m_pi_err'], fmt='o', color='#1565C0',
                        markersize=7, capsize=4)
            if r['m_rho']:
                ax.plot(i, r['m_rho'], 's', color='#F44336', markersize=7)
            ax.annotate(label[:20], (i, r['m_pi']), textcoords="offset points",
                       xytext=(5, 5), fontsize=6, rotation=30)

        ax.set_ylabel('$m \\cdot a$')
        ax.set_title('Meson Masses')
        # Custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#1565C0', markersize=8, label='$m_\\pi$'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#F44336', markersize=8, label='$m_\\rho$'),
        ]
        ax.legend(handles=legend_elements, fontsize=9)
        ax.grid(True, alpha=0.3)
        panel += 1

    # Panel 4: Static potential
    if pot_results:
        ax = axes[panel]
        for i, r in enumerate(sorted(pot_results, key=lambda x: x.get('beta', 0))):
            c = colors[i % len(colors)]
            label = f"$\\beta={r.get('beta', '?')}$"
            ax.errorbar(r['Rs'], r['Vs'], yerr=r['Verrs'], fmt='o-', color=c,
                        markersize=5, capsize=3, label=label)
            # Cornell fit curve
            R_fine = np.linspace(0.8, max(r['Rs']) + 0.5, 100)
            V_fine = -r['alpha']/R_fine + r['sigma']*R_fine + (r['Vs'][0] + r['alpha']/r['Rs'][0] - r['sigma']*r['Rs'][0])
            ax.plot(R_fine, V_fine, '--', color=c, alpha=0.5, linewidth=1)

        ax.set_xlabel('$R/a$')
        ax.set_ylabel('$aV(R)$')
        ax.set_title('Static Potential')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        panel += 1

    # Hide unused panels
    for p in range(panel, 4):
        axes[p].set_visible(False)

    plt.tight_layout()
    outfile = os.path.join(run_dir, 'production_summary.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"\n  Summary figure saved to: {outfile}")
    plt.close()


# ===================================================================
#  Main
# ===================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        run_dir = os.path.join(os.path.dirname(script_dir), 'run')

    print("=" * 65)
    print("  Lattice QCD Production Analysis Pipeline")
    print("=" * 65)
    print(f"  Run directory: {run_dir}")

    # Find all output directories
    dirs = sorted(glob.glob(os.path.join(run_dir, 'SU3_*')))
    dirs = [d for d in dirs if os.path.isdir(d)]
    print(f"  Found {len(dirs)} output directories")

    if not dirs:
        print("  No data directories found.")
        sys.exit(1)

    # Run all analyses
    flow_results = analyze_wilson_flow(dirs, run_dir)
    plaq_results = analyze_plaquettes(dirs, run_dir)
    spectro_results = analyze_spectroscopy(dirs, run_dir)
    pot_results = analyze_static_potential(dirs, run_dir)
    hmc_results = analyze_hmc(dirs, run_dir)

    # Summary
    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)
    print(f"  Wilson flow:    {len(flow_results)} datasets")
    print(f"  Plaquettes:     {len(plaq_results)} datasets")
    print(f"  Spectroscopy:   {len(spectro_results)} datasets")
    print(f"  Static potential: {len(pot_results)} datasets")
    print(f"  HMC runs:       {len(hmc_results)} datasets")

    # Generate figures
    make_figures(flow_results, plaq_results, spectro_results, pot_results, run_dir)

    print()


if __name__ == '__main__':
    main()
