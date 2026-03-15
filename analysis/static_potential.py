#!/usr/bin/env python3
"""
Static potential analysis from Wilson loops.

Reads W(R,T) data, extracts V(R) = -log(W(R,T)/W(R,T+1)) at large T,
fits Cornell potential V(R) = -alpha/R + sigma*R + c, and plots.

Usage:
  python3 static_potential.py [wilson_loop_dir]
"""

import os
import sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jackknife_analysis import jackknife, binned_jackknife, jackknife_func
from autocorrelation_analysis import madras_sokal_tau_int, optimal_bin_size

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


def read_wilson_loops(filename):
    """
    Read wilson_loops.dat file.
    Returns: dict[(R,T)] -> list of W values across configs
    """
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


def extract_potential(wloop_data, T_pair):
    """
    Extract V(R) from Wilson loops at time extent T_pair = (T, T+1).
    V(R) = -log(<W(R,T+1)> / <W(R,T)>)

    Returns: list of (R, V, V_err)
    """
    T1, T2 = T_pair
    results = []

    R_max = max(k[0] for k in wloop_data.keys())

    for R in range(1, R_max + 1):
        key1 = (R, T1)
        key2 = (R, T2)
        if key1 not in wloop_data or key2 not in wloop_data:
            continue

        w1_data = np.array(wloop_data[key1])
        w2_data = np.array(wloop_data[key2])

        if len(w1_data) != len(w2_data):
            continue

        # Jackknife the ratio
        n = len(w1_data)
        w1_mean = np.mean(w1_data)
        w2_mean = np.mean(w2_data)

        if w1_mean <= 0 or w2_mean <= 0:
            continue

        V = -np.log(w2_mean / w1_mean)

        # Jackknife error
        V_jack = np.zeros(n)
        for i in range(n):
            w1_j = (np.sum(w1_data) - w1_data[i]) / (n - 1)
            w2_j = (np.sum(w2_data) - w2_data[i]) / (n - 1)
            if w1_j > 0 and w2_j > 0:
                V_jack[i] = -np.log(w2_j / w1_j)
            else:
                V_jack[i] = V

        # Use binned jackknife if sufficient data
        if n > 10:
            tau, _, _ = madras_sokal_tau_int(V_jack)
            bin_sz = optimal_bin_size(tau)
            _, V_err = binned_jackknife(V_jack, bin_sz)
        else:
            V_err = np.sqrt((n - 1) / n * np.sum((V_jack - V)**2))
        results.append((R, V, V_err))

    return results


def fit_cornell(R_data, V_data, V_err=None):
    """
    Fit Cornell potential: V(R) = -alpha/R + sigma*R + c

    Returns: (alpha, sigma, c, chi2_dof)
    """
    R = np.array(R_data, dtype=float)
    V = np.array(V_data, dtype=float)

    if len(R) < 3:
        # Not enough points for 3-parameter fit, do linear only
        A = np.vstack([R, np.ones(len(R))]).T
        params = np.linalg.lstsq(A, V, rcond=None)[0]
        return 0.0, params[0], params[1], 0.0

    # Design matrix: V = alpha*(-1/R) + sigma*R + c
    A = np.vstack([-1.0/R, R, np.ones(len(R))]).T

    if V_err is not None and np.all(np.array(V_err) > 0):
        W = np.diag(1.0 / np.array(V_err)**2)
        # Weighted least squares
        AW = W @ A
        VW = W @ V
        params = np.linalg.lstsq(AW, VW, rcond=None)[0]
    else:
        params = np.linalg.lstsq(A, V, rcond=None)[0]

    alpha, sigma, c = params

    # Chi-squared
    V_fit = A @ params
    residuals = V - V_fit
    if V_err is not None and np.all(np.array(V_err) > 0):
        chi2 = np.sum((residuals / np.array(V_err))**2)
    else:
        chi2 = np.sum(residuals**2)
    dof = max(len(R) - 3, 1)

    return alpha, sigma, c, chi2/dof


def main():
    import glob

    run_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run')

    if len(sys.argv) > 1:
        dirs = [sys.argv[1]]
    else:
        dirs = sorted(glob.glob(os.path.join(run_dir, 'SU3_heatbath_beta*')))
        dirs = [d for d in dirs if os.path.exists(os.path.join(d, 'wilson_loops.dat'))]

    if not dirs:
        print("No Wilson loop data found.")
        sys.exit(1)

    print("=" * 60)
    print("  Static Potential Analysis")
    print("=" * 60)

    for d in dirs:
        basename = os.path.basename(d)
        wloop_file = os.path.join(d, 'wilson_loops.dat')

        print(f"\n  Directory: {basename}")

        wloop_data = read_wilson_loops(wloop_file)
        n_configs = len(set(len(v) for v in wloop_data.values()))

        R_max = max(k[0] for k in wloop_data.keys())
        T_max = max(k[1] for k in wloop_data.keys())
        print(f"    R_max = {R_max}, T_max = {T_max}")
        print(f"    N_configs ~ {max(len(v) for v in wloop_data.values())}")

        # Wilson loop averages table
        print(f"\n    Wilson Loop Averages:")
        print(f"    {'R':>3} {'T':>3} {'<W(R,T)>':>14} {'Error':>12}")
        for R in range(1, R_max + 1):
            for T in range(1, T_max + 1):
                key = (R, T)
                if key in wloop_data:
                    m, e = jackknife(np.array(wloop_data[key]))
                    print(f"    {R:3d} {T:3d} {m:14.8e} {e:12.4e}")

        # Extract V(R) at different T separations
        print(f"\n    Static Potential V(R):")
        T_pairs = [(T, T+1) for T in range(2, T_max)]

        best_T_pair = None
        for T_pair in T_pairs:
            results = extract_potential(wloop_data, T_pair)
            if len(results) >= 2:
                print(f"\n    T={T_pair[0]}/{T_pair[1]}:")
                for R, V, V_err in results:
                    print(f"      R={R}: V = {V:.6f} +/- {V_err:.6f}")
                best_T_pair = T_pair

        # Fit Cornell potential using best (largest T) extraction
        if best_T_pair:
            results = extract_potential(wloop_data, best_T_pair)
            Rs = [r[0] for r in results]
            Vs = [r[1] for r in results]
            Verrs = [r[2] for r in results]

            alpha, sigma, c, chi2 = fit_cornell(Rs, Vs, Verrs)
            print(f"\n    Cornell fit (T={best_T_pair[0]}/{best_T_pair[1]}):")
            print(f"      V(R) = -{alpha:.4f}/R + {sigma:.4f}*R + {c:.4f}")
            print(f"      alpha (Coulomb) = {alpha:.4f}")
            print(f"      sigma (string)  = {sigma:.4f}")
            print(f"      chi2/dof        = {chi2:.4f}")

            # Physical string tension at beta=6.0
            beta_val = None
            for part in basename.split('_'):
                if part.startswith('beta'):
                    beta_val = float(part[4:])

            if beta_val and abs(beta_val - 6.0) < 0.01:
                a_fm = 0.093  # approximate lattice spacing
                sigma_phys = sigma / a_fm**2
                sqrt_sigma_MeV = np.sqrt(sigma_phys) * 197.3
                print(f"      At a ~ {a_fm} fm: sqrt(sigma) ~ {sqrt_sigma_MeV:.0f} MeV (expected ~440 MeV)")

            # Plot
            if HAS_PLOT:
                fig, ax = plt.subplots(figsize=(8, 6))

                # Plot V(R) at different T
                colors = ['#90CAF9', '#42A5F5', '#1E88E5', '#1565C0']
                for ti, T_pair_i in enumerate(T_pairs):
                    res = extract_potential(wloop_data, T_pair_i)
                    if res:
                        rr = [r[0] for r in res]
                        vv = [r[1] for r in res]
                        ve = [r[2] for r in res]
                        ci = colors[min(ti, len(colors)-1)]
                        ax.errorbar(rr, vv, yerr=ve, fmt='o-', color=ci,
                                    markersize=6, capsize=3,
                                    label=f'$T={T_pair_i[0]}/{T_pair_i[1]}$')

                # Cornell fit curve
                R_fine = np.linspace(0.8, R_max + 0.5, 100)
                V_fine = -alpha/R_fine + sigma*R_fine + c
                ax.plot(R_fine, V_fine, '--', color='gray', linewidth=1.5,
                        label=f'Cornell: $\\alpha={alpha:.3f}, \\sigma={sigma:.3f}$')

                ax.set_xlabel('$R/a$', fontsize=12)
                ax.set_ylabel('$aV(R)$', fontsize=12)
                ax.set_title(f'Static Quark-Antiquark Potential ({basename})')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

                outfile = os.path.join(run_dir, 'static_potential.png')
                plt.savefig(outfile, dpi=150, bbox_inches='tight')
                print(f"\n    Plot saved to: {outfile}")
                plt.close()

    print()


if __name__ == '__main__':
    main()
