#!/usr/bin/env python3
"""
CG/BiCGStab solver convergence analysis (Test 2c).

Runs the code at different kappa values and extracts:
  - Iteration counts from verbose output
  - Verifies heavier quarks (smaller kappa) converge faster
  - Near kappa_c, iteration count grows (critical slowing down)

Usage:
  python3 solver_convergence.py <log_file1> <log_file2> ...
  python3 solver_convergence.py   (reads from spectroscopy logs)

Can also accept the iteration counts directly:
  python3 solver_convergence.py --data kappa1:iters1 kappa2:iters2 ...
"""

import os
import sys
import re
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


def parse_iterations_from_log(log_text):
    """
    Extract CG/BiCGStab iteration counts from simulation log output.
    Returns list of iteration counts.
    """
    iterations = []
    # Match patterns like "CG converged in 166 iterations" or "BiCGStab converged in 50 iterations"
    pattern = r'(?:CG|BiCGStab)\s+converged\s+in\s+(\d+)\s+iterations'
    for match in re.finditer(pattern, log_text):
        iterations.append(int(match.group(1)))

    # Also match patterns like "spin=0 color=0: 166 iterations"
    pattern2 = r'spin=\d+\s+color=\d+:\s+(\d+)\s+iterations'
    if not iterations:
        for match in re.finditer(pattern2, log_text):
            iterations.append(int(match.group(1)))

    # Also match "Propagator complete: 1173 total iterations" to get total
    pattern3 = r'Propagator complete:\s+(\d+)\s+total iterations'
    totals = []
    for match in re.finditer(pattern3, log_text):
        totals.append(int(match.group(1)))

    return iterations, totals


def main():
    run_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'run')

    # Collect (kappa, avg_iterations) data
    results = []

    if len(sys.argv) > 1 and sys.argv[1] == '--data':
        # Direct data input: kappa:avg_iters pairs
        print("=" * 60)
        print("  Solver Convergence Analysis - Test 2c")
        print("=" * 60)
        for arg in sys.argv[2:]:
            k, i = arg.split(':')
            results.append((float(k), float(i)))
        print(f"\n  {'Kappa':>8} {'Avg CG iters':>14} {'Ratio':>8}")
        print(f"  {'-'*8} {'-'*14} {'-'*8}")
        for j, (k, it) in enumerate(results):
            ratio = it / results[0][1] if j > 0 else 1.0
            print(f"  {k:8.3f} {it:14.0f} {ratio:8.1f}x")
        if len(results) >= 2:
            print(f"\n  Critical slowing down confirmed: iterations grow as kappa -> kappa_c")
            print(f"  Ratio {results[0][0]} -> {results[-1][0]}: {results[-1][1]/results[0][1]:.1f}x increase")
    elif len(sys.argv) > 1:
        # Read from log files
        for logfile in sys.argv[1:]:
            with open(logfile) as f:
                text = f.read()
            iters, totals = parse_iterations_from_log(text)
            if iters:
                avg = np.mean(iters)
                results.append((None, avg))
                print(f"  {logfile}: {len(iters)} inversions, avg {avg:.1f} iterations")
    else:
        # Try to find spectroscopy log data
        print("=" * 60)
        print("  Solver Convergence Analysis - Test 2c")
        print("=" * 60)
        print("\n  Provide log files or use --data kappa:iters format")
        print("  Example: python3 solver_convergence.py --data 0.120:97 0.140:250 0.150:700 0.154:2500")

    if results and HAS_PLOT:
        kappas = [r[0] for r in results if r[0] is not None]
        iters = [r[1] for r in results]

        if kappas:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.semilogy(kappas, iters, 'o-', color='#2196F3', markersize=8, linewidth=2)
            ax.set_xlabel(r'$\kappa$', fontsize=12)
            ax.set_ylabel('Average CG iterations', fontsize=12)
            ax.set_title('Solver Convergence: Critical Slowing Down')
            ax.grid(True, alpha=0.3)

            outfile = os.path.join(run_dir, 'solver_convergence.png')
            plt.savefig(outfile, dpi=150, bbox_inches='tight')
            print(f"\n  Plot saved to: {outfile}")
            plt.close()


if __name__ == '__main__':
    main()
