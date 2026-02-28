#!/bin/bash
# ============================================================
#  Lattice QCD - Cluster Test Suite
#  Runs remaining validation tests (3e, 2b-kappa0.154)
#
#  Usage:
#    cd Lattice_QCD/run
#    bash ../inputs/run_cluster_tests.sh
#
#  Estimated time: ~30-60 min on a single node
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$(cd "$SCRIPT_DIR/../run" && pwd)"
cd "$RUN_DIR"

ulimit -s unlimited
export OMP_STACKSIZE=4500m
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Detect thread count
if [ -z "$OMP_NUM_THREADS" ]; then
    NCORES=$(nproc 2>/dev/null || echo 4)
    export OMP_NUM_THREADS=$NCORES
fi

echo "============================================================"
echo "  Lattice QCD Cluster Test Suite"
echo "  Date: $(date)"
echo "  Host: $(hostname)"
echo "  OMP threads: $OMP_NUM_THREADS"
echo "============================================================"

LOGFILE="cluster_test_log_$(date +%Y%m%d_%H%M%S).txt"
echo "Logging to: $LOGFILE"
exec > >(tee "$LOGFILE") 2>&1

# ----------------------------------------------------------
# Test 3e: HMC Thermalization - Hot vs Cold Start
# ----------------------------------------------------------
echo ""
echo "============================================="
echo "  Test 3e: HMC Thermalization (HOT start)"
echo "============================================="

cp "$SCRIPT_DIR/hmc_therm_hot_8x8x8x8_b56.inp" lqcd.inp
time ./lqcd.exe 2>&1 | tee hmc_therm_hot.log

# Save output
HOT_DIR="HMC_therm_hot_beta5.60_8x8x8x8"
if [ -d "SU3_hmc_beta5.60_8x8x8x8" ]; then
    mkdir -p "$HOT_DIR"
    cp SU3_hmc_beta5.60_8x8x8x8/plaquette.dat "$HOT_DIR/"
    cp SU3_hmc_beta5.60_8x8x8x8/polyakov.dat "$HOT_DIR/" 2>/dev/null || true
    echo "Saved hot start data to $HOT_DIR"
fi

echo ""
echo "============================================="
echo "  Test 3e: HMC Thermalization (COLD start)"
echo "============================================="

cp "$SCRIPT_DIR/hmc_therm_cold_8x8x8x8_b56.inp" lqcd.inp
time ./lqcd.exe 2>&1 | tee hmc_therm_cold.log

# Save output
COLD_DIR="HMC_therm_cold_beta5.60_8x8x8x8"
if [ -d "SU3_hmc_beta5.60_8x8x8x8" ]; then
    mkdir -p "$COLD_DIR"
    cp SU3_hmc_beta5.60_8x8x8x8/plaquette.dat "$COLD_DIR/"
    cp SU3_hmc_beta5.60_8x8x8x8/polyakov.dat "$COLD_DIR/" 2>/dev/null || true
    echo "Saved cold start data to $COLD_DIR"
fi

# Quick check: both should converge to similar plaquette
echo ""
echo "--- Test 3e Quick Check ---"
if [ -f "$HOT_DIR/plaquette.dat" ] && [ -f "$COLD_DIR/plaquette.dat" ]; then
    HOT_AVG=$(awk 'NR>1 && NR>101{sum+=$2; n++} END{if(n>0) printf "%.6f", sum/n; else print "N/A"}' "$HOT_DIR/plaquette.dat")
    COLD_AVG=$(awk 'NR>1 && NR>101{sum+=$2; n++} END{if(n>0) printf "%.6f", sum/n; else print "N/A"}' "$COLD_DIR/plaquette.dat")
    echo "  Hot start  <P> (last half): $HOT_AVG"
    echo "  Cold start <P> (last half): $COLD_AVG"
    echo "  (Should converge to same value)"
fi

# ----------------------------------------------------------
# Test 2b continued: Spectroscopy at kappa=0.154
# ----------------------------------------------------------
echo ""
echo "============================================="
echo "  Test 2b: Spectroscopy kappa=0.154"
echo "  (Near kappa_c - expect slow CG convergence)"
echo "============================================="

OUTDIR="SU3_heatbath_beta6.00_8x8x8x16"
cp "$SCRIPT_DIR/spectro_8x8x8x16_b60_k154.inp" lqcd.inp
time ./lqcd.exe 2>&1 | tee spectro_k154.log | grep -E "(Pion:|Config|Propagator complete|converged)"

# Save correlator data
SAVE_DIR="spectro_k0.154"
mkdir -p "$SAVE_DIR"
if [ -d "$OUTDIR" ]; then
    cp ${OUTDIR}/correlator_*.dat "$SAVE_DIR/" 2>/dev/null || true
    echo "Saved kappa=0.154 data to $SAVE_DIR"
fi

# Update kappa map
MAPFILE="spectroscopy_kappa_map.txt"
if [ -f "$MAPFILE" ]; then
    grep -q "0.154" "$MAPFILE" || echo "0.154  $SAVE_DIR" >> "$MAPFILE"
else
    echo "# kappa  directory" > "$MAPFILE"
    echo "0.154  $SAVE_DIR" >> "$MAPFILE"
fi

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
echo ""
echo "============================================================"
echo "  Cluster Test Suite Complete"
echo "  Date: $(date)"
echo "============================================================"
echo ""
echo "  Results:"
echo "    Test 3e logs: hmc_therm_hot.log, hmc_therm_cold.log"
echo "    Test 3e data: $HOT_DIR/, $COLD_DIR/"
echo "    Spectro k=0.154: $SAVE_DIR/"
echo "    Full log: $LOGFILE"
echo ""
echo "  Next steps:"
echo "    1. Run analysis: python3 ../analysis/spectroscopy.py"
echo "    2. Check 3e convergence: compare plaquette in hot/cold dirs"
echo "    3. Update viz: python3 ../analysis/viz_thermalization.py"
echo "============================================================"
