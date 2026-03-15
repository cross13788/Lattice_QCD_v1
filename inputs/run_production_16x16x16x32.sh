#!/bin/bash
#-----------------------------------------------
# Production batch submission script for 16^3 x 32 runs
#
# Loops over all production input files, copies to
# run/lqcd.inp, executes, and organizes output.
#
# Usage:
#   cd run/
#   bash ../inputs/run_production_16x16x16x32.sh [--dry-run]
#-----------------------------------------------
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$(cd "$SCRIPT_DIR/../run" && pwd)"
EXE="$RUN_DIR/lqcd.exe"

DRY_RUN=0
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "=== DRY RUN MODE ==="
fi

if [[ ! -x "$EXE" ]]; then
    echo "Error: Executable not found at $EXE"
    echo "Please run 'make' in src/ first."
    exit 1
fi

# Production input files in execution order
INPUTS=(
    # Quenched pure gauge
    "quenched_16x16x16x32_b58.inp"
    "quenched_16x16x16x32_b60.inp"
    "quenched_16x16x16x32_b62.inp"
    # Quenched spectroscopy (Wilson)
    "spectro_16x16x16x32_b60_k150.inp"
    "spectro_16x16x16x32_b60_k154.inp"
    # Quenched spectroscopy (Clover)
    "spectro_clover_16x16x16x32_b60_k150_csw1769.inp"
    "spectro_clover_16x16x16x32_b60_k154_csw1769.inp"
    # HMC dynamical fermions
    "hmc_16x16x16x32_b56_k155.inp"
    "hmc_clover_16x16x16x32_b56_k155_csw1.inp"
)

echo "==============================================="
echo "  Production runs: 16^3 x 32 lattice"
echo "  ${#INPUTS[@]} input files queued"
echo "==============================================="
echo ""

TOTAL=${#INPUTS[@]}
COMPLETED=0
FAILED=0

for INP in "${INPUTS[@]}"; do
    INPUT_FILE="$SCRIPT_DIR/$INP"

    if [[ ! -f "$INPUT_FILE" ]]; then
        echo "WARNING: Input file not found: $INP — skipping"
        FAILED=$((FAILED + 1))
        continue
    fi

    echo "-------------------------------------------"
    echo "  [$((COMPLETED+FAILED+1))/$TOTAL] Running: $INP"
    echo "-------------------------------------------"

    # Extract run name from input filename (remove .inp extension)
    RUN_NAME="${INP%.inp}"
    LOG_FILE="$RUN_DIR/${RUN_NAME}.log"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  [DRY RUN] Would copy $INP -> lqcd.inp and run"
        COMPLETED=$((COMPLETED + 1))
        continue
    fi

    # Copy input file
    cp "$INPUT_FILE" "$RUN_DIR/lqcd.inp"

    # Run simulation
    START_TIME=$(date +%s)
    echo "  Started: $(date)"

    cd "$RUN_DIR"
    if "$EXE" > "$LOG_FILE" 2>&1; then
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        echo "  Completed in ${ELAPSED}s"
        COMPLETED=$((COMPLETED + 1))
    else
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        echo "  FAILED after ${ELAPSED}s — see $LOG_FILE"
        FAILED=$((FAILED + 1))
    fi

    echo ""
done

echo "==============================================="
echo "  Summary: $COMPLETED completed, $FAILED failed out of $TOTAL"
echo "==============================================="
