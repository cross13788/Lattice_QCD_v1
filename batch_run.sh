#!/bin/bash
# batch_run.sh - Run multiple Lattice QCD inputs sequentially on a remote cluster
#
# Usage:
#   ./batch_run.sh <cluster> <input1.inp> [input2.inp ...]
#   ./batch_run.sh <cluster> <directory_of_inputs/>

set -e

REMOTE_USER="${LQCD_USER:-$(whoami)}"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <cluster> <input1.inp> [input2.inp ...]"
    echo "       $0 <cluster> <directory/>"
    exit 1
fi

HOST_NAME="$1"
shift

# Collect input files
INPUTS=()
for arg in "$@"; do
    if [ -d "$arg" ]; then
        for f in "$arg"/*.inp; do
            [ -f "$f" ] && INPUTS+=("$f")
        done
    elif [ -f "$arg" ]; then
        INPUTS+=("$arg")
    else
        echo "Warning: '$arg' not found, skipping"
    fi
done

if [ ${#INPUTS[@]} -eq 0 ]; then
    echo "Error: No input files found"
    exit 1
fi

echo ""
echo "=== Lattice QCD Batch Run ==="
echo "  Cluster: $HOST_NAME"
echo "  Runs:    ${#INPUTS[@]}"
echo ""
for i in "${!INPUTS[@]}"; do
    echo "  $((i+1)). $(basename "${INPUTS[$i]}")"
done
echo ""

# Pre-establish SSH master so remote_run.sh reuses it
CONTROL_PATH="/tmp/ssh-lqcd-${HOST_NAME}"
MASTER_EXISTED=false
if ssh -O check -o ControlPath="$CONTROL_PATH" ${REMOTE_USER}@${HOST_NAME} 2>/dev/null; then
    MASTER_EXISTED=true
else
    echo "Establishing SSH connection to $HOST_NAME..."
    ssh -o ControlMaster=yes -o ControlPath="$CONTROL_PATH" -o ControlPersist=600 -fN ${REMOTE_USER}@${HOST_NAME}
fi

cleanup() {
    if [ "$MASTER_EXISTED" = false ]; then
        ssh -O exit -o ControlPath="$CONTROL_PATH" ${REMOTE_USER}@${HOST_NAME} 2>/dev/null || true
    fi
}
trap cleanup EXIT

PASSED=0
FAILED=0
RESULTS=()

for i in "${!INPUTS[@]}"; do
    input="${INPUTS[$i]}"
    name=$(basename "$input")
    echo "============================================"
    echo "=== Run $((i+1))/${#INPUTS[@]}: $name ==="
    echo "============================================"
    echo ""

    POLL=90

    if ./remote_run.sh "$HOST_NAME" "$input" "$POLL"; then
        PASSED=$((PASSED + 1))
        RESULTS+=("PASS  $name")
    else
        FAILED=$((FAILED + 1))
        RESULTS+=("FAIL  $name")
    fi
    echo ""
done

echo "============================================"
echo "=== Batch Summary ==="
echo "============================================"
echo ""
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""
echo "  Passed: $PASSED  Failed: $FAILED  Total: ${#INPUTS[@]}"
echo ""
