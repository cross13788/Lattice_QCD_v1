#!/bin/bash
# deploy.sh - Sync Lattice_QCD files to a remote cluster (no compile or run)
#
# Usage:
#   ./deploy.sh <cluster>
#
# Syncs source code, scripts, input files, and analysis scripts.
# Does NOT compile or execute.
# Use remote_run.sh for the full deploy+compile+execute+fetch workflow.

set -e

REMOTE_USER="${LQCD_USER:-$(whoami)}"
BASE_PATH="${LQCD_REMOTE_PATH:-/home/${REMOTE_USER}/Lattice_QCD}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <cluster>"
    echo "  cluster:  remote hostname (e.g., nuclps1, nuclps3, comphys)"
    exit 1
fi

HOST_NAME="$1"

echo ""
echo "=== Lattice QCD Deploy ==="
echo "  Target:  ${HOST_NAME}"
echo "  Path:    ${BASE_PATH}/"
echo ""


ssh ${REMOTE_USER}@${HOST_NAME} \
    "mkdir -p ${BASE_PATH}/src/gpu ${BASE_PATH}/run ${BASE_PATH}/inputs ${BASE_PATH}/analysis"

echo "Syncing src/ (C++ source files, CUDA files, and Makefile)..."
rsync -avz --checksum \
    --include='*.cpp' \
    --include='*.hpp' \
    --include='*.cu' \
    --include='*.cuh' \
    --include='Makefile' \
    --include='gpu/' \
    --include='gpu/**' \
    --exclude='*.o' \
    --exclude='.last_target' \
    --exclude='*' \
    src/ \
    ${REMOTE_USER}@${HOST_NAME}:${BASE_PATH}/src/

echo "Syncing inputs/ (input files and run scripts)..."
rsync -avz --checksum \
    --include='*.inp' \
    --include='*.sh' \
    --exclude='*' \
    inputs/ \
    ${REMOTE_USER}@${HOST_NAME}:${BASE_PATH}/inputs/

echo "Syncing analysis/ (Python analysis scripts)..."
rsync -avz --checksum \
    --include='*.py' \
    --exclude='__pycache__/' \
    --exclude='*' \
    analysis/ \
    ${REMOTE_USER}@${HOST_NAME}:${BASE_PATH}/analysis/

echo "Syncing shell scripts..."
rsync -avz --checksum \
    --include='*.sh' \
    --exclude='*' \
    ./ \
    ${REMOTE_USER}@${HOST_NAME}:${BASE_PATH}/

echo "Syncing run/ (input file only, no data)..."
rsync -avz --checksum \
    --exclude='SU3_*/' \
    --exclude='spectro_*/' \
    --exclude='HMC_*/' \
    --exclude='run_record.dat' \
    --exclude='lqcd.exe' \
    --exclude='lqcd_gpu.exe' \
    --exclude='remote_run.log' \
    --exclude='*.png' \
    --exclude='*.gif' \
    --exclude='*.log' \
    --exclude='*.dat' \
    run/ \
    ${REMOTE_USER}@${HOST_NAME}:${BASE_PATH}/run/


echo ""
echo "Deploy complete. Files synced to ${HOST_NAME}:${BASE_PATH}/"
echo ""
echo "Next steps:"
echo "  Compile:       ssh ${HOST_NAME} 'cd ${BASE_PATH} && bash build.sh'"
echo "  Run tests:     ssh ${HOST_NAME} 'cd ${BASE_PATH}/run && bash ../inputs/run_cluster_tests.sh'"
echo "  Or full cycle: ./remote_run.sh ${HOST_NAME} inputs/hmc_therm_hot_8x8x8x8_b56.inp"
echo ""
