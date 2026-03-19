#!/bin/bash
#-----------------------------------------------
# Run script for Lattice QCD
#
# Usage:
#   ./run.sh [cpu|gpu]
#
# Default: cpu
#-----------------------------------------------
set -e

TARGET="${1:-cpu}"

cd run

# Select executable
if [ "$TARGET" = "gpu" ]; then
    EXEC="./lqcd_gpu.exe"
    if [ ! -f "$EXEC" ]; then
        echo "Error: lqcd_gpu.exe not found in run/. Run ./build.sh gpu first."
        exit 1
    fi
else
    EXEC="./lqcd.exe"
    if [ ! -f "$EXEC" ]; then
        echo "Error: lqcd.exe not found in run/. Run ./build.sh first."
        exit 1
    fi
fi

echo ""
echo "Executing the code ($TARGET)..."

# Parse input file for run_record
beta=$(grep "^Beta:" lqcd.inp | awk -F': ' '{print $2}' | xargs)
nX=$(grep "^nX:" lqcd.inp | awk -F': ' '{print $2}' | xargs)
nY=$(grep "^nY:" lqcd.inp | awk -F': ' '{print $2}' | xargs)
nZ=$(grep "^nZ:" lqcd.inp | awk -F': ' '{print $2}' | xargs)
nT=$(grep "^nT:" lqcd.inp | awk -F': ' '{print $2}' | xargs)
lattice="${nX}x${nY}x${nZ}x${nT}"
update_method=$(grep "^Update Method:" lqcd.inp | awk -F': ' '{print $2}' | xargs)

# Write run record
echo "==============================================" >> run_record.dat
echo "Beta: $beta | Lattice: $lattice | Target: $TARGET" >> run_record.dat
echo "Update: $update_method" >> run_record.dat
echo "Host: $(hostname)" >> run_record.dat
start_time=$(date +%s)
date "+Start: %Y-%m-%d %H:%M:%S" >> run_record.dat

ulimit -s unlimited
export OMP_STACKSIZE=4500m
export OMP_PLACES=cores
export OMP_PROC_BIND=close
$EXEC
EXEC_STATUS=$?

echo ""
end_time=$(date +%s)
date "+End:   %Y-%m-%d %H:%M:%S" >> run_record.dat
elapsed_seconds=$((end_time - start_time))
days=$((elapsed_seconds / 86400))
hours=$(( (elapsed_seconds % 86400) / 3600 ))
minutes=$(( (elapsed_seconds % 3600) / 60 ))
seconds=$((elapsed_seconds % 60))
echo "Elapsed: ${days}d ${hours}h ${minutes}m ${seconds}s" >> run_record.dat
echo "==============================================" >> run_record.dat

if [ $EXEC_STATUS -eq 0 ]; then
    echo "Execution completed successfully."
else
    echo "Execution FAILED with exit code $EXEC_STATUS."
    exit 1
fi
