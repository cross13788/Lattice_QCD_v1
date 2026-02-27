#!/bin/bash
#-----------------------------------------------
# Build script for Lattice QCD
#
# Usage:
#   ./build.sh [clean]
#-----------------------------------------------
set -e

if [ "$1" = "clean" ]; then
    echo "Cleaning build artifacts..."
    cd src/
    make clean
    echo "Done."
    exit 0
fi

echo ""
echo "Compiling..."
cd src/

cpus=$(nproc 2>/dev/null || echo 4)
echo "Number of CPUs: $cpus"
echo ""
make -j$cpus
echo ""
echo "Compilation complete. Executable ready in the run/ directory."
echo ""
