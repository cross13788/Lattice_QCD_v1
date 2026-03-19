#!/bin/bash
#-----------------------------------------------
# Build script for Lattice QCD
#
# Usage:
#   ./build.sh [cpu|gpu] [clean]
#
# Default target: cpu
#
# GPU build uses nvc++ for everything (host C++
# and CUDA .cu files). No nvcc needed — this
# avoids GCC version incompatibilities entirely.
#-----------------------------------------------
set -e

TARGET="${1:-cpu}"
LAST_TARGET_FILE="src/.last_target"

if [ "$TARGET" = "clean" ] || [ "$2" = "clean" ]; then
    echo "Cleaning build artifacts..."
    cd src/
    make clean
    rm -f .last_target
    echo "Done."
    exit 0
fi

# Validate target
if [ "$TARGET" != "cpu" ] && [ "$TARGET" != "gpu" ]; then
    echo "Usage: $0 [cpu|gpu] [clean]"
    echo "  cpu  - Build with g++ (default)"
    echo "  gpu  - Build with nvc++ (requires NVIDIA HPC SDK + CUDA)"
    exit 1
fi

# Check for compiler-switch: if last build was a different target, clean first
if [ -f "$LAST_TARGET_FILE" ]; then
    LAST=$(cat "$LAST_TARGET_FILE")
    if [ "$LAST" != "$TARGET" ]; then
        echo "Target changed from '$LAST' to '$TARGET' — cleaning stale objects..."
        cd src/
        make clean
        cd ..
    fi
fi

echo ""
echo "Building Lattice QCD ($TARGET)..."

EXTRA_MAKE_ARGS=""

# GPU-specific setup
if [ "$TARGET" = "gpu" ]; then
    if ! command -v nvc++ &> /dev/null; then
        echo "Error: nvc++ not found. Load NVIDIA HPC SDK module or add to PATH."
        exit 1
    fi

    # nvc++ GCC toolchain detection (same fix as TDHFBCS_v3)
    for ver in 15 14 13 12 11; do
        gccdir="/usr/lib/gcc/x86_64-redhat-linux/$ver"
        if [ -f "$gccdir/crtbegin.o" ]; then
            EXTRA_MAKE_ARGS="$EXTRA_MAKE_ARGS EXTRA_CXXFLAGS=--gcc-toolchain=/usr"
            echo "  Using GCC toolchain for nvc++: /usr (GCC $ver)"
            break
        fi
    done

    # Auto-detect GPU architecture from nvidia-smi
    GPU_ARCH="cc75"
    if command -v nvidia-smi &>/dev/null; then
        DETECTED_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
        if [ -n "$DETECTED_CC" ]; then
            GPU_ARCH="cc${DETECTED_CC}"
        fi
    fi
    EXTRA_MAKE_ARGS="$EXTRA_MAKE_ARGS GPU_ARCH=$GPU_ARCH"
    echo "  GPU architecture: $GPU_ARCH"

    echo "  Using nvc++: $(nvc++ --version 2>&1 | head -1)"
fi

cd src/

cpus=$(nproc 2>/dev/null || echo 4)
echo "  CPUs:  $cpus"
echo ""

make -j$cpus $TARGET $EXTRA_MAKE_ARGS

# Record which target was built
echo "$TARGET" > .last_target

echo ""
if [ "$TARGET" = "gpu" ]; then
    echo "GPU build complete. Executable: run/lqcd_gpu.exe"
else
    echo "CPU build complete. Executable: run/lqcd.exe"
fi
echo ""
