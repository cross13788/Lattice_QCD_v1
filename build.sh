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

    # nvc++ GCC toolchain binding.
    #
    # The HPC SDK ships a global localrc pinned to whatever GCC was present
    # at install time. When the host GCC is later upgraded (e.g. Fedora
    # 15 -> 16), nvc++ searches the now-deleted GCC tree and dies with
    # "limits.h: no directories in search list". Rather than depend on the
    # stale global localrc (and rather than the old crtbegin.o probe, which
    # silently no-ops on non-redhat layouts), regenerate a project-local
    # localrc against the *current* system g++ and point nvc++ at it via
    # -rc=. This keeps the binding inside the project tree and self-heals
    # across host compiler upgrades.
    PROJECT_ROOT="$PWD"
    NVHPC_RC_DIR="$PROJECT_ROOT/.nvhpc"
    NVHPC_RC="$NVHPC_RC_DIR/localrc"
    MAKELOCALRC="$(dirname "$(command -v nvc++)")/makelocalrc"
    if [ -x "$MAKELOCALRC" ]; then
        mkdir -p "$NVHPC_RC_DIR"
        if "$MAKELOCALRC" "$(dirname "$(command -v nvc++)")" \
                -gcc "$(command -v gcc)" -gpp "$(command -v g++)" \
                -g77 "$(command -v gfortran || command -v gcc)" \
                -x -d "$NVHPC_RC_DIR" >/dev/null 2>&1 && [ -f "$NVHPC_RC" ]; then
            EXTRA_MAKE_ARGS="$EXTRA_MAKE_ARGS EXTRA_CXXFLAGS=-rc=$NVHPC_RC"
            echo "  nvc++ localrc regenerated for GCC $(gcc -dumpfullversion 2>/dev/null || gcc -dumpversion): $NVHPC_RC"
        else
            echo "  Warning: makelocalrc failed; falling back to global nvc++ localrc"
        fi
    else
        echo "  Warning: makelocalrc not found next to nvc++; using global localrc"
    fi

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
