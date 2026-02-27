#!/bin/bash
# remote_run.sh - Deploy, compile, execute, and fetch Lattice QCD results from a remote cluster
#
# Usage:
#   ./remote_run.sh <cluster> [input_file] [poll_seconds]

set -e

REMOTE_USER="${LQCD_USER:-$(whoami)}"
BASE_PATH="${LQCD_REMOTE_PATH:-/home/${REMOTE_USER}/Lattice_QCD}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <cluster> [input_file] [poll_seconds]"
    exit 1
fi

HOST_NAME="$1"
INPUT_FILE="${2:-}"
POLL_INTERVAL="${3:-60}"

if [ -n "$INPUT_FILE" ]; then
    if [ ! -f "$INPUT_FILE" ]; then
        echo "Error: Input file '$INPUT_FILE' not found"
        exit 1
    fi
    cp "$INPUT_FILE" run/lqcd.inp
    echo "Using input: $INPUT_FILE"
fi

INPUT="run/lqcd.inp"
if [ ! -f "$INPUT" ]; then
    echo "Error: $INPUT not found"
    exit 1
fi

# Parse input for output directory naming
beta=$(grep "^Beta:" "$INPUT" | awk -F': ' '{print $2}' | xargs)
nX=$(grep "^nX:" "$INPUT" | awk -F': ' '{print $2}' | xargs)
nY=$(grep "^nY:" "$INPUT" | awk -F': ' '{print $2}' | xargs)
nZ=$(grep "^nZ:" "$INPUT" | awk -F': ' '{print $2}' | xargs)
nT=$(grep "^nT:" "$INPUT" | awk -F': ' '{print $2}' | xargs)
update=$(grep "^Update Method:" "$INPUT" | awk -F': ' '{print $2}' | xargs)

OUTPUT_DIR="SU3_${update}_beta${beta}_${nX}x${nY}x${nZ}x${nT}"

echo ""
echo "=== Lattice QCD Remote Run ==="
echo "  Cluster:    $HOST_NAME"
echo "  Output dir: run/$OUTPUT_DIR/"
echo ""

# SSH multiplexing
CONTROL_PATH="/tmp/ssh-lqcd-${HOST_NAME}"
OWNS_MASTER=false

ssh_cmd() {
    ssh -o ControlPath="$CONTROL_PATH" ${REMOTE_USER}@${HOST_NAME} "$@"
}

if ! ssh -O check -o ControlPath="$CONTROL_PATH" ${REMOTE_USER}@${HOST_NAME} 2>/dev/null; then
    echo "Establishing SSH connection to $HOST_NAME..."
    ssh -o ControlMaster=yes -o ControlPath="$CONTROL_PATH" -o ControlPersist=600 -fN ${REMOTE_USER}@${HOST_NAME}
    OWNS_MASTER=true
fi

cleanup() {
    if [ "$OWNS_MASTER" = true ]; then
        ssh -O exit -o ControlPath="$CONTROL_PATH" ${REMOTE_USER}@${HOST_NAME} 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Deploy
echo "--- Deploying to $HOST_NAME ---"
ssh_cmd "mkdir -p ${BASE_PATH}/src ${BASE_PATH}/run"

echo "  src/..."
rsync -az --checksum \
    -e "ssh -o ControlPath=$CONTROL_PATH" \
    --include='*.cpp' \
    --include='*.hpp' \
    --include='Makefile' \
    --exclude='*.o' \
    --exclude='*' \
    src/ \
    ${REMOTE_USER}@${HOST_NAME}:${BASE_PATH}/src/

echo "  scripts..."
rsync -az --checksum \
    -e "ssh -o ControlPath=$CONTROL_PATH" \
    --include='*.sh' \
    --exclude='*' \
    ./ \
    ${REMOTE_USER}@${HOST_NAME}:${BASE_PATH}/

echo "  run/ input files..."
rsync -az --checksum \
    -e "ssh -o ControlPath=$CONTROL_PATH" \
    --exclude='*/' \
    --exclude='run_record.dat' \
    --exclude='lqcd.exe' \
    --exclude='remote_run.log' \
    run/ \
    ${REMOTE_USER}@${HOST_NAME}:${BASE_PATH}/run/

echo "  Deploy complete."

# Compile
echo ""
echo "--- Compiling on $HOST_NAME ---"
if ! ssh_cmd "bash -l -c 'cd ${BASE_PATH} && bash build.sh'"; then
    echo "Error: Compilation failed on $HOST_NAME"
    exit 1
fi

# Execute
echo ""
echo "--- Starting execution on $HOST_NAME ---"
REMOTE_PID=$(ssh_cmd "cd ${BASE_PATH}/run && rm -f .exit_status && nohup bash -l -c '
ulimit -s unlimited
export OMP_STACKSIZE=4500m
export OMP_PLACES=cores
export OMP_PROC_BIND=close
./lqcd.exe
echo \$? > .exit_status
' > remote_run.log 2>&1 & echo \$!")

if [ -z "$REMOTE_PID" ]; then
    echo "Error: Failed to start remote job"
    exit 1
fi
echo "Remote PID: $REMOTE_PID"

trap '
echo ""
echo "Interrupted. Remote job (PID $REMOTE_PID) still running on $HOST_NAME."
echo "  Check:  ssh $HOST_NAME \"kill -0 $REMOTE_PID 2>/dev/null && echo RUNNING || echo DONE\""
echo "  Log:    ssh $HOST_NAME \"tail ${BASE_PATH}/run/remote_run.log\""
echo "  Fetch:  rsync -avz ${REMOTE_USER}@${HOST_NAME}:${BASE_PATH}/run/${OUTPUT_DIR}/ run/${OUTPUT_DIR}/"
if [ "$OWNS_MASTER" = true ]; then
    ssh -O exit -o ControlPath="$CONTROL_PATH" ${REMOTE_USER}@${HOST_NAME} 2>/dev/null || true
fi
exit 130
' INT

# Poll until complete
echo "Polling every ${POLL_INTERVAL}s... (Ctrl+C to detach, job keeps running)"
echo ""
SECONDS_ELAPSED=0
while true; do
    if ! ssh_cmd "kill -0 $REMOTE_PID 2>/dev/null"; then
        break
    fi

    PROGRESS=$(ssh_cmd "tail -1 ${BASE_PATH}/run/${OUTPUT_DIR}/output_file.out 2>/dev/null" || echo "")
    ELAPSED_MIN=$((SECONDS_ELAPSED / 60))
    if [ -n "$PROGRESS" ]; then
        printf "\r\033[K  [%dm] %s" "$ELAPSED_MIN" "$PROGRESS"
    else
        printf "\r\033[K  [%dm] Waiting for output..." "$ELAPSED_MIN"
    fi

    sleep $POLL_INTERVAL
    SECONDS_ELAPSED=$((SECONDS_ELAPSED + POLL_INTERVAL))
done
printf "\r\033[K"

# Check exit status
EXIT_STATUS=$(ssh_cmd "cat ${BASE_PATH}/run/.exit_status 2>/dev/null" || echo "unknown")

if [ "$EXIT_STATUS" != "0" ]; then
    echo "=== Run FAILED (exit code: $EXIT_STATUS) ==="
    echo "Remote log (last 20 lines):"
    ssh_cmd "tail -20 ${BASE_PATH}/run/remote_run.log 2>/dev/null" || true
    exit 1
fi

ELAPSED_MIN=$((SECONDS_ELAPSED / 60))
echo "=== Run completed in ~${ELAPSED_MIN}m ==="

# Fetch results
echo ""
echo "--- Fetching results ---"
mkdir -p "run/${OUTPUT_DIR}"
rsync -avz \
    -e "ssh -o ControlPath=$CONTROL_PATH" \
    ${REMOTE_USER}@${HOST_NAME}:${BASE_PATH}/run/${OUTPUT_DIR}/ \
    run/${OUTPUT_DIR}/

echo ""
echo "Results saved to: run/${OUTPUT_DIR}/"
