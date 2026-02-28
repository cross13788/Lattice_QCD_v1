#!/bin/bash
# Run quenched spectroscopy at multiple kappa values
# Saves correlator data to separate directories for each kappa

set -e
cd "$(dirname "$0")/../run"

ulimit -s unlimited
export OMP_STACKSIZE=4500m
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Kappa mapping file for analysis
MAPFILE="spectroscopy_kappa_map.txt"
echo "# kappa  directory" > "$MAPFILE"

# Already ran kappa=0.120
OUTDIR="SU3_heatbath_beta6.00_8x8x8x16"
if [ -d "${OUTDIR}" ] && [ -f "${OUTDIR}/correlator_pion.dat" ]; then
    # Save kappa=0.120 data
    SAVE_DIR="spectro_k0.120"
    mkdir -p "$SAVE_DIR"
    cp ${OUTDIR}/correlator_*.dat "$SAVE_DIR/"
    echo "0.120  $SAVE_DIR" >> "$MAPFILE"
    echo "Saved existing kappa=0.120 data to $SAVE_DIR"
fi

for KAPPA_TAG in k140 k150 k154; do
    KAPPA_VAL=$(echo $KAPPA_TAG | sed 's/k/0./')

    echo ""
    echo "========================================="
    echo "  Running kappa = $KAPPA_VAL"
    echo "========================================="

    cp ../inputs/spectro_8x8x8x16_b60_${KAPPA_TAG}.inp lqcd.inp
    ./lqcd.exe 2>&1 | grep -E "(Pion:|Config|Propagator complete|converged)"

    # Save correlator data
    SAVE_DIR="spectro_k${KAPPA_VAL}"
    mkdir -p "$SAVE_DIR"
    cp ${OUTDIR}/correlator_*.dat "$SAVE_DIR/"
    echo "${KAPPA_VAL}  $SAVE_DIR" >> "$MAPFILE"
    echo "Saved kappa=${KAPPA_VAL} data to $SAVE_DIR"
    echo ""
done

echo ""
echo "========================================="
echo "  Spectroscopy scan complete"
echo "========================================="
cat "$MAPFILE"
