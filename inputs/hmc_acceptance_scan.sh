#!/bin/bash
# HMC acceptance rate scan - Test 3c
# Run HMC with different step sizes on 4^4 lattice
# to measure acceptance rate vs step size

set -e
cd "$(dirname "$0")/../run"

ulimit -s unlimited
export OMP_STACKSIZE=4500m
export OMP_PLACES=cores
export OMP_PROC_BIND=close

RESULTS_FILE="hmc_acceptance_scan_results.txt"
echo "# HMC Acceptance Rate Scan - Test 3c" > "$RESULTS_FILE"
echo "# MD_Steps  Step_Size  Acceptance_Rate(%)  Mean_|deltaH|  Mean_Plaquette" >> "$RESULTS_FILE"

for STEPSIZE in 0.01 0.02 0.05 0.10; do
    echo ""
    echo "========================================="
    echo "  Running HMC with step size = $STEPSIZE"
    echo "========================================="

    cat > lqcd.inp << EOF
# HMC acceptance scan: step size = $STEPSIZE

-------------------------------
[Lattice]
-------------------------------
nX: 4
nY: 4
nZ: 4
nT: 4

-------------------------------
[Gauge Action]
-------------------------------
Beta: 5.6
Update Method: hmc
Metropolis Epsilon: 0.2
Start Type: hot

-------------------------------
[Monte Carlo]
-------------------------------
Thermalization Sweeps: 20
Number of Configurations: 5
Sweeps Between Measurements: 2
Overrelaxation Sweeps: 3
Configuration Save Frequency: 10
Save Binary Configs: off
Random Seed: 54321

-------------------------------
[Measurements]
-------------------------------
Measure Wilson Loops: off
Wilson Loop R Max: 2
Wilson Loop T Max: 2
Measure Polyakov: off
Measure Correlators: off
Verbose Output: off

-------------------------------
[Wilson Fermions]
-------------------------------
Kappa: 0.12
Wilson r: 1.0
CG Max Iterations: 500
CG Tolerance: 1.0e-8
Solver Type: bicgstab

-------------------------------
[HMC]
-------------------------------
MD Steps: 10
MD Step Size: $STEPSIZE
HMC Trajectories: 50
EOF

    OUTPUT=$(./lqcd.exe 2>&1)

    # Extract acceptance rate from production line
    ACC_RATE=$(echo "$OUTPUT" | grep "HMC production complete" | grep -oP '[\d.]+%' | head -1 | tr -d '%')
    MEAN_PLAQ=$(echo "$OUTPUT" | grep "HMC traj" | tail -1 | grep -oP '<P>=[\d.]+' | sed 's/<P>=//')
    MEAN_DH=$(echo "$OUTPUT" | grep "HMC traj" | awk '{for(i=1;i<=NF;i++) if($i ~ /dH=/) {v=$i; gsub(/dH=/,"",v); s+=(v>0?v:-v); n++}} END{if(n>0) printf "%.6e", s/n}')

    echo "  Step size = $STEPSIZE: Acceptance = ${ACC_RATE}%"
    echo "10  $STEPSIZE  $ACC_RATE  $MEAN_DH  $MEAN_PLAQ" >> "$RESULTS_FILE"
done

echo ""
echo "========================================="
echo "  HMC Acceptance Scan Results"
echo "========================================="
cat "$RESULTS_FILE"
