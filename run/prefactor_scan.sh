#!/bin/bash
# Scan clover force prefactor to find the value that minimizes |dH|
# at fixed eps=0.001, N=10 (tau=0.01)
#
# Current prefactor: c_sw/32 = 0.03125
# Test: multiply by factors 0.25, 0.5, 1.0, 2.0, 4.0

echo "=============================================="
echo "  Clover force prefactor scan"
echo "  eps=0.001, N=10, tau=0.01, cold start"
echo "=============================================="
echo ""

# Write input template
cat > lqcd.inp << 'EOF'
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
Start Type: cold
-------------------------------
[Monte Carlo]
-------------------------------
Thermalization Sweeps: 0
Number of Configurations: 0
Random Seed: 42
Verbose Output: off
-------------------------------
[Measurements]
-------------------------------
Measure Wilson Loops: off
Measure Polyakov: off
Measure Correlators: off
-------------------------------
[Wilson Fermions]
-------------------------------
Kappa: 0.12
Wilson r: 1.0
Clover Coefficient: 1.0
CG Max Iterations: 1000
CG Tolerance: 1.0e-10
Solver Type: bicgstab
-------------------------------
[HMC]
-------------------------------
MD Steps: 10
MD Step Size: 0.001
HMC Trajectories: 1
EOF

# Test each prefactor multiplier
for MULT in 0.125 0.25 0.5 1.0 2.0 4.0 8.0; do
    # Patch the source with the new prefactor
    PREF=$(echo "$MULT / 32.0" | bc -l | sed 's/^\./0./')
    cd /home/ross/Lattice_QCD_v1/src
    sed -i "s|real_t prefactor = .*|real_t prefactor = c_sw * $PREF;|" clover_fermion_force.cpp
    make cpu 2>&1 | tail -1
    cd /home/ross/Lattice_QCD_v1/run

    DH=$(timeout 120 ./lqcd.exe 2>&1 | grep "traj.*dH" | head -1 | sed 's/.*dH=//;s/ .*//')
    printf "  mult=%5.3f  prefactor=c_sw*%.6f  dH = %s\n" $MULT "$PREF" "$DH"
done
