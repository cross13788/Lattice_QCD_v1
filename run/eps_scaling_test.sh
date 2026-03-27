#!/bin/bash
# Eps-scaling test for clover HMC force validation
# Fixed tau = 0.01, vary eps = tau/N
# If force is correct: dH ~ C * eps^2  (at fixed tau)

echo "=============================================="
echo "  Clover HMC eps-scaling test (tau = 0.01)"
echo "  Expect: dH proportional to eps^2"
echo "=============================================="
echo ""

TAU=0.01

for N in 5 10 20 50; do
    EPS=$(echo "$TAU / $N" | bc -l | sed 's/^\./0./')

    # Write input file
    cat > lqcd.inp << EOF
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
MD Steps: $N
MD Step Size: $EPS
HMC Trajectories: 1
EOF

    # Run and extract dH
    DH=$(timeout 300 ./lqcd.exe 2>&1 | grep "traj.*dH" | head -1 | sed 's/.*dH=//;s/ .*//')
    printf "  N=%3d  eps=%.6f  dH = %s\n" $N "$EPS" "$DH"
done

echo ""
echo "If dH ~ eps^2: doubling N (halving eps) should reduce |dH| by ~4x"
