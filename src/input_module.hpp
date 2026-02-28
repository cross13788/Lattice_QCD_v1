#pragma once
//-----------------------------------------------
// Global input parameters for Lattice QCD
//
// All parameters declared as extern here, defined
// with defaults in input_module.cpp. Populated by
// read_input_file() at startup.
//-----------------------------------------------
#include "constants_module.hpp"
#include <string>

//-----------------------------------------------
// Lattice dimensions
//-----------------------------------------------
extern int nX, nY, nZ, nT;
extern int latticeVolume;     // nX * nY * nZ * nT
extern int spatialVolume;     // nX * nY * nZ

//-----------------------------------------------
// Gauge action parameters
//-----------------------------------------------
extern real_t beta;                     // Coupling: beta = 2*NC/g^2
extern std::string updateMethod;        // "heatbath" or "metropolis"
extern real_t metropolisEpsilon;        // Step size for Metropolis
extern std::string startType;           // "cold" or "hot"

//-----------------------------------------------
// Monte Carlo parameters
//-----------------------------------------------
extern int nTherm;                      // Thermalization sweeps
extern int nConfig;                     // Number of configurations
extern int nSweepsBetween;              // Sweeps between measurements
extern int nOverrelax;                  // Overrelaxation sweeps per combined update
extern int configSaveFrequency;         // Save config every N measurements
extern bool saveBinaryConfigs;          // Write binary gauge configs

//-----------------------------------------------
// Wilson fermion parameters (Phase 2)
//-----------------------------------------------
extern real_t kappa;                    // Hopping parameter
extern real_t wilsonR;                  // Wilson parameter (usually 1)
extern int cgMaxIter;                   // Max CG iterations
extern real_t cgTolerance;              // CG residual tolerance
extern std::string solverType;          // "cg" or "bicgstab"

//-----------------------------------------------
// HMC parameters (Phase 3)
//-----------------------------------------------
extern int nMDsteps;                    // MD steps per trajectory
extern real_t mdStepSize;               // MD step size
extern int nHMCtrajectories;            // Number of HMC trajectories

//-----------------------------------------------
// Measurement parameters
//-----------------------------------------------
extern bool measureWilsonLoops;
extern int wilsonLoopRmax;              // Max R for Wilson loops
extern int wilsonLoopTmax;              // Max T for Wilson loops
extern bool measurePolyakov;
extern bool measureCorrelators;         // Phase 2
extern bool measureActionDensity;       // Per-site action density
extern int nCoolingSteps;               // APE cooling steps (0 = no cooling)
extern real_t coolingAlpha;             // APE smearing fraction

//-----------------------------------------------
// I/O parameters
//-----------------------------------------------
extern std::string outputDirectory;
extern bool verboseOutput;

//-----------------------------------------------
// Random number seed
//-----------------------------------------------
extern int randomSeed;
