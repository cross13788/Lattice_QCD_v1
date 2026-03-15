//-----------------------------------------------
// Global input parameter definitions with defaults
//-----------------------------------------------
#include "input_module.hpp"

//-----------------------------------------------
// Lattice dimensions
//-----------------------------------------------
int nX = 8;
int nY = 8;
int nZ = 8;
int nT = 16;
int latticeVolume = 0;
int spatialVolume = 0;

//-----------------------------------------------
// Gauge action parameters
//-----------------------------------------------
real_t beta = 6.0;
std::string updateMethod = "heatbath";
real_t metropolisEpsilon = 0.2;
std::string startType = "cold";

//-----------------------------------------------
// Monte Carlo parameters
//-----------------------------------------------
int nTherm = 500;
int nConfig = 100;
int nSweepsBetween = 10;
int nOverrelax = 4;
int configSaveFrequency = 10;
bool saveBinaryConfigs = true;

//-----------------------------------------------
// Wilson fermion parameters (Phase 2)
//-----------------------------------------------
real_t kappa = 0.15;
real_t wilsonR = 1.0;
int cgMaxIter = 1000;
real_t cgTolerance = 1.0e-10;
std::string solverType = "bicgstab";
real_t c_sw = 0.0;
bool useClover = false;

//-----------------------------------------------
// HMC parameters (Phase 3)
//-----------------------------------------------
int nMDsteps = 20;
real_t mdStepSize = 0.01;
int nHMCtrajectories = 100;

//-----------------------------------------------
// Measurement parameters
//-----------------------------------------------
bool measureWilsonLoops = true;
int wilsonLoopRmax = 8;
int wilsonLoopTmax = 8;
bool measurePolyakov = true;
bool measureCorrelators = false;
bool measureActionDensity = false;
int nCoolingSteps = 0;
real_t coolingAlpha = 0.45;
bool measureWilsonFlow = false;
real_t flowStepSize = 0.01;
real_t maxFlowTime = 1.0;

//-----------------------------------------------
// I/O parameters
//-----------------------------------------------
std::string outputDirectory = "";
bool verboseOutput = true;

//-----------------------------------------------
// Random number seed
//-----------------------------------------------
int randomSeed = 12345;
