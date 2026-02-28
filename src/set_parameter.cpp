//-----------------------------------------------
// Map section/key/value strings to global variables
//
// Called by read_input_file() for each Key: Value
// pair found in the input file.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static bool str_eq(const char* a, const char* b)
{
    return strcmp(a, b) == 0;
}

static bool parse_bool(const char* value)
{
    return str_eq(value, "on") || str_eq(value, "true") || str_eq(value, "yes") || str_eq(value, "1");
}

void set_parameter(const char* section, const char* key, const char* value)
{
//-----------------------------------------------
//   Lattice section
//-----------------------------------------------
    if (str_eq(section, "Lattice")) {
        if (str_eq(key, "nX"))         { nX = atoi(value); }
        else if (str_eq(key, "nY"))    { nY = atoi(value); }
        else if (str_eq(key, "nZ"))    { nZ = atoi(value); }
        else if (str_eq(key, "nT"))    { nT = atoi(value); }
        else { printf("  Warning: Unknown key '%s' in [%s]\n", key, section); }
    }

//-----------------------------------------------
//   Gauge Action section
//-----------------------------------------------
    else if (str_eq(section, "Gauge Action")) {
        if (str_eq(key, "Beta"))                { beta = atof(value); }
        else if (str_eq(key, "Update Method"))  { updateMethod = value; }
        else if (str_eq(key, "Metropolis Epsilon")) { metropolisEpsilon = atof(value); }
        else if (str_eq(key, "Start Type"))     { startType = value; }
        else { printf("  Warning: Unknown key '%s' in [%s]\n", key, section); }
    }

//-----------------------------------------------
//   Monte Carlo section
//-----------------------------------------------
    else if (str_eq(section, "Monte Carlo")) {
        if (str_eq(key, "Thermalization Sweeps"))     { nTherm = atoi(value); }
        else if (str_eq(key, "Number of Configurations")) { nConfig = atoi(value); }
        else if (str_eq(key, "Sweeps Between Measurements")) { nSweepsBetween = atoi(value); }
        else if (str_eq(key, "Overrelaxation Sweeps")) { nOverrelax = atoi(value); }
        else if (str_eq(key, "Configuration Save Frequency")) { configSaveFrequency = atoi(value); }
        else if (str_eq(key, "Save Binary Configs"))  { saveBinaryConfigs = parse_bool(value); }
        else if (str_eq(key, "Random Seed"))           { randomSeed = atoi(value); }
        else { printf("  Warning: Unknown key '%s' in [%s]\n", key, section); }
    }

//-----------------------------------------------
//   Measurements section
//-----------------------------------------------
    else if (str_eq(section, "Measurements")) {
        if (str_eq(key, "Measure Wilson Loops"))  { measureWilsonLoops = parse_bool(value); }
        else if (str_eq(key, "Wilson Loop R Max")) { wilsonLoopRmax = atoi(value); }
        else if (str_eq(key, "Wilson Loop T Max")) { wilsonLoopTmax = atoi(value); }
        else if (str_eq(key, "Measure Polyakov"))  { measurePolyakov = parse_bool(value); }
        else if (str_eq(key, "Measure Correlators")) { measureCorrelators = parse_bool(value); }
        else if (str_eq(key, "Measure Action Density")) { measureActionDensity = parse_bool(value); }
        else if (str_eq(key, "Cooling Steps"))     { nCoolingSteps = atoi(value); }
        else if (str_eq(key, "Cooling Alpha"))     { coolingAlpha = atof(value); }
        else if (str_eq(key, "Verbose Output"))    { verboseOutput = parse_bool(value); }
        else { printf("  Warning: Unknown key '%s' in [%s]\n", key, section); }
    }

//-----------------------------------------------
//   Wilson Fermions section (Phase 2)
//-----------------------------------------------
    else if (str_eq(section, "Wilson Fermions")) {
        if (str_eq(key, "Kappa"))               { kappa = atof(value); }
        else if (str_eq(key, "Wilson r"))        { wilsonR = atof(value); }
        else if (str_eq(key, "CG Max Iterations")) { cgMaxIter = atoi(value); }
        else if (str_eq(key, "CG Tolerance"))    { cgTolerance = atof(value); }
        else if (str_eq(key, "Solver Type"))     { solverType = value; }
        else { printf("  Warning: Unknown key '%s' in [%s]\n", key, section); }
    }

//-----------------------------------------------
//   HMC section (Phase 3)
//-----------------------------------------------
    else if (str_eq(section, "HMC")) {
        if (str_eq(key, "MD Steps"))             { nMDsteps = atoi(value); }
        else if (str_eq(key, "MD Step Size"))    { mdStepSize = atof(value); }
        else if (str_eq(key, "HMC Trajectories")) { nHMCtrajectories = atoi(value); }
        else { printf("  Warning: Unknown key '%s' in [%s]\n", key, section); }
    }

    else {
        printf("  Warning: Unknown section [%s] (key: %s)\n", section, key);
    }
}
