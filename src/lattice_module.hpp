#pragma once
//-----------------------------------------------
// Lattice geometry: site indexing, neighbor tables
//
// Uses periodic boundary conditions. Neighbor
// tables are precomputed at startup to avoid
// modular arithmetic in hot loops.
//-----------------------------------------------
#include "constants_module.hpp"

// Neighbor index arrays (allocated after reading input)
// Access: neighborPlus[site * 4 + mu]  = site + mu_hat
//         neighborMinus[site * 4 + mu] = site - mu_hat
extern int* neighborPlus;
extern int* neighborMinus;

// Convert (x,y,z,t) coordinates to linear site index
int site_index(int x, int y, int z, int t);

// Convert linear site index back to (x,y,z,t) coordinates
void site_coords(int idx, int& x, int& y, int& z, int& t);

// Parity of a site: 0 (even) or 1 (odd)
// Used for checkerboard decomposition
int site_parity(int idx);

// Allocate and fill neighbor tables with periodic BCs
void initialize_lattice_geometry();

// Free neighbor table memory
void free_lattice_geometry();
