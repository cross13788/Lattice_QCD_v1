//-----------------------------------------------
// Lattice geometry implementation
//
// Site indexing: idx = x + nX*(y + nY*(z + nZ*t))
// This gives contiguous spatial sites for fixed t,
// which is cache-friendly for time-slice operations.
//-----------------------------------------------
#include "lattice_module.hpp"
#include "input_module.hpp"
#include <cstdlib>
#include <cstdio>

int* neighborPlus  = nullptr;
int* neighborMinus = nullptr;

int site_index(int x, int y, int z, int t)
{
    return x + nX * (y + nY * (z + nZ * t));
}

void site_coords(int idx, int& x, int& y, int& z, int& t)
{
    x = idx % nX;
    idx /= nX;
    y = idx % nY;
    idx /= nY;
    z = idx % nZ;
    t = idx / nZ;
}

int site_parity(int idx)
{
    int x, y, z, t;
    site_coords(idx, x, y, z, t);
    return (x + y + z + t) % 2;
}

void initialize_lattice_geometry()
{
//-----------------------------------------------
//   Compute derived quantities
//-----------------------------------------------
    latticeVolume = nX * nY * nZ * nT;
    spatialVolume = nX * nY * nZ;

//-----------------------------------------------
//   Allocate neighbor tables
//-----------------------------------------------
    neighborPlus  = (int*)malloc(latticeVolume * 4 * sizeof(int));
    neighborMinus = (int*)malloc(latticeVolume * 4 * sizeof(int));

    if (!neighborPlus || !neighborMinus) {
        printf("Error: Failed to allocate neighbor tables\n");
        exit(1);
    }

//-----------------------------------------------
//   Fill neighbor tables with periodic BCs
//-----------------------------------------------
    for (int idx = 0; idx < latticeVolume; idx++) {
        int x, y, z, t;
        site_coords(idx, x, y, z, t);

        // mu = 0 (x-direction)
        neighborPlus [idx * 4 + 0] = site_index((x + 1) % nX, y, z, t);
        neighborMinus[idx * 4 + 0] = site_index((x - 1 + nX) % nX, y, z, t);

        // mu = 1 (y-direction)
        neighborPlus [idx * 4 + 1] = site_index(x, (y + 1) % nY, z, t);
        neighborMinus[idx * 4 + 1] = site_index(x, (y - 1 + nY) % nY, z, t);

        // mu = 2 (z-direction)
        neighborPlus [idx * 4 + 2] = site_index(x, y, (z + 1) % nZ, t);
        neighborMinus[idx * 4 + 2] = site_index(x, y, (z - 1 + nZ) % nZ, t);

        // mu = 3 (t-direction)
        neighborPlus [idx * 4 + 3] = site_index(x, y, z, (t + 1) % nT);
        neighborMinus[idx * 4 + 3] = site_index(x, y, z, (t - 1 + nT) % nT);
    }

    if (verboseOutput) {
        printf("Lattice geometry initialized: %d x %d x %d x %d = %d sites\n",
               nX, nY, nZ, nT, latticeVolume);
    }
}

void free_lattice_geometry()
{
    free(neighborPlus);
    free(neighborMinus);
    neighborPlus  = nullptr;
    neighborMinus = nullptr;
}
