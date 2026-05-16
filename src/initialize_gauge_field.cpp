//-----------------------------------------------
// Initialize gauge field configuration
//
// "cold" start:        all links set to identity (ordered)
// "hot" start:         all links set to random SU(3) (disordered)
// "file:<path>" start: load links from a binary config (same format as
//                      write_gauge_config). Deterministic, RNG-free —
//                      used to feed an identical fixed configuration to
//                      both the CPU and GPU builds for correctness gates.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "interfaces_module.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>

void initialize_gauge_field(SU3matrix* gaugeField, const char* type)
{
//-----------------------------------------------
//   Local variables
//-----------------------------------------------
    int nLinks = latticeVolume * 4;

    if (strcmp(type, "cold") == 0) {
        printf("Initializing gauge field: cold start (all links = identity)\n");
        for (int i = 0; i < nLinks; i++) {
            su3_identity(gaugeField[i]);
        }
    }
    else if (strcmp(type, "hot") == 0) {
        printf("Initializing gauge field: hot start (random SU(3) links)\n");
        for (int i = 0; i < nLinks; i++) {
            su3_random_uniform(gaugeField[i]);
        }
    }
    else if (strncmp(type, "file:", 5) == 0) {
        const char* fileName = type + 5;
        printf("Initializing gauge field: loading from %s\n", fileName);

        FILE* fp = fopen(fileName, "rb");
        if (!fp) {
            printf("Error: Cannot open gauge config file '%s'\n", fileName);
            exit(1);
        }

        int fileNX, fileNY, fileNZ, fileNT, fileConfig;
        real_t fileBeta;
        fread(&fileNX, sizeof(int), 1, fp);
        fread(&fileNY, sizeof(int), 1, fp);
        fread(&fileNZ, sizeof(int), 1, fp);
        fread(&fileNT, sizeof(int), 1, fp);
        fread(&fileBeta, sizeof(real_t), 1, fp);
        fread(&fileConfig, sizeof(int), 1, fp);

        if (fileNX != nX || fileNY != nY || fileNZ != nZ || fileNT != nT) {
            printf("Error: Config file lattice (%d,%d,%d,%d) != current (%d,%d,%d,%d)\n",
                   fileNX, fileNY, fileNZ, fileNT, nX, nY, nZ, nT);
            fclose(fp);
            exit(1);
        }

        size_t nRead = fread(gaugeField, sizeof(SU3matrix), nLinks, fp);
        if ((int)nRead != nLinks) {
            printf("Error: Expected %d links, read %ld from %s\n",
                   nLinks, (long)nRead, fileName);
            fclose(fp);
            exit(1);
        }
        fclose(fp);
        printf("  Loaded config %d (beta=%.4f) from %s\n",
               fileConfig, fileBeta, fileName);
    }
    else {
        printf("Error: Unknown start type '%s'. Use 'cold', 'hot', or 'file:<path>'.\n", type);
        exit(1);
    }
}
