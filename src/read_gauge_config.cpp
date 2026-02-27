//-----------------------------------------------
// Read a gauge configuration from binary file
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "interfaces_module.hpp"
#include <cstdio>
#include <cstdlib>

void read_gauge_config(SU3matrix* gaugeField, int configNum,
                       const char* directory)
{
//-----------------------------------------------
//   Local variables
//-----------------------------------------------
    char fileName[512];
    int nLinks = latticeVolume * 4;
    int fileNX, fileNY, fileNZ, fileNT, fileConfig;
    real_t fileBeta;

    snprintf(fileName, sizeof(fileName), "%s/config_%04d.bin", directory, configNum);

    FILE* fp = fopen(fileName, "rb");
    if (!fp) {
        printf("Error: Cannot read gauge config from %s\n", fileName);
        exit(1);
    }

    // Read header
    fread(&fileNX, sizeof(int), 1, fp);
    fread(&fileNY, sizeof(int), 1, fp);
    fread(&fileNZ, sizeof(int), 1, fp);
    fread(&fileNT, sizeof(int), 1, fp);
    fread(&fileBeta, sizeof(real_t), 1, fp);
    fread(&fileConfig, sizeof(int), 1, fp);

    // Verify dimensions match
    if (fileNX != nX || fileNY != nY || fileNZ != nZ || fileNT != nT) {
        printf("Error: Config file lattice (%d,%d,%d,%d) != current (%d,%d,%d,%d)\n",
               fileNX, fileNY, fileNZ, fileNT, nX, nY, nZ, nT);
        fclose(fp);
        exit(1);
    }

    // Read all link matrices
    size_t nRead = fread(gaugeField, sizeof(SU3matrix), nLinks, fp);
    if ((int)nRead != nLinks) {
        printf("Error: Expected %d links, read %ld from %s\n",
               nLinks, (long)nRead, fileName);
        fclose(fp);
        exit(1);
    }

    fclose(fp);

    if (verboseOutput) {
        printf("  Loaded config %d from %s (beta=%.4f)\n",
               fileConfig, fileName, fileBeta);
    }
}
