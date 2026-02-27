//-----------------------------------------------
// Save gauge configuration to binary file
//
// Format: small ASCII header line, then raw binary
// data (all SU3 matrices written sequentially).
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "interfaces_module.hpp"
#include <cstdio>

void write_gauge_config(const SU3matrix* gaugeField, int configNum,
                        const char* directory)
{
//-----------------------------------------------
//   Local variables
//-----------------------------------------------
    char fileName[512];
    int nLinks = latticeVolume * 4;

    snprintf(fileName, sizeof(fileName), "%s/config_%04d.bin", directory, configNum);

    FILE* fp = fopen(fileName, "wb");
    if (!fp) {
        printf("Warning: Cannot write gauge config to %s\n", fileName);
        return;
    }

    // Write header as binary metadata
    fwrite(&nX, sizeof(int), 1, fp);
    fwrite(&nY, sizeof(int), 1, fp);
    fwrite(&nZ, sizeof(int), 1, fp);
    fwrite(&nT, sizeof(int), 1, fp);
    fwrite(&beta, sizeof(real_t), 1, fp);
    fwrite(&configNum, sizeof(int), 1, fp);

    // Write all link matrices
    fwrite(gaugeField, sizeof(SU3matrix), nLinks, fp);

    fclose(fp);

    if (verboseOutput) {
        printf("  Saved config %d to %s\n", configNum, fileName);
    }
}
