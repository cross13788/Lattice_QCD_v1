//-----------------------------------------------
// Write meson correlator and effective mass data
// to ASCII files.
//
// Output files:
//   correlator_CHANNEL.dat — C(t) vs t
//   effective_mass_CHANNEL.dat — m_eff(t) vs t
//
// Each row: config_number  t  C(t)  m_eff(t)
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "interfaces_module.hpp"
#include <cstdio>
#include <cstring>

static const char* channel_name(int channel)
{
    switch (channel) {
        case 0: return "pion";
        case 1: return "rho_x";
        case 2: return "rho_y";
        case 3: return "rho_z";
        case 4: return "scalar";
        case 5: return "a1_x";
        case 6: return "a1_y";
        case 7: return "a1_z";
        default: return "unknown";
    }
}

void write_correlators(int configNum, int channel,
                       const real_t* corr, const real_t* meff,
                       const char* directory)
{
//-----------------------------------------------
//   Build filename
//-----------------------------------------------
    char filename[512];
    snprintf(filename, sizeof(filename), "%s/correlator_%s.dat",
             directory, channel_name(channel));

//-----------------------------------------------
//   Open file (append mode for multiple configs)
//-----------------------------------------------
    FILE* fp;
    if (configNum == 1) {
        fp = fopen(filename, "w");
        if (!fp) {
            printf("Error: cannot open %s\n", filename);
            return;
        }
        fprintf(fp, "# config  t  C(t)  m_eff(t)\n");
    } else {
        fp = fopen(filename, "a");
        if (!fp) {
            printf("Error: cannot open %s\n", filename);
            return;
        }
    }

//-----------------------------------------------
//   Write data
//-----------------------------------------------
    for (int t = 0; t < nT; t++) {
        fprintf(fp, "%5d  %3d  %20.12e  %16.10f\n",
                configNum, t, corr[t], meff[t]);
    }

    fclose(fp);
}
