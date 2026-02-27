//-----------------------------------------------
// Initialize gauge field configuration
//
// "cold" start: all links set to identity (ordered)
// "hot" start:  all links set to random SU(3) (disordered)
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "su3_module.hpp"
#include "interfaces_module.hpp"
#include <cstdio>
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
    else {
        printf("Error: Unknown start type '%s'. Use 'cold' or 'hot'.\n", type);
        exit(1);
    }
}
