//-----------------------------------------------
// INI-style input file parser
//
// Format:
//   # Comment line
//   [Section Name]
//   Key: Value
//
// Reads run/lqcd.inp line by line, detects
// [Section] headers, splits Key: Value on the
// colon, and calls set_parameter() for each.
//-----------------------------------------------
#include "constants_module.hpp"
#include "input_module.hpp"
#include "interfaces_module.hpp"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>

void read_input_file()
{
//-----------------------------------------------
//   Local variables
//-----------------------------------------------
    FILE* fp;
    char line[256];
    char section[64] = "";
    char key[128];
    char value[128];

//-----------------------------------------------
//   Open input file
//-----------------------------------------------
    fp = fopen("lqcd.inp", "r");
    if (!fp) {
        printf("Error: Cannot open input file lqcd.inp\n");
        exit(1);
    }

    printf("Reading input file: lqcd.inp\n");
    printf("-----------------------------------------------\n");

//-----------------------------------------------
//   Parse line by line
//-----------------------------------------------
    while (fgets(line, sizeof(line), fp)) {
        // Remove trailing newline
        int len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';

        // Skip empty lines and comment lines
        if (len == 0) continue;
        if (line[0] == '#') continue;

        // Skip dashed separator lines
        if (line[0] == '-') continue;

        // Check for section header [Section Name]
        if (line[0] == '[') {
            char* closing = strchr(line, ']');
            if (closing) {
                int sectionLen = closing - line - 1;
                strncpy(section, line + 1, sectionLen);
                section[sectionLen] = '\0';

                // Trim whitespace
                char* s = section;
                while (*s == ' ') s++;
                char* e = section + strlen(section) - 1;
                while (e > s && *e == ' ') *e-- = '\0';
                if (s != section) memmove(section, s, strlen(s) + 1);
            }
            continue;
        }

        // Check for Key: Value pair
        char* colon = strchr(line, ':');
        if (!colon) continue;

        // Extract key (before colon)
        int keyLen = colon - line;
        strncpy(key, line, keyLen);
        key[keyLen] = '\0';

        // Extract value (after colon)
        strncpy(value, colon + 1, sizeof(value) - 1);
        value[sizeof(value) - 1] = '\0';

        // Trim whitespace from key
        char* ks = key;
        while (*ks == ' ') ks++;
        char* ke = key + strlen(key) - 1;
        while (ke > ks && *ke == ' ') *ke-- = '\0';
        if (ks != key) memmove(key, ks, strlen(ks) + 1);

        // Trim whitespace from value
        char* vs = value;
        while (*vs == ' ') vs++;
        char* ve = value + strlen(value) - 1;
        while (ve > vs && *ve == ' ') *ve-- = '\0';
        if (vs != value) memmove(value, vs, strlen(vs) + 1);

        // Set the parameter
        set_parameter(section, key, value);
    }

    fclose(fp);

//-----------------------------------------------
//   Compute derived quantities
//-----------------------------------------------
    latticeVolume = nX * nY * nZ * nT;
    spatialVolume = nX * nY * nZ;

    printf("-----------------------------------------------\n");
    printf("\n");
}
