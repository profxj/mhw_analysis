/*
Support algorithms for MHW System analysis
*/

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#include "analysis.h"
#include "build.h"


// Add up the number of systems that a given location shows during the full period 
void spatial_systems(int *mask, int *shape, int *img, int *systems, int n_good, int max_Id) {

    // Init
    int DimX = shape[0];
    int DimY = shape[1];
    int DimZ = shape[2];

    long i,j,k;
    long ss;
    long idx;

    int* flag_tot = (int*) malloc ((max_Id+1) * sizeof(int));  // Add 1 for 0-indexing

    for (i = 0; i<DimX; i++)
        for (j = 0; j<DimY; j++) {
            // Init the systems (we don't care about the others)
            for (ss = 0; ss<n_good; ss++) {
                flag_tot[systems[ss]] = 0;
            }
            // Setup
            for (k = 0; k<DimZ; k++) {
                idx = convert_indices(i,j,k, DimY, DimZ);
                if (mask[idx] == 0)
                    continue;
                flag_tot[mask[idx]] = 1;
            }
            // Add em in
            for (ss = 0; ss<n_good; ss++) {
                if (flag_tot[systems[ss]] == 1) {
                    idx = i*DimY + j;
                    img[idx] += 1;
                }
            }
        }
}


// Add up the number of days a location is in a system category
void days_in_systems(int *mask, int *shape, int *img, int *systems) {

    // Init
    int DimX = shape[0];
    int DimY = shape[1];
    int DimZ = shape[2];

    long i,j,k;
    long idx;

    for (i = 0; i<DimX; i++)
        for (j = 0; j<DimY; j++) {
            // Setup
            for (k = 0; k<DimZ; k++) {
                idx = convert_indices(i,j,k, DimY, DimZ);
                // Any system?
                if (mask[idx] == 0)
                    continue;
                // One in our category?
                if (systems[mask[idx]] == 0)
                    continue;
                // Add it in
                idx = i*DimY + j;
                img[idx] += 1;
            }
        }
}

// Add up the number of days a location is in a system category
void days_in_systems_by_year(int *mask, int *shape, int *img, int *systems, int *year) {

    // Init
    int DimX = shape[0];
    int DimY = shape[1];
    int DimZ = shape[2];
    int DimZ2 = shape[3];

    long i,j,k;
    long idx, idx2;

    for (i = 0; i<DimX; i++)
        for (j = 0; j<DimY; j++) {
            // Loop on day
            for (k = 0; k<DimZ; k++) {
                idx = convert_indices(i,j,k, DimY, DimZ);
                // Any system?
                if (mask[idx] == 0)
                    continue;
                // One in our category?
                if (systems[mask[idx]] == 0)
                    continue;
                // Add it in
                idx2 = convert_indices(i,j,year[k], DimY, DimZ2);
                img[idx2] += 1;
            }
        }
}