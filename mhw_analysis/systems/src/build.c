/*
Support algorithms for bspline.
*/

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#include "build.h"

void lunion(int *parent, int x, int y) {
    /*

    Args:
       parent:
       x:
       y:
    */
    //..find root labels
    while (parent[x] != 0)
        x = parent[x];

    while (parent[y] != 0)
        y = parent[y];

    if (x < y) {
        parent[y] = x;
    }
    else if (x > y) {
        parent[x] = y;
    }
}

long convert_indices(long i, long j, long k, int DimY, int DimZ) {
    return i*DimY*DimZ + j*DimZ + k;
}

void first_pass(char *cube, int *mask, int *shape, int *parent, int *category) {
    /*
    */

    // Init
    int label = 0;

    int DimX = shape[0];
    int DimY = shape[1];
    int DimZ = shape[2];

    // printf("%d, %d, %d\n", DimX, DimY, DimZ);

    long idx, idx2;
    long i,j,k;
    long ii,jj,kk;
    int p;
    int count;

    int all_zero;

    // Init the mask
    for (i = 0; i<DimX; i++)
        for (j = 0; j<DimY; j++)
            for (k = 0; k<DimZ; k++) {
                idx = convert_indices(i,j,k, DimY, DimZ);
                mask[idx] = 0;
            }

    // Labels
    int prior_labels[27];
    int minlabel;
    int maxnlabels = 100000000;
    int this_label;

    // Loop me!
    for (i = 1; i<DimX-1; i++)
        for (j = 1; j<DimY-1; j++)
            for (k = 1; k<DimZ-1; k++) {
                idx = convert_indices(i,j,k, DimY, DimZ);
                // Debuggin
                //if (j == 1 && k == 1)
                //    printf("idx = %ld\n", idx);

                if (cube[idx] == 0)
                    continue;

                // prior_labels = mask[i-1:i+2,j-1:j+2,k-1:k+2].flatten()
                // Fill prior_labels and check for non-zero values
                // Also determine the minimum value of the label
                count = 0;
                all_zero = 0;
                minlabel = maxnlabels;
                for (ii=i-1; ii<=i+1; ii++)
                    for (jj=j-1; jj<=j+1; jj++)
                        for (kk=k-1; kk<=k+1; kk++) {
                            idx2 = convert_indices(ii,jj,kk, DimY, DimZ);
                            // prior_labels[count] = mask[ii,jj,kk];
                            prior_labels[count] = mask[idx2];
                            if (prior_labels[count] != 0) {
                                all_zero = 1;
                                if (prior_labels[count] < minlabel)
                                    minlabel = prior_labels[count];
                            }
                            count++;
                        }
                /* Debuggin
                if (i==2 && j==5 && k==3) {
                    for (p=0; p<27; p++) { // p in range(prior_labels.size):
                        printf("A %d\n", prior_labels[p]);
                    }
                    printf("A2 %d\n", minlabel);
                }
                */

                if (all_zero == 0) {
                    label=label+1;
                    if (label > maxnlabels) // # STOP "Increase stack size (maxnlabels)!"
                        printf("Exception happened here\n");
                    // mask[i][j][k] = label;
                    mask[idx] = label;
                    category[label] = cube[idx];
                } else { // # !..this spaxel is connected to another one
                    this_label = minlabel; // np.min(prior_labels[prior_labels > 0])
                    mask[idx] = this_label;
                    // #!..update parent tree
                    for (p=0; p<27; p++) { // p in range(prior_labels.size):
                        if (prior_labels[p] != 0  &&  prior_labels[p] != this_label)
                            lunion(parent, this_label, prior_labels[p]);
                    }
                    category[minlabel] = fmax(category[minlabel], cube[idx]);
                }
             }
}

void second_pass(int *mask, int *parent, int *shape, int *NSpax, int *category) {

    // Init
    int DimX = shape[0];
    int DimY = shape[1];
    int DimZ = shape[2];

    long i,j,k;
    long idx;
    int this_label;
    int p;

    // !..second pass:
    // !... replace labels using the parent tree
    // !... get NSpax for each individual connected component
    // for i in range(DimX):
    //    for j in range(DimY):
    //        for k in range(DimZ):
    for (i = 0; i<DimX; i++)
        for (j = 0; j<DimY; j++)
            for (k = 0; k<DimZ; k++) {
                idx = convert_indices(i,j,k, DimY, DimZ);
                this_label=mask[idx];
                if (this_label != 0) {
                    // #!..assign value from parent tree
                    p = this_label;
                    while (parent[p] != 0)
                       p = parent[p];

                    mask[idx] = p;
                    category[p] = category[this_label];  // This may be superfluous
                    // !..update NSpax counter associated with this label
                    NSpax[p] = NSpax[p]+1;
                }
            }
}

void final_pass(int ndet, int *mask, int *shape, float *xcen, float *ycen, float *zcen, int *xboxmin, int *xboxmax,
                int *yboxmin, int *yboxmax, int *zboxmin, int *zboxmax, int *NSpax, int *dcat, int *LabelToId, int *category) {

    // Init
    int DimX = shape[0];
    int DimY = shape[1];
    int DimZ = shape[2];

    long i,j,k;
    long idx;
    int id;
    int this_label;

    // # Fill !..find bounding boxes and centroid for each objects
    for (i = 0; i<DimX; i++)
        for (j = 0; j<DimY; j++)
            for (k = 0; k<DimZ; k++) {
                idx = convert_indices(i,j,k, DimY, DimZ);
                this_label = mask[idx];
                if (this_label != 0) {
                    id = LabelToId[this_label]; //  #!..get object associated with pixel  (0-based)
                    if (id != -1) {
                        xcen[id] += i - 0.5;
                        ycen[id] += j - 0.5;
                        zcen[id] += k - 0.5;

                        xboxmin[id] = fmin(xboxmin[id], i);
                        yboxmin[id] = fmin(yboxmin[id], j);
                        zboxmin[id] = fmin(zboxmin[id], k);

                        xboxmax[id] = fmax(xboxmax[id], i);
                        yboxmax[id] = fmax(yboxmax[id], j);
                        zboxmax[id] = fmax(zboxmax[id], k);

                        dcat[id] = category[this_label];
                    } else { // # Cleanup mask
                        mask[idx] = 0;
                    }
                }
            }


    // # !..finalize geometrical centroid calculation
    for (i = 0; i<ndet; i++) {
        xcen[i] = xcen[i] / NSpax[i];
        ycen[i] = ycen[i] / NSpax[i];
        zcen[i] = zcen[i] / NSpax[i];
    }


}

void max_areas(int *mask, int *areas, int max_label, int *shape) {

    // Init
    int DimX = shape[0];
    int DimY = shape[1];
    int DimZ = shape[2];

    long i,j,k, ii;
    long idx;

    int sub_areas[max_label+1];

    // # Fill !..find bounding boxes and centroid for each objects
    for (k = 0; k<DimZ; k++) {
        for (i = 0; i<DimX; i++)
            for (j = 0; j<DimY; j++) {
                if (i == 0 && j == 0) {
                    // Reset to 0
                    for (ii = 0; ii <= max_label; ii++)
                        sub_areas[ii] = 0;
                }
                idx = convert_indices(i,j,k, DimY, DimZ);
                if (mask[idx] == 0)
                    continue;
                // Increment
                sub_areas[mask[idx]]++;
            }
        // Update
        for (ii = 0; ii <= max_label; ii++)
            areas[ii] = fmax(areas[ii], sub_areas[ii]);
    }
}