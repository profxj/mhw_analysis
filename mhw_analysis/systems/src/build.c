/*
Support algorithms for bspline.
*/

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#include "bspline.h"


int* first_pass(bool *cube, int *shape, bool upper_left) {
    /*
    Compute the indices in a flattened 2D array that contain the
    upper triangle.

    To get the indices for the *upper-left* triangle, set
    upper_left=true.

    For example, the upper-right triangle of a 3x3 matrix has
    flattened indices of: 0, 1, 2, 4, 5, 8. The upper-left indices
    are: 0, 1, 2, 3, 4, 6.

    Args:
        *cube:
            Size of the square array.
        upper_left:
            Return the upper-left triangle instead of the
            upper-right. The upper triangle is typically defined as
            the upper-right triangle.

    Returns:
        Returns the pointer to the integer array with the indices.
    */

    // Init
    int label = 0;
    // Should probably allocate this outside the routine
    int (*mask)[shape[1]][shape[2]] = malloc( sizeof(int[shape[0]][shape[1]][shape[2]]));

    int DimX = shape[0];
    int DimY = shape[1];
    int DimZ = shape[2];

    int i,j,k;
    int ii,jj,kk;
    int count;

    bool all_zero;

    for (i = 0; i<DimX; i++)
        for (j = 0; i<DimY; j++)
            for (k = 0; i<DimZ; k++)
                mask[i][j][k] = 0;

    // Labels
    int prior_labels[27];
    int maxnlabels = 10000000;
    int minlabel;
    int *parent = (int*) malloc (maxnlabels * sizeof(int));
    int *NSpax = (int*) malloc (maxnlabels * sizeof(int));

    // Loop me!
    for (i = 1; i<DimX-1; i++)
        for (j = 1; i<DimY-1; j++)
            for (k = 1; i<DimZ-1; k++) {

                if (cube[i][j][k] == false)
                    continue;

                // prior_labels = mask[i-1:i+2,j-1:j+2,k-1:k+2].flatten()
                // Fill prior_labels and check for non-zero values
                // Also determine the minimum value of the label
                count = 0;
                all_zero = true;
                minlabel = 0;
                for (ii=i-1; ii<=i+1, ii++)
                    for (jj=j-1; jj<=j+1, jj++)
                        for (kk=k-1; kk<=k+1, kk++) {
                            prior_labels[count] = mask[ii,jj,kk];
                            if (prior_labels[count] != 0) {
                                all_zero = false;
                                if (prior_labels[count] > minlabel)
                                    minlabel = prior_labels[count];
                            }
                            count++;
                        }

                if (all_zero) {
                    label=label+1;
                    if (label > maxnlabels) // # STOP "Increase stack size (maxnlabels)!"
                        printf("Exception happened here\n")
                    mask[i][j][k] = label
                } else { // # !..this spaxel is connected to another one
                    this_label = minlabel; // np.min(prior_labels[prior_labels > 0])
                    mask[i][j][k] = this_label;
                    // #!..update parent tree
                    for (ii=0; ii<27; ii++) { // p in range(prior_labels.size):
                        if (prior_labels[p] != 0) && (prior_labels[p] != this_label)
                            union(parent, this_label, prior_labels[p])
                }

    /* Return */
    return mask;
}

void union(int *parent, int x, int y):
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




