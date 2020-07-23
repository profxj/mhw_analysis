
#ifndef _BUILD_H_
#define _BUILD_H_

#include <stdbool.h>

void first_pass(bool *cube, int *mask, int *shape, int *parent);
void second_pass(int *mask, int *parent, int *shape, int *NSpax);
void final_pass(int ndet, int *mask, int *shape, float *xcen, float *ycen, float *zcen, int *xboxmin, int *xboxmax,
                int *yboxmin, int *yboxmax, int *zboxmin, int *zboxmax, int *NSpax, int *LabelToId);
void lunion(int *parent, int x, int y);
int convert_indices(int i, int j, int k, int DimY, int DimZ);

#endif // _BUILD_H_

