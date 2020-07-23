
#ifndef _BUILD_H_
#define _BUILD_H_

#include <stdbool.h>

void first_pass(bool *cube, int *mask, int *shape, int *parent);
void lunion(int *parent, int x, int y);
int convert_indices(int i, int j, int k, int DimY, int DimZ);

#endif // _BUILD_H_

