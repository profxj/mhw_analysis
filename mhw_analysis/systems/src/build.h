
#ifndef _BUILD_H_
#define _BUILD_H_

#include <stdbool.h>

int column_to_row_major_index(int k, int nr, int nc);
void flat_row_major_indices(int k, int nr, int nc, int *i, int *j);

#endif // _BSPLINE_H_

