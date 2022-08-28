
#ifndef _ANALYSIS_H_
#define _ANALYSIS_H_

#include <stdbool.h>

void spatial_systems(int *mask, int *shape, int *img, int *systems, int n_good, int tot_systems);
void days_in_systems(int *mask, int *shape, int *img, int *systems); 
void days_in_systems_by_year(int *mask, int *shape, int *img, int *systems, int *year); 

#endif // _BUILD_H_

