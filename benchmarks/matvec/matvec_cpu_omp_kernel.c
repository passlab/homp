#include "matvec.h"
#include <homp.h>

void matvec_cpu_omp_wrapper(omp_offloading_t *off, long n, long start_n, long length_n,REAL *a,REAL *x,REAL *y)
{
    int num_omp_threads = off->dev->num_cores;
    int i, j;
#pragma omp parallel for shared(y, x, a, start_n, length_n) private(i,j) num_threads(num_omp_threads)
    for (i = start_n; i < start_n + length_n; i++) {
        for (j = 0; j < n; j++)
            y[i] += a[i*n + j] * x[j];
        //printf ("error part!!");
    }
}
