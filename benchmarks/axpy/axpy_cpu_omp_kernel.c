//
// Created by Yonghong Yan on 1/7/16.
//
#include "axpy.h"
#include <homp.h>

void axpy_cpu_omp_wrapper(omp_offloading_t *off, long start_n,  long length_n,REAL a,REAL *x,REAL *y) {
    int num_omp_threads = off->dev->num_cores;
    int i;
    #pragma omp parallel for shared(y, x, a, start_n, length_n) private(i) num_threads(num_omp_threads)
    for (i=start_n; i<start_n + length_n; i++) {
        y[i] += a*x[i];
//			printf("x[%d]: %f, y[%d]: %f\n", i, x[i], i, y[i]);
    }
}
