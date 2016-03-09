//
// Created by Yonghong Yan on 1/7/16.
//
#include "sum.h"
#include <homp.h>
#ifdef USE_INTEL_MKL
#include <mkl.h>
#endif

void sum_cpu_omp_wrapper(omp_offloading_t *off, long start_n, long length_n,  REAL *x, REAL *y, REAL *result) {
    int num_omp_threads = off->dev->num_cores;
    long i;

    REAL sum = 0.0;
    #pragma omp parallel for simd shared(y, x, start_n, length_n) private(i) num_threads(num_omp_threads) reduction(+:sum)
    for (i=start_n; i<start_n + length_n; i++) {
        sum += y[i] * x[i];
//			printf("x[%d]: %f, y[%d]: %f\n", i, x[i], i, y[i]);
    }
    *result = sum;
}
