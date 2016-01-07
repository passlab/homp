//
// Created by Yonghong Yan on 1/7/16.
//
#include "axpy.h"
#include <offload.h>
#include <omp.h>

void axpy_itlmic_wrapper( long start_n,  long length_n,REAL a,REAL *x,REAL *y) {
    double start_timer = omp_get_wtime();

    double alloc_time = omp_get_wtime();
#pragma offload target(mic) in (x: length(length_n-start_n) alloc_if(1) free_if(0)) \
                                in (y: length(length_n-start_n) alloc_if(1) free_if(0))
    {
    }
    alloc_time = omp_get_wtime() - alloc_time;

    double kernel_time = omp_get_wtime();
#pragma offload target(mic) nocopy (x: length(length_n-start_n) alloc_if(0) free_if(0)) \
                                nocopy (y: length(length_n-start_n) alloc_if(0) free_if(0))
    #pragma omp parallel for simd
        for (i = 0; i < length_n-start_n; i++) {
            x[i] = x[i] * a + y[i];
        }
    kernel_time = omp_get_wtime() - kernel_time;

    double free_time = omp_get_wtime();
#pragma offload target(mic) nocopy (x: length(length_n-start_n) alloc_if(0) free_if(1)) \
                                nocopy (y: length(length_n-start_n) alloc_if(0) free_if(1))
    {
    }
    free_time = omp_get_wtime() - free_time;

    double walltime = omp_get_wtime() - start_timer;
}