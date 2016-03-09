//
// Created by Yonghong Yan on 1/7/16.
//
#ifdef __cplusplus
extern "C" {
#endif

#include "sum.h"
#include <offload.h>
#include <homp.h>

#ifdef USE_INTEL_MKL
#include <mkl.h>
#endif

void sum_itlmic_wrapper(omp_offloading_t *off, long start_n, long length_n, REAL *x, REAL *y, REAL * result) {
    int sysid = off->dev->sysid;
    int num_cores = off->dev->num_cores;
    int i;

//    printf("x: %X, y: %X: %d\n", x, y, (length_n - start_n)*sizeof(REAL));

    REAL sum;
#ifndef ITLMIC_COMBINED_OFFLOADING
    #pragma offload target(mic:sysid) in (x: length(0) alloc_if(0) free_if(0)) \
                                in (y: length(0) alloc_if(0) free_if(0)) out(sum)
#else
#pragma offload target(mic:sysid) in (x: length(length_n-start_n) align(64))  \
                               inout (y: length(length_n-start_n) align(64)) out(sum)
#endif
    {
        #pragma omp parallel for simd private(i) num_threads(num_cores) reduction(+:sum)
        for (i = 0; i < length_n-start_n; i++) {
            sum += x[i] * y[i];
        }
    }

    *result = sum;
}
#ifdef __cplusplus
}
#endif
