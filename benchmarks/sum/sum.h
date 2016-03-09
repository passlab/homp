#ifndef __AXPY_H__
#define __AXPY_H__

/* change this to do ssum or dsum : single precision or double precision*/
#define REAL double
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include "homp.h"

#ifdef __cplusplus
extern "C" {
#endif

/* both the omp version and ompacc version */
extern REAL sum(long n, REAL* x, REAL* y);
extern double sum_ompacc_mdev(int ndevs, int *targets, long n, REAL *x, REAL *y, REAL *result);
extern void sum_cpu_omp_wrapper(omp_offloading_t *off, long start_n, long length_n, REAL *x, REAL *y, REAL *result);
extern void sum_nvgpu_cuda_wrapper(omp_offloading_t *off, long start_n, long length_n, REAL *x, REAL *y, REAL *result);
extern void sum_nvgpu_opencl_wrapper(omp_offloading_t *off, long start_n, long length_n, REAL *x, REAL *y, REAL *result);
extern void sum_itlmic_wrapper(omp_offloading_t *off, long start_n, long length_n, REAL *x, REAL *y, REAL *result);
#ifdef __cplusplus
 }
#endif

#endif
