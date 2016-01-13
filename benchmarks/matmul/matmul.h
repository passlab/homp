#ifndef __MATMUL_H__
#define __MATMUL_H__

#define REAL float

#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include "homp.h"

#ifdef __cplusplus
extern "C" {
#endif

extern void matmul_cpu_omp_wrapper(omp_offloading_t *off, long i, long j,long k,float *A,float *B,float *C);
extern void matmul_nvgpu_cuda_wrapper(omp_offloading_t *off, long i, long j,long k,REAL *A,REAL *B,REAL *C);
extern void matmul_itlmic_wrapper(omp_offloading_t *off, long i, long j,long k,REAL *a,REAL *b,REAL *c);

#ifdef __cplusplus
 }
#endif

#endif
