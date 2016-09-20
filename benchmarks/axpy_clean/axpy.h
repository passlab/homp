#ifndef __AXPY_H__
#define __AXPY_H__

/* change this to do saxpy or daxpy : single precision or double precision*/
#define REAL float
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include "homp.h"

#ifdef __cplusplus
extern "C" {
#endif

/* both the omp version and ompacc version */
extern void axpy(REAL* x, REAL* y,  long n, REAL a); 

/* the driver into the runtime for offloading */
extern double axpy_ompacc_mdev(int ndevs, int *targets, REAL *x, REAL *y, long n, REAL a);

/* each function is the dev-specific version of the axpy kernel code */
extern void axpy_cpu_omp_wrapper(omp_offloading_t *off, long start_n,  long length_n,REAL a,REAL *x,REAL *y);
extern void axpy_nvgpu_cuda_wrapper(omp_offloading_t *off, long start_n,  long length_n,REAL a,REAL *x,REAL *y);
extern void axpy_nvgpu_opencl_wrapper(omp_offloading_t *off, long start_n,  long length_n,REAL a,REAL *x,REAL *y);
extern void axpy_itlmic_wrapper(omp_offloading_t *off, long start_n,  long length_n,REAL a,REAL *x,REAL *y);

#ifdef __cplusplus
 }
#endif

#endif
