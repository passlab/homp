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
extern double axpy_ompacc_mdev(REAL* x, REAL* y,  long n, REAL a);
extern void axpy_itlmic_wrapper(omp_offloading_t *off, long start_n,  long length_n,REAL a,REAL *x,REAL *y);
#ifdef __cplusplus
 }
#endif

#endif
