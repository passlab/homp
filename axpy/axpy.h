#ifndef __AXPY_H__
#define __AXPY_H__

/* change this to do saxpy or daxpy : single precision or double precision*/
#define REAL double
#include <omp.h>
#include "homp.h"

#ifdef __cplusplus
extern "C" {
#endif

/* both the omp version and ompacc version */
extern void axpy_omp(REAL* x, REAL* y,  long n, REAL a); 
extern void axpy_ompacc(REAL* x, REAL* y,  long n, REAL a); 
extern void axpy_ompacc_mdev_1(REAL* x, REAL* y,  long n, REAL a);
extern void axpy_ompacc_mdev_v2(REAL* x, REAL* y,  long n, REAL a);
extern double read_timer(); /* in second */
extern double read_timer_ms(); /* in ms */
#ifdef __cplusplus
 }
#endif

#endif
