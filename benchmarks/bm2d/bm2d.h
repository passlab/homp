//
// Created by Yonghong Yan on 4/26/15.
//
#ifndef OFFOMP_STENCIL2D_H
#define OFFOMP_STENCIL2D_H
#ifdef __cplusplus
extern "C" {
#endif
// flexible between REAL and double
#define REAL float
#include "homp.h"

/* this should go to a compiler generated header file and to be included in the souce files (launcher, off pragma call, etc)*/
struct bm2d_off_args {
    long n; long m; REAL *u; int maxwin; REAL *coeff; int num_its;

    long u_dimX;
    long u_dimY;
    int coeff_dimX;
    REAL * coeff_center;
    REAL * uold;

    int it_num; /* which iteration we are in, used to switch u and uold */
};

extern int dist_dim;
extern int dist_policy;
extern double bm2d_omp_mdev_iterate(int ndevs, int *targets, long n, long m, REAL *u, int maxwin, REAL *coeff, int num_its);
extern double bm2d_omp_mdev(int ndevs, int *targets, long n, long m, REAL *u, int maxwin, REAL *coeff, int num_its);

extern void bm2d_cpu_omp_wrapper(omp_offloading_t *off, long start_n, long len_n, long n, long m, long u_dimX,
                                 long u_dimY, REAL *u, REAL *uold, int maxwin, int coeff_dimX, REAL *coeff);
extern void bm2d_nvgpu_cuda_wrapper(omp_offloading_t *off, long start_n, long len_n, long n, long m, long u_dimX,
                                    long u_dimY, REAL *u, REAL *uold, int maxwin, int coeff_dimX, REAL *coeff);
extern void bm2d_itlmic_wrapper(omp_offloading_t *off, long start, long len, long n, long m, long u_dimX, long u_dimY,
                                REAL *u, REAL *uold, int maxwin, int coeff_dimX, REAL *coeff);

#ifdef __cplusplus
 }
#endif
#endif //OFFOMP_STENCIL2D_H
