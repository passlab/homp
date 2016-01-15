//
// Created by Yonghong Yan on 1/14/16.
//

#ifndef OFFOMP_JACOBI_H
#define OFFOMP_JACOBI_H

#define REAL float

extern void jacobi_cpu_omp_wrapper1(omp_offloading_t *off, long n,long m,REAL omega,REAL ax,REAL ay,REAL b,REAL *_dev_u,REAL *_dev_f, \
 REAL *_dev_uold, long uold_m, int uold_0_offset, int uold_1_offset, int start_i, int start_j, REAL *error);
extern void jacobi_cpu_omp_wrapper2(omp_offloading_t *off, long n,long m,REAL *u,REAL *uold,long uold_n, long uold_m, int uold_0_offset, int uold_1_offset);

extern void jacobi_nvgpu_cuda_wrapper1(omp_offloading_t *off, long n,long m,REAL omega,REAL ax,REAL ay,REAL b,REAL *_dev_u,REAL *_dev_f, \
 REAL *_dev_uold, long uold_m, int uold_0_offset, int uold_1_offset, int start_i, int start_j, REAL *error);
extern void jacobi_nvgpu_cuda_wrapper2(omp_offloading_t *off, long n,long m,REAL *u,REAL *uold,long uold_n, long uold_m, int uold_0_offset, int uold_1_offset);

extern void jacobi_itlmic_wrapper1(omp_offloading_t *off, long n,long m,REAL omega,REAL ax,REAL ay,REAL b,REAL *u,REAL *f, \
 REAL *uold, long uold_m, int uold_0_offset, int uold_1_offset, int i_start, int j_start, REAL *error);
extern void jacobi_itlmic_wrapper2(omp_offloading_t *off, long n,long m,REAL *u,REAL *uold,long uold_n, long uold_m, int uold_0_offset, int uold_1_offset);

#endif //OFFOMP_JACOBI_H
