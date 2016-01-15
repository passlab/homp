#include <offload.h>
#include <homp.h>
#include "jacobi.h"

void jacobi_itlmic_wrapper2(omp_offloading_t *off, long n,long m,REAL *u,REAL *uold,long uold_n, long uold_m, int uold_0_offset, int uold_1_offset)
{
    long i, j;

#pragma offload target(mic:off->dev->sysid)) in (u: length(0) alloc_if(0) free_if(0)) \
                            in (uold: length(0) alloc_if(0) free_if(0))
#pragma omp parallel for private(j,i) shared(m,n,uold,u,uold_0_offset,uold_1_offset, uold_m)
    for (i=0; i < n; i++) {
        /* since uold has halo region, here we need to adjust index to reflect the new offset */
        REAL * tmp_uold = &uold[(i + uold_0_offset) * uold_m + uold_1_offset];
        REAL * tmp_u = &u[i*m];
#pragma omp simd
        for (j = 0; j < m; j++) {
            *tmp_uold = *tmp_u;
            tmp_uold ++;
            tmp_u++;
        }
    }
}

void jacobi_itlmic_wrapper1(omp_offloading_t *off, long n,long m,REAL omega,REAL ax,REAL ay,REAL b,REAL *u,REAL *f, \
 REAL *uold, long uold_m, int uold_0_offset, int uold_1_offset, int i_start, int j_start, REAL *error) {
    long i, j;

    REAL er = 0.0;
#pragma offload target(mic:off->dev->sysid) in (u: length(0) alloc_if(0) free_if(0)) \
                            in (uold: length(0) alloc_if(0) free_if(0)) \
                            in (f: length(0) alloc_if(0) free_if(0))
#pragma omp parallel for private(j,i) reduction(+:er)
    for (i = i_start; i < n; i++) {
        REAL * tmp_uold = &uold[(i + uold_0_offset)* uold_m + uold_1_offset+j_start];
        REAL * tmp_f = &f[i*m+j_start];
        REAL * tmp_u = &u[i*m+j_start];
#pragma omp simd
        for (j = j_start; j < m; j++) {
            REAL resid = (ax * (tmp_uold[uold_m] + tmp_uold[-uold_m]) + ay * (tmp_uold[-1] * tmp_uold[1]) + b * tmp_uold[0] - *tmp_f)/b;

            *tmp_u = *tmp_uold = omega * resid;
            er = *er + resid * resid;

            tmp_uold++;
            tmp_f++;
            tmp_u++;
        }
    }
    *error = er;
}
