#include <offload.h>
#include <omp.h>
#define REAL float

void jacobi_itlmic_wrapper(long n,long m,REAL *u,REAL *uold,long uold_m, int uold_0_offset, int uold_1_offset)
{
    long i, j;
    double start_timer = omp_get_wtime();

    double alloc_time = omp_get_wtime();
#pragma offload target(mic) in (u: length(n*m) alloc_if(1) free_if(0)) \
                            in (uold: length(n*m) alloc_if(1) free_if(0))
    {
    }
    alloc_time = omp_get_wtime() - alloc_time;

    double kernel_time = omp_get_wtime();
#pragma offload target(mic) nocopy (u: length(n*m) alloc_if(0) free_if(0)) \
                            nocopy (uold: length(n*m) alloc_if(0) free_if(0))
#pragma omp parallel for simd
    {
//#pragma omp parallel for private(j,i) shared(m,n,uold,u,uold_0_offset,uold_1_offset)
        for (i=0; i < n; i++)
            for (j=0; j < m; j++) {
                /* since uold has halo region, here we need to adjust index to reflect the new offset */
                uold[i+uold_0_offset][j+uold_1_offset] = u[i][j];
            }
    }

    kernel_time = omp_get_wtime() - kernel_time;

    double free_time = omp_get_wtime();
#pragma offload target(mic) nocopy (u: length(n*m) alloc_if(0) free_if(1)) \
                            nocopy (uold: length(n*m) alloc_if(0) free_if(1))
    {
    }
    free_time = omp_get_wtime() - free_time;

    double walltime = omp_get_wtime() - start_timer;
}