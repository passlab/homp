#include "matvec.h"
#include <offload.h>
#include <homp.h>
#ifdef USE_INTEL_MKL
#include <mkl.h>
#endif

void matvec_itlmic_wrapper(omp_offloading_t *off, long n, long start_n, long length_n,REAL *a,REAL *x,REAL *y)
{
    int i, j;
    int sysid = off->dev->sysid;
    int num_cores = off->dev->num_cores;
#ifdef USE_INTEL_MKL
    REAL alpha = 1;
    REAL beta = 0;
    //mkl_mic_enable();
#endif

#ifdef ITLMIC_COMBINED_OFFLOADING
#pragma offload target(mic:sysid) in (a: length((length_n+start_n)*n)) \
                                  in (x: length(n))  \
                                inout (y: length(start_n+length_n))
#else
#pragma offload target(mic:sysid) in (a: length(0) alloc_if(0) free_if(0)) \
                                in (x: length(0) alloc_if(0) free_if(0)) \
                                in (y: length(0) alloc_if(0) free_if(0))
#endif
    {
#ifdef USE_INTEL_MKL
     cblas_sgemv(CblasColMajor, CblasNoTrans, length_n , n, alpha, a, length_n, x, 1, beta, y, 1);

#else
        #pragma omp parallel for simd shared(y, x, a, start_n, length_n) private(i,j) num_threads(num_cores)
        for (i = start_n; i < start_n + length_n; i++) {
            for (j = 0; j < n; j++)
                y[i] += a[i*n + j] * x[j];
        }
#endif
    }
}
