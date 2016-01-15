#include "matvec.h"
#include <offload.h>
#include <homp.h>
#include <mkl.h>
void matvec_itlmic_wrapper(omp_offloading_t *off, long n, long start_n, long length_n,REAL *a,REAL *x,REAL *y)
{
    int i, j;
    int sysid = off->dev->sysid;

#ifndef ITLMIC_COMBINED_OFFLOADING
#pragma offload target(mic:sysid) in (a: length(0) alloc_if(0) free_if(0)) \
                                in (x: length(0) alloc_if(0) free_if(0)) \
                                in (y: length(0) alloc_if(0) free_if(0))
#else
#pragma offload target(mic:sysid) in (a: length(length_n*n)) \
                                  in (x: length(length_n))  \
                                inout (y: length(length_n))
#endif
    {
        //#pragma omp parallel for simd
        //#pragma omp simd
#ifdef USE_INTEL_MKL
     REAL alpha = 1;
     REAL beta = 0;

     cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                i, 1, j, alpha, a, j, x, 1, beta, y, 1);
#else
        #pragma omp parallel for shared(y, x, a, start_n, length_n) private(i,j)
        //#pragma omp simd
        for (i = start_n; i < start_n + length_n; i++) {
            for (j = 0; j < n; j++)
                y[i] += a[i*n + j] * x[j];
        }
#endif
    }


#if 0
    omp_dev_stream_t *stream = off->stream;
    omp_offloading_info_t * off_info = off->off_info;
    omp_event_t *events = off->events;

#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_start(&events[acc_mapto_event_index], stream, "ACC_MAPTO", "Accumulated time for mapto data movement for all array");
#endif

#pragma offload target(mic) in (x: length(length_n*sizeof(REAL)) alloc_if(1) free_if(0)) \
                            in (y: length(length_n*sizeof(REAL)) alloc_if(1) free_if(0)) \
                            in (a: length(length_n*n*sizeof(REAL)) alloc_if(1) free_if(0))
    {
    }
#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_stop(&events[acc_mapto_event_index]);
    omp_event_record_start(&events[acc_kernel_exe_event_index], stream, "KERN", "Time for kernel (%s) execution", off_info->name);
#endif

#pragma offload target(mic) in (x: length(0) alloc_if(0) free_if(0)) \
                            in (y: length(0) alloc_if(0) free_if(0)) \
                            in (a: length(0) alloc_if(0) free_if(0))
//#pragma omp parallel for simd
    {
//#pragma omp parallel for shared(y, x, a, start_n, length_n) private(i,j)
        for (i = start_n; i < start_n + length_n; i++) {
            for (j = 0; j < n; j++)
                y[i] += a[i*n + j] * x[j];
            //printf ("error part!!");
        }
    }

#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_stop(&events[acc_kernel_exe_event_index]);
    omp_event_record_start(&events[acc_mapfrom_event_index], stream,  "ACC_MAPFROM", "Accumulated time for mapfrom data movement for all array");
#endif

#pragma offload target(mic) nocopy (x: length(0) alloc_if(0) free_if(1)) \
                            out (y: length(0) alloc_if(0) free_if(1)) \
                            nocopy (a: length(0) alloc_if(0) free_if(1))
    {
    }
#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_stop(&events[acc_mapfrom_event_index]);
#endif
#endif

}
