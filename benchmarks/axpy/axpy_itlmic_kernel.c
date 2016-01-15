//
// Created by Yonghong Yan on 1/7/16.
//
#ifdef __cplusplus
extern "C" {
#endif

#include "axpy.h"
#include <offload.h>
#include <homp.h>

#ifdef USE_INTEL_MKL
#include <mkl.h>
#endif

void axpy_itlmic_wrapper(omp_offloading_t *off, long start_n,  long length_n,REAL a,REAL *x,REAL *y) {
//    omp_event_t *events = off->events;
//    omp_dev_stream_t *stream = off->stream;
//    omp_offloading_info_t * off_info = off->off_info;
    int sysid = off->dev->sysid;
    int i;

//    printf("x: %X, y: %X: %d\n", x, y, (length_n - start_n)*sizeof(REAL));

#ifndef ITLMIC_COMBINED_OFFLOADING
    #pragma offload target(mic:sysid) in (x: length(0) alloc_if(0) free_if(0)) \
                                in (y: length(0) alloc_if(0) free_if(0))
#else
#pragma offload target(mic:sysid) in (x: length(length_n) align(64))  \
                                inout (y: length(length_n) align(64))
#endif
    {
        //#pragma omp parallel for simd
        //#pragma omp simd
#ifdef USE_INTEL_MKL
     cblas_saxpy(length_n, a, x, 1, y, 1);
#else
        for (i = 0; i < length_n-start_n; i++) {
            y[i] = x[i] * a + y[i];
        }
#endif
    }

//    printf("x: %X, y: %X: %d\n", x, y, (length_n - start_n)*sizeof(REAL));
#if 0
    double alloc_time = omp_get_wtime();
#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_start(&events[acc_mapto_event_index], stream, "ACC_MAPTO", "Accumulated time for mapto data movement for all array");
#endif

#pragma offload target(mic:sysid) in (x: length(length_n-start_n) alloc_if(1) free_if(0)) \
                                in (y: length(length_n-start_n) alloc_if(1) free_if(0))
    {
    }

#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_stop(&events[acc_mapto_event_index]);
    omp_event_record_start(&events[acc_kernel_exe_event_index], stream, "KERN", "Time for kernel (%s) execution", off_info->name);
#endif
    alloc_time = omp_get_wtime() - alloc_time;

    double kernel_time = omp_get_wtime();
#pragma offload target(mic:sysid) nocopy (x: length(length_n-start_n) alloc_if(0) free_if(0)) \
                                nocopy (y: length(length_n-start_n) alloc_if(0) free_if(0))
    #pragma omp parallel for simd num_threads(240)
        for (i = 0; i < length_n-start_n; i++) {
            y[i] = x[i] * a + y[i];
        }
#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_stop(&events[acc_kernel_exe_event_index]);
    omp_event_record_start(&events[acc_mapfrom_event_index], stream,  "ACC_MAPFROM", "Accumulated time for mapfrom data movement for all array");
#endif
    kernel_time = omp_get_wtime() - kernel_time;

    double free_time = omp_get_wtime();

#pragma offload target(mic:sysid) nocopy (x: length(length_n-start_n) alloc_if(0) free_if(1)) \
                                out (y: length(length_n-start_n) alloc_if(0) free_if(1))
    {
    }
#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_stop(&events[acc_mapfrom_event_index]);
#endif
    free_time = omp_get_wtime() - free_time;
#endif

//    double walltime = omp_get_wtime() - start_timer;

//    printf("PASS axpy\n\n");
//    printf("Alloc time = %.2f sec\n\n", alloc_time);
//    printf("Kernel time = %.2f sec\n\n", kernel_time);
//    printf("Free time = %.2f sec\n\n", free_time);
//    printf("Total time = %.2f sec\n\n", walltime);

}
#ifdef __cplusplus
}
#endif
