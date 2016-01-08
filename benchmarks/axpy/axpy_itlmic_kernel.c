//
// Created by Yonghong Yan on 1/7/16.
//
#ifdef __cplusplus
extern "C" {
#endif

#include "axpy.h"
#include <offload.h>
#include <homp.h>

void axpy_itlmic_wrapper(omp_offloading_t *off, long start_n,  long length_n,REAL a,REAL *x,REAL *y) {
    omp_event_t *events = off->events;
    omp_dev_stream_t *stream = off->stream;
    omp_offloading_info_t * off_info = off->off_info;
    int sysid = off->dev->sysid;

    int i;
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
#pragma offload target(mic:sysid) nocopy (x: length(length_n-start_n) alloc_if(0) free_if(0)) \
                                nocopy (y: length(length_n-start_n) alloc_if(0) free_if(0))
    #pragma omp parallel for simd
        for (i = 0; i < length_n-start_n; i++) {
            y[i] = x[i] * a + y[i];
        }
#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_stop(&events[acc_kernel_exe_event_index]);
    omp_event_record_start(&events[acc_mapfrom_event_index], stream,  "ACC_MAPFROM", "Accumulated time for mapfrom data movement for all array");
#endif
#pragma offload target(mic:sysid) nocopy (x: length(length_n-start_n) alloc_if(0) free_if(1)) \
                                out (y: length(length_n-start_n) alloc_if(0) free_if(1))
    {
    }
#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_stop(&events[acc_mapfrom_event_index]);
#endif

}
#ifdef __cplusplus
}
#endif
