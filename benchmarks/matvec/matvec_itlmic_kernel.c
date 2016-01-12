#include "matvec.h"
#include <offload.h>
#include <homp.h>

void matvec_itlmic_wrapper(omp_offloading_t *off, long n, long start_n, long length_n,REAL *a,REAL *x,REAL *y)
{
    int i, j;

#pragma offload target(mic:off->dev->sysid) in (x: length(0) alloc_if(0) free_if(0)) \
                            in (y: length(0) alloc_if(0) free_if(0)) \
                            in (a: length(0) alloc_if(0) free_if(0))
    {
#pragma omp parallel for simd
//#pragma omp parallel for shared(y, x, a, start_n, length_n) private(i,j)
        for (i = start_n; i < start_n + length_n; i++) {
            for (j = 0; j < n; j++)
                y[i] += a[i*n + j] * x[j];
            //printf ("error part!!");
        }
    }

#if 0
    omp_event_t *events = off->events;

#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_start(&events[acc_mapto_event_index], stream, "ACC_MAPTO", "Accumulated time for mapto data movement for all array");
#endif

#pragma offload target(mic) in (x: length(length_n-start_n) alloc_if(1) free_if(0)) \
                            in (y: length(length_n-start_n) alloc_if(1) free_if(0)) \
                            in (a: length(length_n-start_n) alloc_if(1) free_if(0))
    {
    }
#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_stop(&events[acc_mapto_event_index]);
    omp_event_record_start(&events[acc_kernel_exe_event_index], stream, "KERN", "Time for kernel (%s) execution", off_info->name);
#endif

#pragma offload target(mic) nocopy (x: length(length_n-start_n) alloc_if(0) free_if(0)) \
                            nocopy (y: length(length_n-start_n) alloc_if(0) free_if(0)) \
                            nocopy (a: length(length_n-start_n) alloc_if(0) free_if(0))
#pragma omp parallel for simd
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

#pragma offload target(mic) nocopy (x: length(length_n-start_n) alloc_if(0) free_if(1)) \
                            nocopy (y: length(length_n-start_n) alloc_if(0) free_if(1)) \
                            nocopy (a: length(length_n-start_n) alloc_if(0) free_if(1))
    {
    }
#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_stop(&events[acc_mapfrom_event_index]);
#endif
#endif

}
