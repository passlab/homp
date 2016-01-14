#include <offload.h>
#include <homp.h>
#include "matmul.h"

void matmul_itlmic_wrapper(omp_offloading_t *off, long i, long j,long k,REAL *a,REAL *b,REAL *c)
{
    long ii, jj, kk;

    int sysid = off->dev->sysid;

#pragma offload target(mic:sysid) in (a: length(0) alloc_if(0) free_if(0)) \
                            in (b: length(0) alloc_if(0) free_if(0)) \
                            in (c: length(0) alloc_if(0) free_if(0))
    {
//#pragma omp parallel shared(i, j, k, a, b, c) private(ii, jj, kk)
//#pragma omp for
        for (ii = 0; ii < i; ii++) {
            for (jj = 0; jj < j; jj++) {
                REAL sum = 0.0;
                for (kk = 0; kk < k; kk++) {
                    sum += a[ii * k + kk] * b[kk * j + jj];
                }
                c[ii * j + jj] = sum;
            }
        }
    }

#if 0
#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_start(&events[acc_mapto_event_index], stream, "ACC_MAPTO", "Accumulated time for mapto data movement for all array");
#endif
#pragma offload target(mic) in (a: length(i*k) alloc_if(1) free_if(0)) \
                            in (b: length(k*j) alloc_if(1) free_if(0)) \
                            in (c: length(i*j) alloc_if(1) free_if(0))
    {
    }

#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_stop(&events[acc_mapto_event_index]);
    omp_event_record_start(&events[acc_kernel_exe_event_index], stream, "KERN", "Time for kernel (%s) execution", off_info->name);
#endif

#pragma offload target(mic) nocopy (a: length(i*k) alloc_if(0) free_if(0)) \
                            nocopy (b: length(k*j) alloc_if(0) free_if(0)) \
                            nocopy (c: length(i*j) alloc_if(0) free_if(0))
#pragma omp parallel for simd
    {
        for (ii = 0; ii < i; ii++) {
            for (jj = 0; jj < j; jj++) {
                REAL sum = 0.0;
                for (kk = 0; kk < k; kk++) {
                    sum += a[ii * k + kk] * b[kk * j + jj];
                }
                c[ii * j + jj] = sum;
            }
        }
    }

#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_stop(&events[acc_kernel_exe_event_index]);
    omp_event_record_start(&events[acc_mapfrom_event_index], stream,  "ACC_MAPFROM", "Accumulated time for mapfrom data movement for all array");
#endif

#pragma offload target(mic) nocopy (a: length(i*k) alloc_if(0) free_if(1)) \
                            nocopy (b: length(k*j) alloc_if(0) free_if(1)) \
                            nocopy (c: length(i*j) alloc_if(0) free_if(1))
    {
    }
#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_stop(&events[acc_mapfrom_event_index]);
#endif
#endif

}
