#include <offload.h>
#include <homp.h>
#define REAL float

void jacobi_itlmic_wrapper2(omp_offloading_t *off, long n,long m,REAL *u,REAL *uold,long uold_m, int uold_0_offset, int uold_1_offset)
{
    long i, j;
#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_start(&events[acc_mapto_event_index], stream, "ACC_MAPTO", "Accumulated time for mapto data movement for all array");
#endif
#pragma offload target(mic) in (u: length(n*m) alloc_if(1) free_if(0)) \
                            in (uold: length(n*m) alloc_if(1) free_if(0))
    {
    }
#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_stop(&events[acc_mapto_event_index]);
    omp_event_record_start(&events[acc_kernel_exe_event_index], stream, "KERN", "Time for kernel (%s) execution", off_info->name);
#endif
#pragma offload target(mic) in (u: length(0) alloc_if(0) free_if(0)) \
                            in (uold: length(0) alloc_if(0) free_if(0))
#pragma omp parallel for simd
    {
//#pragma omp parallel for private(j,i) shared(m,n,uold,u,uold_0_offset,uold_1_offset)
        for (i=0; i < n; i++)
            for (j=0; j < m; j++) {
                /* since uold has halo region, here we need to adjust index to reflect the new offset */
                uold[i+uold_0_offset][j+uold_1_offset] = u[i][j];
            }
    }

#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_stop(&events[acc_kernel_exe_event_index]);
    omp_event_record_start(&events[acc_mapfrom_event_index], stream,  "ACC_MAPFROM", "Accumulated time for mapfrom data movement for all array");
#endif
#pragma offload target(mic) nocopy (u: length(n*m) alloc_if(0) free_if(1)) \
                            out (uold: length(n*m) alloc_if(0) free_if(1))
    {
    }
#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_stop(&events[acc_mapfrom_event_index]);
#endif
}






void jacobi_itlmic_wrapper1(omp_offloading_t *off, long n,long m,REAL omega,REAL ax,REAL ay,REAL b,REAL *u,REAL *f, \
 REAL *uold, long uold_m, int uold_0_offset, int uold_1_offset, int i_start, int j_start, REAL *_dev_per_block_error, REAL resid)
{
    long i, j;
#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_start(&events[acc_mapto_event_index], stream, "ACC_MAPTO", "Accumulated time for mapto data movement for all array");
#endif
#pragma offload target(mic) in (u:    length(n*m) alloc_if(1) free_if(0)) \
                            in (uold: length(n*m) alloc_if(1) free_if(0)) \
                            in (f:    length(n*m) alloc_if(1) free_if(0))
    {
    }
#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_stop(&events[acc_mapto_event_index]);
    omp_event_record_start(&events[acc_kernel_exe_event_index], stream, "KERN", "Time for kernel (%s) execution", off_info->name);
#endif
#pragma offload target(mic) in (u: length(0) alloc_if(0) free_if(0)) \
                            in (uold: length(0) alloc_if(0) free_if(0)) \
                            in (f: length(0) alloc_if(0) free_if(0))
#pragma omp parallel for simd
#pragma omp parallel for private(resid,j,i) reduction(+:error)
    for (i=i_start; i <n; i++) {
        for (j=j_start; j <m; j++) {
            resid = (ax * (uold[i - 1 + uold_0_offset][j + uold_1_offset] + uold[i + 1 + uold_0_offset][j+uold_1_offset]) + ay * (uold[i+uold_0_offset][j - 1+uold_1_offset] + uold[i+uold_0_offset][j + 1+uold_1_offset]) + b * uold[i+uold_0_offset][j+uold_1_offset] - f[i][j]) / b;

            u[i][j] = uold[i+uold_0_offset][j+uold_1_offset] - omega * resid;
            error = error + resid * resid;
        }
    }
#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_stop(&events[acc_kernel_exe_event_index]);
    omp_event_record_start(&events[acc_mapfrom_event_index], stream,  "ACC_MAPFROM", "Accumulated time for mapfrom data movement for all array");
#endif
#pragma offload target(mic) out (u: length(n*m) alloc_if(0) free_if(1)) \
                            nocopy (uold: length(n*m) alloc_if(0) free_if(1)) \
                            nocopy (f: length(n*m) alloc_if(0) free_if(1))
    {
    }
#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_stop(&events[acc_mapfrom_event_index]);
#endif

}