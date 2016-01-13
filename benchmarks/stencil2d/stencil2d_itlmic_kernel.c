#include "stencil2d.h"
#include <offload.h>
#include <homp.h>

void stencil2d_itlmic_wrapper(omp_offloading_t *off, int start, int len, long n, long m, int u_dimX, int u_dimY, REAL *u, REAL *uold, int radius,int coeff_dimX, REAL *coeff)
{
    int ix, iy, ir;

#pragma offload target(mic:off->dev->sysid) in (u: length(0) alloc_if(0) free_if(0)) \
                            in (uold: length(0) alloc_if(0) free_if(0)) \
                            in (coeff: length(0) alloc_if(0) free_if(0))
//#pragma omp parallel for simd
    {
        for (ix = start; ix < start+len; ix++) {
            REAL * temp_u = &u[(ix+radius)*u_dimY+radius];
            REAL * temp_uold = &uold[(ix+radius)*u_dimY+radius];
            for (iy = 0; iy < m; iy++) {
//                    if (off->devseqid == 0)printf("dev: %d, [%d][%d]:%f\n", off->devseqid, ix, iy, temp_u[0]);
                REAL result = temp_uold[0] * coeff[0];
                /* 2/4 way loop unrolling */
                for (ir = 1; ir <= radius; ir++) {
                    result += coeff[ir] * temp_uold[ir];           		//horizontal right
                    result += coeff[-ir]* temp_uold[-ir];                  // horizontal left
                    result += coeff[-ir*coeff_dimX] * temp_uold[-ir * u_dimY]; //vertical up
                    result += coeff[ir*coeff_dimX] * temp_uold[ir * u_dimY]; // vertical bottom
#ifdef SQUARE_SETNCIL
						result += coeff[-ir*coeff_dimX-ir] * temp_uold[-ir * u_dimY-ir] // left upper corner
						result += coeff[-ir*coeff_dimX+ir] * temp_uold[-ir * u_dimY+ir] // right upper corner
						result += coeff[ir*coeff_dimX-ir] * temp_uold[ir * u_dimY]-ir] // left bottom corner
						result += coeff[ir*coeff_dimX+ir] * temp_uold[ir * u_dimY]+ir] // right bottom corner
#endif
                }
                *temp_u = result/count;
                temp_u++;
                temp_uold++;
            }
        }
    }

#if 0
#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_start(&events[acc_mapto_event_index], stream, "ACC_MAPTO", "Accumulated time for mapto data movement for all array");
#endif

#pragma offload target(mic) in (u: length(u_dimX*u_dimY) alloc_if(1) free_if(0)) \
                            in (uold: length(u_dimX*u_dimY) alloc_if(1) free_if(0)) \
                            in (coeff: length(radius*radius) alloc_if(1) free_if(0))
    {
    }
#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_stop(&events[acc_mapto_event_index]);
    omp_event_record_start(&events[acc_kernel_exe_event_index], stream, "KERN", "Time for kernel (%s) execution", off_info->name);
#endif

#pragma offload target(mic) in (u: length(0) alloc_if(0) free_if(0)) \
                            in (uold: length(0) alloc_if(0) free_if(0)) \
                            in (coeff: length(0) alloc_if(0) free_if(0))
#pragma omp parallel for simd
    {
        for (ix = start; ix < start+len; ix++) {
            REAL * temp_u = &u[(ix+radius)*u_dimY+radius];
            REAL * temp_uold = &uold[(ix+radius)*u_dimY+radius];
            for (iy = 0; iy < m; iy++) {
//                    if (off->devseqid == 0)printf("dev: %d, [%d][%d]:%f\n", off->devseqid, ix, iy, temp_u[0]);
                REAL result = temp_uold[0] * coeff[0];
                /* 2/4 way loop unrolling */
                for (ir = 1; ir <= radius; ir++) {
                    result += coeff[ir] * temp_uold[ir];           		//horizontal right
                    result += coeff[-ir]* temp_uold[-ir];                  // horizontal left
                    result += coeff[-ir*coeff_dimX] * temp_uold[-ir * u_dimY]; //vertical up
                    result += coeff[ir*coeff_dimX] * temp_uold[ir * u_dimY]; // vertical bottom
#ifdef SQUARE_SETNCIL
						result += coeff[-ir*coeff_dimX-ir] * temp_uold[-ir * u_dimY-ir] // left upper corner
						result += coeff[-ir*coeff_dimX+ir] * temp_uold[-ir * u_dimY+ir] // right upper corner
						result += coeff[ir*coeff_dimX-ir] * temp_uold[ir * u_dimY]-ir] // left bottom corner
						result += coeff[ir*coeff_dimX+ir] * temp_uold[ir * u_dimY]+ir] // right bottom corner
#endif
                }
                *temp_u = result/count;
                temp_u++;
                temp_uold++;
            }
        }
    }

#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_stop(&events[acc_kernel_exe_event_index]);
    omp_event_record_start(&events[acc_mapfrom_event_index], stream,  "ACC_MAPFROM", "Accumulated time for mapfrom data movement for all array");
#endif

#pragma offload target(mic) out (u: length(u_dimX*u_dimY) alloc_if(0) free_if(1)) \
                            nocopy (uold: length(u_dimX*u_dimY) alloc_if(0) free_if(1)) \
                            nocopy (coeff: length(radius*radius) alloc_if(0) free_if(1))
    {
    }
#if defined (OMP_BREAKDOWN_TIMING)
    omp_event_record_stop(&events[acc_mapfrom_event_index]);
#endif
#endif
}