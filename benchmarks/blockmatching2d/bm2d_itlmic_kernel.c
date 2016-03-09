#include "bm2d.h"
#include <offload.h>
#include <homp.h>

void bm2d_itlmic_wrapper(omp_offloading_t *off, int start, int len, long n, long m, int u_dimX, int u_dimY, REAL *u, REAL *uold, int radius,int coeff_dimX, REAL *coeff)
{
    int ix, iy, ir;
    int count = 4*radius+1;
#ifdef SQUARE_SETNCIL
	count = coeff_dimX * coeff_dimX;
#endif

#pragma offload target(mic:off->dev->sysid) in (u: length(0) alloc_if(0) free_if(0)) \
                            in (uold: length(0) alloc_if(0) free_if(0)) \
                            in (coeff: length(0) alloc_if(0) free_if(0))
    {
#pragma omp parallel for
        for (ix = start; ix < start+len; ix++) {
            REAL * temp_u = &u[(ix+radius)*u_dimY+radius];
            REAL * temp_uold = &uold[(ix+radius)*u_dimY+radius];
            #pragma omp simd
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
}
