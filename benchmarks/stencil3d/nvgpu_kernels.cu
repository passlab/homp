#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include "homp.h"
#include "stencil3d.h"

#include "xomp_cuda_lib_inlined.cu"

//#define LOOP_COLLAPSE 1
#if !LOOP_COLLAPSE

__global__ void stencil3d_nvgpu_kernel(int start_n, int len_n, long n, long m, long k, int u_dimX, int u_dimY, int u_dimZ, REAL *u,
                                       REAL *uold, int radius, int coeff_dimX, REAL *coeff) {
    long ix, iy, iz, ir;
    int count = 6*radius+1;
#ifdef SQUARE_SETNCIL
	count = coeff_dimX * coeff_dimX * coeff_dimX;
#endif
    ix = blockIdx.x * blockDim.x + threadIdx.x;
    iy = blockIdx.y * blockDim.y + threadIdx.y;
    iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix>=start_n&&ix<=len_n&&iy>=0&&iy<=m&&iz>=0&&iz<=k) {
        REAL *temp_u = &u[(ix + radius) * u_dimY *u_dimZ + radius + iy + iz];//added - check
        REAL *temp_uold = &uold[(ix + radius) * u_dimY * u_dimZ + radius + iy + iz];
        REAL result = temp_uold[0] * coeff[0];
        /* 2/4 way loop unrolling */
        for (ir = 1; ir <= radius; ir++) {
            result += coeff[ir] * temp_uold[ir];                //horizontal right
            result += coeff[-ir] * temp_uold[-ir];                  // horizontal left
            result += coeff[-ir * coeff_dimX] * temp_uold[-ir * u_dimY]; //vertical up
            result += coeff[ir * coeff_dimX] * temp_uold[ir * u_dimY]; // vertical bottom
            result += coeff[-ir * coeff_dimX] * temp_uold[-ir * u_dimZ]; //vertical up - z
            result += coeff[ir * coeff_dimX] * temp_uold[ir * u_dimZ]; // vertical bottom - z
#ifdef SQUARE_SETNCIL
		    result += coeff[-ir*coeff_dimX-ir] * temp_uold[-ir * u_dimY-ir] // left upper corner
		    result += coeff[-ir*coeff_dimX+ir] * temp_uold[-ir * u_dimY+ir] // right upper corner
		    result += coeff[ir*coeff_dimX-ir] * temp_uold[ir * u_dimY]-ir] // left bottom corner
		    result += coeff[ir*coeff_dimX+ir] * temp_uold[ir * u_dimY]+ir] // right bottom corner
        result += coeff[ir*coeff_dimX-ir] * temp_uold[ir * u_dimZ]-ir] // left bottom corner - z
		    result += coeff[ir*coeff_dimX+ir] * temp_uold[ir * u_dimZ]+ir] // right bottom corner - z
#endif
        }
        *temp_u = result/count;
    }
}

#else
/* this works only for 1-d row-wise partition */
__global__ void stencil3d_nvgpu_kernel(int start_n, int len_n, long n, long m, long k, int u_dimX, int u_dimY, REAL *u, REAL *uold, int radius, int coeff_dimX, REAL *coeff) {
    long ix, iy, iz, ir;
    long ixy;
    long ixy_lower, ixy_upper;
    int count = 6*radius+1;
#ifdef SQUARE_SETNCIL
	count = coeff_dimX * coeff_dimX * coeff_dimX;
#endif

    // variables for adjusted loop info considering both original chunk size and step(strip)
    long _dev_loop_chunk_size;
    long _dev_loop_sched_index;
    long _dev_loop_stride;

    // 1-D thread block:
    long _dev_thread_num = gridDim.x * blockDim.x * blockDim.x; // added
    long _dev_thread_id = blockDim.x * blockIdx.x * blockIdx.x+ threadIdx.x; // added

    //TODO: adjust bound to be inclusive later
    long orig_start = 0;
    long orig_end = len_n * m *k; /* Linearized iteration space */ //added
    long orig_step = 1;
    long orig_chunk_size = 1;

    XOMP_static_sched_init(orig_start, orig_end, orig_step, orig_chunk_size, _dev_thread_num, _dev_thread_id, \
      &_dev_loop_chunk_size, &_dev_loop_sched_index, &_dev_loop_stride);

    //XOMP_accelerator_loop_default (1, (n-1)*(m-1)-1, 1, &_dev_lower, &_dev_upper);
    while (XOMP_static_sched_next(&_dev_loop_sched_index, orig_end, orig_step, _dev_loop_stride, _dev_loop_chunk_size,
                                  _dev_thread_num, _dev_thread_id, &ixy_lower, &ixy_upper)) {
        for (ixy = ixy_lower; ixy <= ixy_upper; ixy++) {
            ix = (ixy / m)+(ixy / k); //added
            iy = ixy % m;
            iz = ixy % k;
            if (!(ix>=start_n && ix<=start_n+len_n-1 && iy>=0 && iy<=m-1 && iz>=0 && iz<=k-1)) continue;

            REAL *temp_u = &u[(ix + radius) * u_dimY * u_dimZ + radius+iy+iz];
            REAL *temp_uold = &uold[(ix + radius) * u_dimY * u_dimZ + radius+iy+iz];//added-check
            REAL result = temp_uold[0] * coeff[0];
            /* 2/4 way loop unrolling */
            for (ir = 1; ir <= radius; ir++) {
                result += coeff[ir] * temp_uold[ir];                //horizontal right
                result += coeff[-ir] * temp_uold[-ir];                  // horizontal left
                result += coeff[-ir * coeff_dimX] * temp_uold[-ir * u_dimY]; //vertical up
                result += coeff[ir * coeff_dimX] * temp_uold[ir * u_dimY]; // vertical bottom
                result += coeff[-ir * coeff_dimX] * temp_uold[-ir * u_dimZ]; //vertical up - Z
                result += coeff[ir * coeff_dimX] * temp_uold[ir * u_dimZ]; // vertical bottom - Z
#ifdef SQUARE_SETNCIL
				result += coeff[-ir*coeff_dimX-ir] * temp_uold[-ir * u_dimY-ir] // left upper corner
				result += coeff[-ir*coeff_dimX+ir] * temp_uold[-ir * u_dimY+ir] // right upper corner
				result += coeff[ir*coeff_dimX-ir] * temp_uold[ir * u_dimY]-ir] // left bottom corner
				result += coeff[ir*coeff_dimX+ir] * temp_uold[ir * u_dimY]+ir] // right bottom corner
        result += coeff[ir*coeff_dimX-ir] * temp_uold[ir * u_dimZ]-ir] // left bottom corner - z
				result += coeff[ir*coeff_dimX+ir] * temp_uold[ir * u_dimZ]+ir] // right bottom corner - z
#endif
            }
            *temp_u = result/count;
        }
    }
}

#endif /* LOOP_CLAPSE */
