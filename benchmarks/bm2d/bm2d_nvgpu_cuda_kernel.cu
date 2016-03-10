#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include "homp.h"
#include "bm2d.h"
#include "xomp_cuda_lib_inlined.cu"
#include <curand.h>
#include <curand_kernel.h>

/* this works only for 1-d row-wise partition */
__global__ void bm2d_nvgpu_kernel(int start_n, int len_n, long n, long m, int u_dimX, int u_dimY, REAL *u, REAL *uold, int maxwin, int coeff_dimX, REAL *coeff) {
    long ix, iy, ir;
    long ixy;
    long ixy_lower, ixy_upper;
    curandState_t state;

    /* we have to initialize the state */
    curand_init(0, /* the seed controls the sequence of random values that are produced */
                  0, /* the sequence number is only important with multiple cores */
                  0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                  &state);

    // variables for adjusted loop info considering both original chunk size and step(strip)
    long _dev_loop_chunk_size;
    long _dev_loop_sched_index;
    long _dev_loop_stride;

    // 1-D thread block:
    long _dev_thread_num = gridDim.x * blockDim.x;
    long _dev_thread_id = blockDim.x * blockIdx.x + threadIdx.x;

    //TODO: adjust bound to be inclusive later
    long orig_start = 0;
    long orig_end = len_n * m; /* Linearized iteration space */
    long orig_step = 1;
    long orig_chunk_size = 1;

    XOMP_static_sched_init(orig_start, orig_end, orig_step, orig_chunk_size, _dev_thread_num, _dev_thread_id, \
      &_dev_loop_chunk_size, &_dev_loop_sched_index, &_dev_loop_stride);

    //XOMP_accelerator_loop_default (1, (n-1)*(m-1)-1, 1, &_dev_lower, &_dev_upper);
    while (XOMP_static_sched_next(&_dev_loop_sched_index, orig_end, orig_step, _dev_loop_stride, _dev_loop_chunk_size,
                                  _dev_thread_num, _dev_thread_id, &ixy_lower, &ixy_upper)) {
        for (ixy = ixy_lower; ixy <= ixy_upper; ixy++) {
            ix = ixy / m;
            iy = ixy % m;
            if (!(ix>=start_n && ix<=start_n+len_n-1 && iy>=0 && iy<=m-1)) continue;
            int radius = curand_uniform_double(&state) * maxwin;
            if (radius < 1) continue;

            int count = 4*radius+1;
#ifdef SQUARE_SETNCIL
	        count = coeff_dimX * coeff_dimX;
#endif
            long offset = (ix+radius)*u_dimY+radius+iy;
            REAL *temp_u = &u[offset];
            REAL *temp_uold = &uold[offset];
            REAL result = temp_uold[0] * coeff[0];
            /* 2/4 way loop unrolling */
            for (ir = 1; ir <= radius; ir++) {
                result += coeff[ir] * temp_uold[ir];                //horizontal right
                result += coeff[-ir] * temp_uold[-ir];                  // horizontal left
                result += coeff[-ir * coeff_dimX] * temp_uold[-ir * u_dimY]; //vertical up
                result += coeff[ir * coeff_dimX] * temp_uold[ir * u_dimY]; // vertical bottom
#ifdef SQUARE_SETNCIL
				result += coeff[-ir*coeff_dimX-ir] * temp_uold[-ir * u_dimY-ir] // left upper corner
				result += coeff[-ir*coeff_dimX+ir] * temp_uold[-ir * u_dimY+ir] // right upper corner
				result += coeff[ir*coeff_dimX-ir] * temp_uold[ir * u_dimY]-ir] // left bottom corner
				result += coeff[ir*coeff_dimX+ir] * temp_uold[ir * u_dimY]+ir] // right bottom corner
#endif
            }
            *temp_u = result/count;
        }
    }
}

void bm2d_nvgpu_cuda_wrapper(omp_offloading_t *off, int start, int len, long n, long m, int u_dimX, int u_dimY, REAL *u, REAL *uold, int maxwin, int coeff_dimX, REAL *coeff) {
    dim3 threads_per_team(16, 16);
    dim3 teams_per_league((len + threads_per_team.x - 1) / threads_per_team.x,
                          (m + threads_per_team.y - 1) / threads_per_team.y); /* we assume dividable */
    bm2d_nvgpu_kernel<<<teams_per_league, threads_per_team, 0, off->stream->systream.cudaStream>>>
                                                                       (start, len, n, m, u_dimX, u_dimY, u, uold, maxwin, coeff_dimX, coeff);
}


