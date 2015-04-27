#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include "homp.h"
#include "stencil2d.h"

#include "xomp_cuda_lib_inlined.cu"

#define LOOP_COLLAPSE 1
#if !LOOP_COLLAPSE
__global__ void OUT__2__10550__(long n,long m,REAL *_dev_u,REAL *_dev_uold,long uold_m, int uold_0_offset, int uold_1_offset)
{
  long _p_j;
  long _dev_i ;
  long _dev_lower, _dev_upper;
  XOMP_accelerator_loop_default (0, n-1, 1, &_dev_lower, &_dev_upper);
  for (_dev_i = _dev_lower ; _dev_i <= _dev_upper; _dev_i++) {
    for (_p_j = 0; _p_j < m; _p_j++)
      _dev_uold[(_dev_i+uold_0_offset) * uold_m + _p_j+uold_1_offset] = _dev_u[_dev_i * m + _p_j];
  }
}

__global__ void OUT__1__10550__(long n,long m,REAL omega,REAL ax,REAL ay,REAL b,REAL *_dev_u,REAL *_dev_f, REAL *_dev_uold,
		long uold_m, int uold_0_offset, int uold_1_offset, int start_i, int start_j, REAL *_dev_per_block_error)

{
	long _p_j;
  REAL _p_error;
  _p_error = 0;
  REAL _p_resid;
  long _dev_lower, _dev_upper;
  XOMP_accelerator_loop_default (start_i, n-1, 1, &_dev_lower, &_dev_upper);
  long _dev_i;
  for (_dev_i = _dev_lower; _dev_i<= _dev_upper; _dev_i ++) {
    for (_p_j = 1; _p_j < (m - 1); _p_j++) { /* this only works for dist=1 partition */
      _p_resid = (((((ax * (_dev_uold[(_dev_i - 1 + uold_0_offset) * uold_m + _p_j+uold_1_offset] + _dev_uold[(_dev_i + 1+uold_0_offset) * uold_m + _p_j+uold_1_offset])) +
    		  (ay * (_dev_uold[(_dev_i+uold_0_offset) * uold_m + (_p_j - 1+uold_1_offset)] + _dev_uold[(_dev_i + uold_0_offset) * uold_m + (_p_j + 1+uold_1_offset)]))) + (b * _dev_uold[(_dev_i + uold_0_offset) * uold_m + _p_j+uold_1_offset])) -
    		  _dev_f[(_dev_i + uold_0_offset) * uold_m + _p_j+uold_1_offset]) / b);
      _dev_u[_dev_i * uold_m + _p_j] = (_dev_uold[(_dev_i + uold_0_offset) * uold_m + _p_j + uold_1_offset] - (omega * _p_resid));
      _p_error = (_p_error + (_p_resid * _p_resid));
    }
  }
  xomp_inner_block_reduction_float(_p_error,_dev_per_block_error,6);
}
#else
__global__ void OUT__2__10550__(long n,long m,REAL *_dev_u,REAL *_dev_uold,long uold_m, int uold_0_offset, int uold_1_offset)
{
    long _p_j;
    long ij;
    long _dev_lower, _dev_upper;

    long _dev_i ;

    // variables for adjusted loop info considering both original chunk size and step(strip)
    long _dev_loop_chunk_size;
    long _dev_loop_sched_index;
    long _dev_loop_stride;

// 1-D thread block:
    long _dev_thread_num = gridDim.x * blockDim.x;
    long _dev_thread_id = blockDim.x * blockIdx.x + threadIdx.x;

    long orig_start =0;
    long orig_end = n*m-1; // inclusive upper bound
    long orig_step = 1;
    long orig_chunk_size = 1;

    XOMP_static_sched_init (orig_start, orig_end, orig_step, orig_chunk_size, _dev_thread_num, _dev_thread_id, \
                         & _dev_loop_chunk_size , & _dev_loop_sched_index, & _dev_loop_stride);

    //XOMP_accelerator_loop_default (1, (n-1)*(m-1)-1, 1, &_dev_lower, &_dev_upper);
    while (XOMP_static_sched_next (&_dev_loop_sched_index, orig_end, orig_step,_dev_loop_stride, _dev_loop_chunk_size, _dev_thread_num, _dev_thread_id, & _dev_lower
            , & _dev_upper))
    {
        for (ij = _dev_lower ; ij <= _dev_upper; ij ++) {
            //  for (_dev_i = _dev_lower ; _dev_i <= _dev_upper; _dev_i++) {
            //    for (_p_j = 0; _p_j < m; _p_j++)
            _dev_i = ij/m;
            _p_j = ij%m;
            _dev_uold[(_dev_i+uold_0_offset) * uold_m + _p_j+uold_1_offset] = _dev_u[_dev_i * m + _p_j];

        }
    }
}

__global__ void OUT__1__10550__(long n,long m,REAL omega,REAL ax,REAL ay,REAL b,REAL *_dev_u,REAL *_dev_f, REAL *_dev_uold,
                                long uold_m, int uold_0_offset, int uold_1_offset, int start_i, int start_j, REAL *_dev_per_block_error)
{
    long _dev_i;
    long ij;
    long _p_j;
    long _dev_lower, _dev_upper;

    REAL _p_error;
    _p_error = 0;
    REAL _p_resid;

    // variables for adjusted loop info considering both original chunk size and step(strip)
    long _dev_loop_chunk_size;
    long _dev_loop_sched_index;
    long _dev_loop_stride;

    // 1-D thread block:
    long _dev_thread_num = gridDim.x * blockDim.x;
    long _dev_thread_id = blockDim.x * blockIdx.x + threadIdx.x;

    //TODO: adjust bound to be inclusive later
    long orig_start =start_i*m;
    long orig_end = (n - start_i)*m-1; /* Linearized iteration space */
    long orig_step = 1;
    long orig_chunk_size = 1;

    XOMP_static_sched_init (orig_start, orig_end, orig_step, orig_chunk_size, _dev_thread_num, _dev_thread_id, \
      & _dev_loop_chunk_size , & _dev_loop_sched_index, & _dev_loop_stride);

    //XOMP_accelerator_loop_default (1, (n-1)*(m-1)-1, 1, &_dev_lower, &_dev_upper);
    while (XOMP_static_sched_next (&_dev_loop_sched_index, orig_end,orig_step, _dev_loop_stride, _dev_loop_chunk_size, _dev_thread_num, _dev_thread_id, & _dev_lower, & _dev_upper))
    {
        for (ij = _dev_lower; ij <= _dev_upper; ij ++) {
            _dev_i = ij/(m-1);
            _p_j = ij%(m-1);

            if (_dev_i>=start_i && _dev_i< (n) && _p_j>=1 && _p_j< (m-1)) // must preserve the original boudary conditions here!!
            {
                _p_resid = (((((ax * (_dev_uold[(_dev_i - 1 + uold_0_offset) * uold_m + _p_j+uold_1_offset] + _dev_uold[(_dev_i + 1+uold_0_offset) * uold_m + _p_j+uold_1_offset])) +
                               (ay * (_dev_uold[(_dev_i+uold_0_offset) * uold_m + (_p_j - 1+uold_1_offset)] + _dev_uold[(_dev_i + uold_0_offset) * uold_m + (_p_j + 1+uold_1_offset)]))) + (b * _dev_uold[(_dev_i + uold_0_offset) * uold_m + _p_j+uold_1_offset])) -
                             _dev_f[(_dev_i + uold_0_offset) * uold_m + _p_j+uold_1_offset]) / b);
                _dev_u[_dev_i * uold_m + _p_j] = (_dev_uold[(_dev_i + uold_0_offset) * uold_m + _p_j + uold_1_offset] - (omega * _p_resid));
                _p_error = (_p_error + (_p_resid * _p_resid));
            }
        }
    }

    xomp_inner_block_reduction_float(_p_error,_dev_per_block_error,6);
}
#endif /* LOOP_CLAPSE */