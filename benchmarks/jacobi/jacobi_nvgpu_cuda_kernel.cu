#include <homp.h>

#if defined (DEVICE_NVGPU_CUDA_SUPPORT)
#include "xomp_cuda_lib_inlined.cu"

#define LOOP_COLLAPSE 1
#if !LOOP_COLLAPSE
__global__ void jacobi_nvgpu_cuda_kernel2(long n,long m,REAL *_dev_u,REAL *_dev_uold,long uold_m, int uold_0_offset, int uold_1_offset)
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

__global__ void jacobi_nvgpu_cuda_kernel1(long n,long m,REAL omega,REAL ax,REAL ay,REAL b,REAL *_dev_u,REAL *_dev_f, REAL *_dev_uold,
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
__global__ void jacobi_nvgpu_cuda_kernel2(long n,long m,REAL *_dev_u,REAL *_dev_uold,long uold_m, int uold_0_offset, int uold_1_offset)
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

__global__ void jacobi_nvgpu_cuda_kernel1(long n,long m,REAL omega,REAL ax,REAL ay,REAL b,REAL *_dev_u,REAL *_dev_f, REAL *_dev_uold,
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




void jacobi_nvgpu_cuda_wrapper2(omp_offloading_t *off, long n,long m,REAL *u,REAL *uold,long uold_m, int uold_0_offset, int uold_1_offset)
{
int threads_per_team = omp_get_optimal_threads_per_team(off->dev);
		int teams_per_league = omp_get_optimal_teams_per_league(off->dev, threads_per_team, n*m);
		jacobi_nvgpu_cuda_kernel2<<<teams_per_league, threads_per_team, 0,off->stream->systream.cudaStream>>>(n, m,(REAL*)u,(REAL*)uold, uold_1_length,uold_0_offset, uold_1_offset);
}


void jacobi_nvgpu_cuda_wrapper1(omp_offloading_t *off, long n,long m,REAL omega,REAL ax,REAL ay,REAL b,REAL *_dev_u,REAL *_dev_f, \
 REAL *_dev_uold, long uold_m, int uold_0_offset, int uold_1_offset, int start_i, int start_j, REAL *_dev_per_block_error)
{
		int threads_per_team = omp_get_optimal_threads_per_team(off->dev);
		int teams_per_league = omp_get_optimal_teams_per_league(off->dev, threads_per_team, n*m);

		/* for reduction operation */
		REAL * _dev_per_block_error = (REAL*)omp_map_malloc_dev(off->dev, teams_per_league * sizeof(REAL));
		//printf("dev: %d teams per league, err block mem: %X\n", teams_per_league, _dev_per_block_error);
		REAL _host_per_block_error[teams_per_league];
		//printf("%d device: original offset: %d, mapped_offset: %d, length: %d\n", __i__, offset_n, start_n, length_n);
		/* Launch CUDA kernel ... */
		/** since here we do the same mapping, so will reuse the _threads_per_block and _num_blocks */
		jacobi_nvgpu_cuda_kernel1<<<teams_per_league, threads_per_team,(threads_per_team * sizeof(REAL)),
				off->stream->systream.cudaStream>>>(n, m,
				omega, ax, ay, b, (REAL*)u, (REAL*)f, (REAL*)uold,uold_1_length, uold_0_offset, uold_1_offset, i_start, j_start, _dev_per_block_error);

		/* copy back the results of reduction in blocks */
		//printf("copy back reduced error: %X <-- %X\n", _host_per_block_error, _dev_per_block_error);
		omp_map_memcpy_from_async(_host_per_block_error, _dev_per_block_error, off->dev, sizeof(REAL)*teams_per_league, off->stream);
		omp_stream_sync(off->stream);

		iargs->error[off->devseqid] = xomp_beyond_block_reduction_float(_host_per_block_error, teams_per_league, XOMP_REDUCTION_PLUS);
		//cudaStreamAddCallback(__dev_stream__[__i__].systream.cudaStream, xomp_beyond_block_reduction_float_stream_callback, args, 0);
		omp_map_free_dev(off->dev, _dev_per_block_error);
}

#endif