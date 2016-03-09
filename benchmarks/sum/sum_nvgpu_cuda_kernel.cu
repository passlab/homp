#include "homp.h"
#include "sum.h"
#include <cublas_v2.h>

#include "xomp_cuda_lib_inlined.cu"
__global__ void axpy_nvgpu_cuda_kernel( long start_n,  long length_n,REAL a,REAL *x,REAL *y, REAL *block_result)
{
  long i;
  long _dev_lower;
  long  _dev_upper;
  long _dev_loop_chunk_size;
  long _dev_loop_sched_index;
  long _dev_loop_stride;
  REAL result = 0.0;

  int _dev_thread_num = getCUDABlockThreadCount(1);
  int _dev_thread_id = getLoopIndexFromCUDAVariables(1);
  XOMP_static_sched_init(start_n,start_n + length_n - 1,1,1,_dev_thread_num,_dev_thread_id,&_dev_loop_chunk_size,&_dev_loop_sched_index,&_dev_loop_stride);
  while(XOMP_static_sched_next(&_dev_loop_sched_index,start_n + length_n - 1,1,_dev_loop_stride,_dev_loop_chunk_size,_dev_thread_num,_dev_thread_id,&_dev_lower,&_dev_upper))
    for (i = _dev_lower; i <= _dev_upper; i += 1) {
      result += y[i] * x[i];
    }

  //xomp_inner_block_reduction_float(result,block_result,6);
  xomp_inner_block_reduction_double(result,block_result,6);
}

void sum_nvgpu_cuda_wrapper(omp_offloading_t *off, long start_n,  long length_n,REAL *x,REAL *y, REAL*result) {
    int threads_per_team = omp_get_optimal_threads_per_team(off->dev);
    int teams_per_league = omp_get_optimal_teams_per_league(off->dev, threads_per_team, length_n);
    //printf("threads: %d, blocks: %d\n", threads_per_team, teams_per_league);

    /* for reduction operation */
	REAL * _dev_per_block_result = (REAL*)omp_map_malloc_dev(off->dev, NULL, teams_per_league * sizeof(REAL));
	//printf("dev: %d teams per league, err block mem: %X\n", teams_per_league, _dev_per_block_result);
	REAL _host_per_block_result[teams_per_league];
    sum_nvgpu_cuda_kernel<<<teams_per_league,threads_per_team, 0, off->stream->systream.cudaStream>>>(start_n, length_n,x,y,_dev_per_block_result);

	/* copy back the results of reduction in blocks */
	//printf("copy back reduced result: %X <-- %X\n", _host_per_block_result, _dev_per_block_result);
	omp_map_memcpy_from_async(_host_per_block_result, _dev_per_block_result, off->dev, sizeof(REAL)*teams_per_league, off->stream);
	omp_stream_sync(off->stream);

	*result = xomp_beyond_block_reduction_double(_host_per_block_result, teams_per_league, XOMP_REDUCTION_PLUS);
	//cudaStreamAddCallback(__dev_stream__[__i__].systream.cudaStream, xomp_beyond_block_reduction_double_stream_callback, args, 0);
	omp_map_free_dev(off->dev, _dev_per_block_result, 0);
}
