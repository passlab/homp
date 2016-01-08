#include "homp.h"
#include "axpy.h"

#if defined (DEVICE_NVGPU_CUDA_SUPPORT)
#include "xomp_cuda_lib_inlined.cu"
__global__ void axpy_nvgpu_cuda_kernel( long start_n,  long length_n,REAL a,REAL *_dev_x,REAL *_dev_y)
{
  int _p_i;
  long _dev_lower;
  long  _dev_upper;
  long _dev_loop_chunk_size;
  long _dev_loop_sched_index;
  long _dev_loop_stride;
  int _dev_thread_num = getCUDABlockThreadCount(1);
  int _dev_thread_id = getLoopIndexFromCUDAVariables(1);
  XOMP_static_sched_init(start_n,start_n + length_n - 1,1,1,_dev_thread_num,_dev_thread_id,&_dev_loop_chunk_size,&_dev_loop_sched_index,&_dev_loop_stride);
  while(XOMP_static_sched_next(&_dev_loop_sched_index,start_n + length_n - 1,1,_dev_loop_stride,_dev_loop_chunk_size,_dev_thread_num,_dev_thread_id,&_dev_lower,&_dev_upper))
    for (_p_i = _dev_lower; _p_i <= _dev_upper; _p_i += 1) {
      _dev_y[_p_i] += a * _dev_x[_p_i];
//		printf("x[%d]: %f, y[%d]: %f\n", i, x[i], i, y[i]);
    }
}
void axpy_nvgpu_cuda_wrapper(omp_offloading_t *off, long start_n,  long length_n,REAL a,REAL *x,REAL *y) {
    int threads_per_team = omp_get_optimal_threads_per_team(off->dev);
    int teams_per_league = omp_get_optimal_teams_per_league(off->dev, threads_per_team, length_n);
    axpy_nvgpu_cuda_kernel<<<teams_per_league,threads_per_team, 0, off->stream->systream.cudaStream>>>(start_n, length_n,a,x,y);
}
#endif