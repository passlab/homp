#include "matvec.h"
#include <homp.h>
#include "cublas_v2.h"

#include "xomp_cuda_lib_inlined.cu"
__global__ void matverc_nvgpu_cuda_kernel(long n, long start_n, long length_n,REAL *_dev_a,REAL *_dev_x,REAL *_dev_y)
{
  int i,j;
  long _dev_lower;
  long  _dev_upper;
  long _dev_loop_chunk_size;
  long _dev_loop_sched_index;
  long _dev_loop_stride;
  int _dev_thread_num = getCUDABlockThreadCount(1);
  int _dev_thread_id = getLoopIndexFromCUDAVariables(1);
  XOMP_static_sched_init(start_n,start_n + length_n - 1,1,1,_dev_thread_num,_dev_thread_id,&_dev_loop_chunk_size,&_dev_loop_sched_index,&_dev_loop_stride);
  while(XOMP_static_sched_next(&_dev_loop_sched_index,start_n + length_n - 1,1,_dev_loop_stride,_dev_loop_chunk_size,_dev_thread_num,_dev_thread_id,&_dev_lower,&_dev_upper))
    for (i = _dev_lower; i <= _dev_upper; i += 1) {
        for (j = 0; j<n; j++)
         _dev_y[i] += _dev_a[i*n+j] * _dev_x[j];
//		printf("x[%d]: %f, y[%d]: %f\n", i, x[i], i, y[i]);
    }
}

void matvec_nvgpu_cuda_wrapper(omp_offloading_t *off, long n, long start_n, long length_n,REAL *a,REAL *x,REAL *y)
{
int threads_per_team = omp_get_optimal_threads_per_team(off->dev);
		int teams_per_league = omp_get_optimal_teams_per_league(off->dev, threads_per_team, length_n);
       // matverc_nvgpu_cuda_kernel<<<teams_per_league,threads_per_team, 0, off->stream->systream.cudaStream>>>(n, start_n, length_n,(REAL *)a,(REAL *)x,(REAL *)y);
        //cublasHandle_t handle;
        //cublasCreate(&handle);
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        cublasSgemv((cublasHandle_t)off->dev->cublas_handle, CUBLAS_OP_N, length_n - start_n, n,  &alpha, a, n, x, 1, &beta, y, 1);
        //cublasDestroy(handle);
}
