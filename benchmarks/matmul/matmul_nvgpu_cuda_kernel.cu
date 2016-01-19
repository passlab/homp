#include <homp.h>
#include "matmul.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>


#include "xomp_cuda_lib_inlined.cu"
__global__ void matmul_nvgpu_cuda_kernel(long i, long j,long k,REAL *_dev_a,REAL *_dev_b,REAL *_dev_c)
{
 long ij;
  long  _dev_i, _dev_j, _dev_k;
  long _dev_lower, _dev_upper;
  // variables for adjusted loop info considering both original chunk size and step(strip)
  long _dev_loop_chunk_size;
  long _dev_loop_sched_index;
  long _dev_loop_stride;

  // 1-D thread block:
  long _dev_thread_num = gridDim.x * blockDim.x;
  long _dev_thread_id = blockDim.x * blockIdx.x + threadIdx.x;

  //adjust bound to be inclusive later
  long orig_start =0;
  long orig_end = i*j-1;
  long orig_step = 1;
  long orig_chunk_size = 1;

//  XOMP_accelerator_loop_default (0, MSIZE*MSIZE -1 , 1, &_dev_lower, &_dev_upper);
  XOMP_static_sched_init (orig_start, orig_end, orig_step, orig_chunk_size, _dev_thread_num, _dev_thread_id, \
      & _dev_loop_chunk_size , & _dev_loop_sched_index, & _dev_loop_stride);

  //XOMP_accelerator_loop_default (1, (n-1)*(m-1)-1, 1, &_dev_lower, &_dev_upper);
  while (XOMP_static_sched_next (&_dev_loop_sched_index, orig_end,orig_step, _dev_loop_stride, _dev_loop_chunk_size, _dev_thread_num, _dev_thread_id, & _dev_lower, & _dev_upper))
  {
  for (ij = _dev_lower; ij <= _dev_upper; ij ++)
//  for (_dev_i = _dev_lower; _dev_i<= _dev_upper; _dev_i ++)
//    for (j = 0; j < MSIZE; j++)
    {
      _dev_i = ij/k;
      _dev_j = ij%k;
      REAL c= 0.0;
      for (_dev_k = 0; _dev_k < k; _dev_k++)
        c += _dev_a[_dev_i * k + _dev_k] * _dev_b[_dev_k * j + _dev_j];
      _dev_c[_dev_i * j + _dev_j] = c;
    }
  } // end while
}

void matmul_nvgpu_cuda_wrapper(omp_offloading_t *off, long i, long j,long k,REAL *A,REAL *B,REAL *C)
{
int threads_per_team = omp_get_optimal_threads_per_team(off->dev);
		int teams_per_league = omp_get_optimal_teams_per_league(off->dev, threads_per_team, i*j);
		//	printf("device: %d, range: %d:%d\n", __i__, start_i, length_i);
//    cublasHandle_t handle;
//    cublasCreate(&handle);
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    cublasSgemm(off->dev->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, i, j, k, &alpha, A, i, B, k, &beta, C, i);
//    cublasDestroy(handle);
		//matmul_nvgpu_cuda_kernel<<<teams_per_league,threads_per_team, 0, off->stream->systream.cudaStream>>>
		//(i, j, k, (REAL *)A, (REAL *)B, (REAL *)C);
}
