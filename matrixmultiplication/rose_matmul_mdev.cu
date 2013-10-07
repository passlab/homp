/*
 * Rectangular matrix multiplication, started from MIT Cilk matmul.cilk example
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "omp.h"
#include <pthread.h>
#include <string.h>
#define REAL float
#include "libxomp.h" 
#include "xomp_cuda_lib_inlined.cu" 

void zero(float *A,int n)
{
  int i;
  int j;
{
    int _p_i;
    int _p_j;
    long p_index_;
    long p_lower_;
    long p_upper_;
    XOMP_loop_default(0,n - 1,1,&p_lower_,&p_upper_);
    for (p_index_ = p_lower_; p_index_ <= p_upper_; p_index_ += 1) {
      for (_p_j = 0; _p_j < n; _p_j++) {
        A[(p_index_ * n) + _p_j] = 0.0;
      }
    }
    XOMP_barrier();
  }
}

void init(float *A,int n)
{
  int i;
  int j;
{
    int _p_i;
    int _p_j;
    long p_index_;
    long p_lower_;
    long p_upper_;
    XOMP_loop_default(0,n - 1,1,&p_lower_,&p_upper_);
    for (p_index_ = p_lower_; p_index_ <= p_upper_; p_index_ += 1) {
      for (_p_j = 0; _p_j < n; _p_j++) {
        A[(p_index_ * n) + _p_j] = (drand48());
      }
    }
    XOMP_barrier();
  }
}

double maxerror(float *A,float *B,int n)
{
  int i;
  int j;
  double error = 0.0;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      double diff = ((A[(i * n) + j] - B[(i * n) + j]) / A[(i * n) + j]);
//        printf("%4f -- %4f\n", A[i*n+j], B[i*n+j]);
      if (diff < 0) 
        diff = -diff;
      if (diff > error) 
        error = diff;
    }
  }
  return error;
}

void iter_matmul(float *A,float *B,float *C,int n)
{
  int i;
  int j;
  int k;
  for (i = 0; i < n; i++) 
    for (k = 0; k < n; k++) {
      float c = 0.0;
      for (j = 0; j < n; j++) 
        c += (A[(i * n) + j] * B[(j * n) + k]);
      C[(i * n) + k] = c;
    }
}

struct OUT__3__7117___data 
{
  void *A_p;
  void *B_p;
  void *C_p;
  void *n_p;
}
;
static void OUT__3__7117__(void *__out_argv);

void omp_matmul(float *A,float *B,float *C,int n)
{
  int i;
  int j;
  int k;
  struct OUT__3__7117___data __out_argv2__7117__;
  __out_argv2__7117__.n_p = ((void *)(&n));
  __out_argv2__7117__.C_p = ((void *)(&C));
  __out_argv2__7117__.B_p = ((void *)(&B));
  __out_argv2__7117__.A_p = ((void *)(&A));
  XOMP_parallel_start(OUT__3__7117__,&__out_argv2__7117__,1,0,"/data/yy8/2013-8-multiple-gpu-work/benchmarks/matrixmultiplication/matmul.c",73);
  XOMP_parallel_end("/data/yy8/2013-8-multiple-gpu-work/benchmarks/matrixmultiplication/matmul.c",80);
}
/* one device */

__global__ void OUT__2__7117__(int n,float *_dev_A,float *_dev_B,float *_dev_C)
{
  int _p_i;
  int _p_j;
  int _p_k;
  int _dev_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (_dev_i >= 0 && _dev_i <= n - 1) {
    for (_p_k = 0; _p_k < n; _p_k++) {
      float c = 0.0;
      for (_p_j = 0; _p_j < n; _p_j++) 
        c += (_dev_A[(_dev_i * n) + _p_j] * _dev_B[(_p_j * n) + _p_k]);
      _dev_C[(_dev_i * n) + _p_k] = c;
    }
  }
}

void ompacc_matmul(float *A,float *B,float *C,int n)
{
  int i;
  int j;
  int k;
{
    float *_dev_A;
    int _dev_A_size = sizeof(float ) * (n - 0) * (n - 0);
    _dev_A = ((float *)(xomp_deviceMalloc(_dev_A_size)));
    xomp_memcpyHostToDevice(((void *)_dev_A),((const void *)A),_dev_A_size);
    float *_dev_B;
    int _dev_B_size = sizeof(float ) * (n - 0) * (n - 0);
    _dev_B = ((float *)(xomp_deviceMalloc(_dev_B_size)));
    xomp_memcpyHostToDevice(((void *)_dev_B),((const void *)B),_dev_B_size);
    float *_dev_C;
    int _dev_C_size = sizeof(float ) * (n - 0) * (n - 0);
    _dev_C = ((float *)(xomp_deviceMalloc(_dev_C_size)));
/* Launch CUDA kernel ... */
    int _threads_per_block_ = xomp_get_maxThreadsPerBlock();
    int _num_blocks_ = xomp_get_max1DBlock(n - 1 - 0 + 1);
    OUT__2__7117__<<<_num_blocks_,_threads_per_block_>>>(n,_dev_A,_dev_B,_dev_C);
    xomp_freeDevice(_dev_A);
    xomp_freeDevice(_dev_B);
    xomp_memcpyDeviceToHost(((void *)C),((const void *)_dev_C),_dev_C_size);
    xomp_freeDevice(_dev_C);
  }
}
#if 0
/* multiple device */
#endif

void openacc_matmul(float *A,float *B,float *C,int n)
{
  int i;
  int j;
  int k;
/* #pragma acc kernels copyin(A[0:n][0:n],B[0:n][0:n]) copyout(C[0:n][0:n]) */
//#pragma acc kernels loop copyin(A[0:n*n],B[0:n*n]) copyout(C[0:n*n])
  
#pragma acc parallel loop copyin ( A [ 0 : n * n ], B [ 0 : n * n ] ) copyout ( C [ 0 : n * n ] ) collapse ( 2 )
  for (i = 0; i < n; i++) 
    for (k = 0; k < n; k++) {
      float c = 0.0;
      for (j = 0; j < n; j++) 
        c += (A[(i * n) + j] * B[(j * n) + k]);
      C[(i * n) + k] = c;
    }
}

struct OUT__1__7117___data 
{
  void *n_p;
  void *num_threads_p;
  void *A_p;
  void *B_p;
  void *C_seq_p;
  void *C_omp_for_p;
  void *C_acc_p;
}
;
static void OUT__1__7117__(void *__out_argv);

int main(int argc,char *argv[])
{
  int n;
  int num_threads;
  float *A;
  float *B;
  float *C_seq;
  float *C_omp_for;
  float *C_acc;
  double seq_elapsed;
  double omp_for_elapsed;
  double acc_elapsed;
  if (argc != 2) {
    fprintf(stderr,"Usage: matmul <n>\n");
    exit(1);
  }
  n = atoi(argv[1]);
  A = ((float *)(malloc(((n * n) * sizeof(float )))));
  B = ((float *)(malloc(((n * n) * sizeof(float )))));
  C_seq = ((float *)(malloc(((n * n) * sizeof(float )))));
  C_omp_for = ((float *)(malloc(((n * n) * sizeof(float )))));
  C_acc = ((float *)(malloc(((n * n) * sizeof(float )))));
  srand48((1 << 12));
  struct OUT__1__7117___data __out_argv1__7117__;
  __out_argv1__7117__.C_acc_p = ((void *)(&C_acc));
  __out_argv1__7117__.C_omp_for_p = ((void *)(&C_omp_for));
  __out_argv1__7117__.C_seq_p = ((void *)(&C_seq));
  __out_argv1__7117__.B_p = ((void *)(&B));
  __out_argv1__7117__.A_p = ((void *)(&A));
  __out_argv1__7117__.num_threads_p = ((void *)(&num_threads));
  __out_argv1__7117__.n_p = ((void *)(&n));
  XOMP_parallel_start(OUT__1__7117__,&__out_argv1__7117__,1,0,"/data/yy8/2013-8-multiple-gpu-work/benchmarks/matrixmultiplication/matmul.c",152);
  XOMP_parallel_end("/data/yy8/2013-8-multiple-gpu-work/benchmarks/matrixmultiplication/matmul.c",163);
/* sequential run */
  seq_elapsed = omp_get_wtime();
//    iter_matmul(A, B, C_seq, n);
  seq_elapsed = (omp_get_wtime() - seq_elapsed);
/* openmp parallel for version */
  omp_for_elapsed = omp_get_wtime();
//    omp_matmul(A, B, C_omp_for, n);
  omp_for_elapsed = (omp_get_wtime() - omp_for_elapsed);
/* we currently cannot do the OpenMP acc and OpenACC run in once */
#ifndef OPENACC
/* openmp acc version */
  acc_elapsed = omp_get_wtime();
  ompacc_matmul(A,B,C_acc,n);
  acc_elapsed = (omp_get_wtime() - acc_elapsed);
#else
#endif
  printf("=======================================================================\n");
  printf("\t\tmatmul(%dx%d) example on %d threads(cores)\n",n,n,num_threads);
  printf("-----------------------------------------------------------------------\n");
  printf("Performance:  Runtime (s)\t MFLOPS\t\t\t Error\n");
  printf("-----------------------------------------------------------------------\n");
  printf("Sequential      :  %4f \t\t %4f\t\t%g\n",seq_elapsed,((((2.0 * n) * n) * n) / (1.0e6 * seq_elapsed)),maxerror(C_seq,C_seq,n));
  printf("OMP For         :  %4f \t\t %4f\t\t%g\n",omp_for_elapsed,((((2.0 * n) * n) * n) / (1.0e6 * omp_for_elapsed)),maxerror(C_seq,C_omp_for,n));
#ifndef OPENACC
  printf("OMP ACC         :  %4f \t\t %4f\t\t%g\n",acc_elapsed,((((2.0 * n) * n) * n) / (1.0e6 * acc_elapsed)),maxerror(C_seq,C_acc,n));
#else
#endif
  free(C_acc);
  free(C_omp_for);
  free(C_seq);
  free(B);
  free(A);
  return 0;
}

static void OUT__1__7117__(void *__out_argv)
{
  int *n = (int *)(((struct OUT__1__7117___data *)__out_argv) -> n_p);
  int *num_threads = (int *)(((struct OUT__1__7117___data *)__out_argv) -> num_threads_p);
  float **A = (float **)(((struct OUT__1__7117___data *)__out_argv) -> A_p);
  float **B = (float **)(((struct OUT__1__7117___data *)__out_argv) -> B_p);
  float **C_seq = (float **)(((struct OUT__1__7117___data *)__out_argv) -> C_seq_p);
  float **C_omp_for = (float **)(((struct OUT__1__7117___data *)__out_argv) -> C_omp_for_p);
  float **C_acc = (float **)(((struct OUT__1__7117___data *)__out_argv) -> C_acc_p);
  if (XOMP_master()) {
     *num_threads = omp_get_num_threads();
  }
  init( *A, *n);
  init( *B, *n);
  zero( *C_seq, *n);
  zero( *C_omp_for, *n);
  zero( *C_acc, *n);
}

static void OUT__3__7117__(void *__out_argv)
{
  float **A = (float **)(((struct OUT__3__7117___data *)__out_argv) -> A_p);
  float **B = (float **)(((struct OUT__3__7117___data *)__out_argv) -> B_p);
  float **C = (float **)(((struct OUT__3__7117___data *)__out_argv) -> C_p);
  int *n = (int *)(((struct OUT__3__7117___data *)__out_argv) -> n_p);
  int _p_i;
  int _p_j;
  int _p_k;
  long p_index_;
  long p_lower_;
  long p_upper_;
  XOMP_loop_default(0, *n - 1,1,&p_lower_,&p_upper_);
  for (p_index_ = p_lower_; p_index_ <= p_upper_; p_index_ += 1) {
    for (_p_k = 0; _p_k <  *n; _p_k++) {
      float c = 0.0;
      for (_p_j = 0; _p_j <  *n; _p_j++) 
        c += (( *A)[(p_index_ *  *n) + _p_j] * ( *B)[(_p_j *  *n) + _p_k]);
      ( *C)[(p_index_ *  *n) + _p_k] = c;
    }
  }
  XOMP_barrier();
}

#if 0
/* multiple device */
/* B, C column-major partition */
void ompacc_matmul_mdev_2(REAL *A, REAL *B, REAL *C, int n)
{
    int i, j, k;
#pragma omp target device(*) map(from:C{0:n}[0:n]>>(*)), map(to:n,A[0:n][0:n],B{0:n}[0:n]>>(*)
    for (i = 0; i < n; i++)
#pragma omp parallel for private(i,j,k) dist_iteration match_range C{}[]
        for (k = 0; k < n; k++) {
            REAL c = 0.0;
            for (j = 0; j < n; j++)
                c += A[i * n + j] * B[j * n + k];
            C[i * n + k] = c;
        }
}

/* A,B, C row-column partition */
void ompacc_matmul_mdev_3(REAL *A, REAL *B, REAL *C, int n)
{
    int i, j, k;
#pragma omp target device(*)=>(:)(:) map(from:C[0:n][0:n]>>(:)(:)), map(to:n,A[0:n]{0:n}>>(:){:},B{0:n}[0:n]>>{:}())
#pragma omp parallel for private(i,j,k) dist_iteration match_range C[]{}
    for (i = 0; i < n; i++)
#pragma omp parallel for private(i,j,k) dist_iteration match_range C{}[]
        for (k = 0; k < n; k++) {
            REAL c = 0.0;
            for (j = 0; j < n; j++)
                c += A[i * n + j] * B[j * n + k];
            C[i * n + k] = c;
        }
}
#endif

#if 0
/* multiple device */

/* A, C row-major partition */
void ompacc_matmul_mdev_1(REAL *A, REAL *B, REAL *C, int n)
{
    int i, j, k;
#pragma omp target device(*) map(from:C[0:n]{0:n}>>(*)), map(to:n,A[0:n]{0:n}>>(*),B[0:n][0:n])
#pragma omp parallel for private(i,j,k) dist_iteration match_range C[:]
    for (i = 0; i < n; i++)
        for (k = 0; k < n; k++) {
            REAL c = 0.0;
            for (j = 0; j < n; j++)
                c += A[i * n + j] * B[j * n + k];
            C[i * n + k] = c;
        }
}
#endif
void matmul_ompacc_mdev_v1(REAL *A, REAL *B, REAL *c,  int n)
{
    /* get number of target devices specified by the programmers */
    int __num_target_devices__ = 4; /*XXX: = runtime or compiler generated code */
    int __num_mapped_variables__ = 2; /* XXX: need compiler output */
     
    /* declare the streams used for the async launching
     * For each device, there is one stream
     */
    cudaStream_t __dev_stream__[__num_target_devices__];

    /* for each mapped variables on each device, we need a data_map for this,
     * our compiler should help to create an unique id for each of the mapped variables, which will
     * be used to access the map array
     *
     * in this example, A is 0, B is 1, C is 2;
     */
    omp_data_map_t __data_maps__[__num_target_devices__][__num_mapped_variables__];
    int __ndev_i__;
    for (__ndev_i__ = 0; __ndev_i__<__num_target_devices__; __ndev_i__++) {
        cudaSetDevice(__ndev_i__);
        cudaStreamCreate(&__dev_stream__[__ndev_i__]);

        /***************** for each mapped variable has to and tofrom, if it has region mapped to this __ndev_i__ id, we need code here *******************************/
        omp_data_map_t * dev_map_A__ = &__data_maps__[__ndev_i__][0]; /* 0 is given by compiler here */
        dev_map_A__->device_id = __ndev_i__; //omp_get_device(__ndev_i__)->sysid;
        dev_map_A__->map_type = OMP_MAP_TO;  /* from compiler */
        dev_map_A__->source_ptr = x;
        dev_map_A__->dim[0] = n;
        dev_map_A__->dim[1] = n;
        dev_map_A__->dim[2] = 1;
        dev_map_A__->sizeof_element = sizeof(double);

        dev_map_A__->map_offset[0] = 0;/* from compiler */
        dev_map_A__->map_offset[1] = __ndev_i__ * n/__num_target_devices__; /* chunking n into __ndev_i__ pieces, from compiler */
        dev_map_A__->map_offset[2] = 0;/* from compiler */

        dev_map_A__->map_dim[0] = 1;
        dev_map_A__->map_dim[1] = n/__num_target_devices__;/* from compiler */
        dev_map_A__->map_dim[2] = 1;

        omp_map_buffer(dev_map_A__, 0);
        dev_map_A__->stream = &__dev_stream__[__ndev_i__];

        omp_deviceMalloc_memcpyHostToDeviceAsync(dev_map_A__);
        omp_print_data_map(dev_map_A__);
		/***************************************************************** for B *********************************************************************/
        omp_data_map_t * dev_map_B__ = &__data_maps__[0][0]; /* 0 is given by compiler here */
        dev_map_B__->device_id = __ndev_i__; //omp_get_device(__ndev_i__)->sysid;
        dev_map_B__->map_type = OMP_MAP_TO;  /* from compiler */
        dev_map_B__->source_ptr = x;
        dev_map_B__->dim[0] = n;
        dev_map_B__->dim[1] = n;
        dev_map_B__->dim[2] = 1;
        dev_map_B__->sizeof_element = sizeof(double);

        dev_map_B__->map_offset[0] = 0;/* from compiler */
        dev_map_B__->map_offset[1] = 0;/* from compiler */
        dev_map_B__->map_offset[2] = 0;/* from compiler */

        dev_map_B__->map_dim[0] = 1;
        dev_map_B__->map_dim[1] = 1;
        dev_map_B__->map_dim[2] = 1;

        omp_map_buffer(dev_map_B__, 0);
        dev_map_B__->stream = &__dev_stream__[__ndev_i__];

        omp_deviceMalloc_memcpyHostToDeviceAsync(dev_map_B__);
        omp_print_data_map(dev_map_B__);
		/***************************************************************** for C *********************************************************************/
        omp_data_map_t * dev_map_C__ = &__data_maps__[__ndev_i__][0]; /* 0 is given by compiler here */
        dev_map_C__->device_id = __ndev_i__; //omp_get_device(__ndev_i__)->sysid;
        dev_map_C__->map_type = OMP_MAP_TO;  /* from compiler */
        dev_map_C__->source_ptr = x;
        dev_map_C__->dim[0] = n;
        dev_map_C__->dim[1] = n;
        dev_map_C__->dim[2] = 1;
        dev_map_C__->sizeof_element = sizeof(double);

        dev_map_C__->map_offset[0] = 0;/* from compiler */
        dev_map_C__->map_offset[1] = __ndev_i__ * n/__num_target_devices__; /* chunking n into __ndev_i__ pieces, from compiler */
        dev_map_C__->map_offset[2] = 0;/* from compiler */

        dev_map_C__->map_dim[0] = 1;
        dev_map_C__->map_dim[1] = n/__num_target_devices__;/* from compiler */
        dev_map_C__->map_dim[2] = 1;

        omp_map_buffer(dev_map_C__, 0);
        dev_map_C__->stream = &__dev_stream__[__ndev_i__];

        omp_deviceMalloc_memcpyHostToDeviceAsync(dev_map_C__);
        omp_print_data_map(dev_map_C__);
        /*************************************************************************************************************************************************************/
        /* Launch CUDA kernel ... */
        int _threads_per_block_ = xomp_get_maxThreadsPerBlock();
        int _num_blocks_ = xomp_get_max1DBlock(n - 1 - 0 + 1);
        /* in this example, this information could be provided by compiler analysis, but we can also retrive this from runtime as a more
         * general solution */
         long start_n, length_n;
        omp_loop_map_range(dev_map_A__, 0, -1, -1, &start_n, &length_n);
        /* the argu for this function should be the original pointer (x in this example) and the runtime should search and retrieve the
         * device map object
         */
        printf("device: %d, range: %d:%d\n", __ndev_i__, start_n, length_n);

        OUT__2__7117__<<<_num_blocks_,_threads_per_block_, 0, __dev_stream__[__ndev_i__]>>>(start_n, length_n,a,(double *)dev_map_A__->map_dev_ptr, (double *)__dev_map_B__->map_dev_ptr, (double *)__dev_map_C__->map_dev_ptr);

        /***************************************************************************************************************************************************/
        /****************** for each from and tofrom, we need call to DeviceToHost memcpy */
        omp_memcpyDeviceToHostAsync(__dev_map_C__);
    }

    omp_postACCKernel(__num_target_devices__, __num_mapped_variables__, __dev_stream__, (omp_data_map_t*)__data_maps__);
}

#if 0
/* multiple device */
/* B, C column-major partition */
void ompacc_matmul_mdev_2(REAL *A, REAL *B, REAL *C, int n)
{
    int i, j, k;
#pragma omp target device(*) map(from:C{0:n}[0:n]>>(*)), map(to:n,A[0:n][0:n],B{0:n}[0:n]>>(*)
    for (i = 0; i < n; i++)
#pragma omp parallel for private(i,j,k) dist_iteration match_range C{}[]
        for (k = 0; k < n; k++) {
            REAL c = 0.0;
            for (j = 0; j < n; j++)
                c += A[i * n + j] * B[j * n + k];
            C[i * n + k] = c;
        }
}
#endif

void matmul_ompacc_mdev_v2(REAL *A, REAL *B, REAL *c,  int n)
{
    /* get number of target devices specified by the programmers */
    int __num_target_devices__ = 4; /*XXX: = runtime or compiler generated code */
    int __num_mapped_variables__ = 2; /* XXX: need compiler output */
     
    /* declare the streams used for the async launching
     * For each device, there is one stream
     */
    cudaStream_t __dev_stream__[__num_target_devices__];

    /* for each mapped variables on each device, we need a data_map for this,
     * our compiler should help to create an unique id for each of the mapped variables, which will
     * be used to access the map array
     *
     * in this example, A is 0, B is 1, C is 2;
     */
    omp_data_map_t __data_maps__[__num_target_devices__][__num_mapped_variables__];
    int __ndev_i__;
    for (__ndev_i__ = 0; __ndev_i__<__num_target_devices__; __ndev_i__++) {
        cudaSetDevice(__ndev_i__);
        cudaStreamCreate(&__dev_stream__[__ndev_i__]);

        /***************** for each mapped variable has to and tofrom, if it has region mapped to this __ndev_i__ id, we need code here *******************************/
        omp_data_map_t * dev_map_A__ = &__data_maps__[0][0]; /* 0 is given by compiler here */
        dev_map_A__->device_id = __ndev_i__; //omp_get_device(__ndev_i__)->sysid;
        dev_map_A__->map_type = OMP_MAP_TO;  /* from compiler */
        dev_map_A__->source_ptr = x;
        dev_map_A__->dim[0] = n;
        dev_map_A__->dim[1] = n;
        dev_map_A__->dim[2] = 1;
        dev_map_A__->sizeof_element = sizeof(double);

        dev_map_A__->map_offset[0] = 0;/* from compiler */
        dev_map_A__->map_offset[1] = 0;/* from compiler */
        dev_map_A__->map_offset[2] = 0;/* from compiler */

        dev_map_A__->map_dim[0] = 1;
        dev_map_A__->map_dim[1] = 1;
        dev_map_A__->map_dim[2] = 1;

        omp_map_buffer(dev_map_A__, 0);
        dev_map_A__->stream = &__dev_stream__[__ndev_i__];

        omp_deviceMalloc_memcpyHostToDeviceAsync(dev_map_A__);
        omp_print_data_map(dev_map_A__);
		/***************************************************************** for B *********************************************************************/
        omp_data_map_t * dev_map_B__ = &__data_maps__[0][__ndev_i__]; /* 0 is given by compiler here */
        dev_map_B__->device_id = __ndev_i__; //omp_get_device(__ndev_i__)->sysid;
        dev_map_B__->map_type = OMP_MAP_TO;  /* from compiler */
        dev_map_B__->source_ptr = x;
        dev_map_B__->dim[0] = n;
        dev_map_B__->dim[1] = n;
        dev_map_B__->dim[2] = 1;
        dev_map_B__->sizeof_element = sizeof(double);

        dev_map_B__->map_offset[0] = __ndev_i__ * n/__num_target_devices__; /* chunking n into __ndev_i__ pieces, from compiler */
        dev_map_B__->map_offset[1] = 0;/* from compiler */
        dev_map_B__->map_offset[2] = 0;/* from compiler */

        dev_map_B__->map_dim[0] = n/__num_target_devices__;/* from compiler */
        dev_map_B__->map_dim[1] = 1;
        dev_map_B__->map_dim[2] = 1;

        omp_map_buffer(dev_map_B__, 0);
        dev_map_B__->stream = &__dev_stream__[__ndev_i__];

        omp_deviceMalloc_memcpyHostToDeviceAsync(dev_map_B__);
        omp_print_data_map(dev_map_B__);
		/***************************************************************** for C *********************************************************************/
        omp_data_map_t * dev_map_C__ = &__data_maps__[0][__ndev_i__]; /* 0 is given by compiler here */
        dev_map_C__->device_id = __ndev_i__; //omp_get_device(__ndev_i__)->sysid;
        dev_map_C__->map_type = OMP_MAP_TO;  /* from compiler */
        dev_map_C__->source_ptr = x;
        dev_map_C__->dim[0] = n;
        dev_map_C__->dim[1] = n;
        dev_map_C__->dim[2] = 1;
        dev_map_C__->sizeof_element = sizeof(double);

        dev_map_C__->map_offset[0] = __ndev_i__ * n/__num_target_devices__; /* chunking n into __ndev_i__ pieces, from compiler */
        dev_map_C__->map_offset[1] = 0;/* from compiler */
        dev_map_C__->map_offset[2] = 0;/* from compiler */

        dev_map_C__->map_dim[0] = n/__num_target_devices__;/* from compiler */
        dev_map_C__->map_dim[1] = 1;
        dev_map_C__->map_dim[2] = 1;

        omp_map_buffer(dev_map_C__, 0);
        dev_map_C__->stream = &__dev_stream__[__ndev_i__];

        omp_deviceMalloc_memcpyHostToDeviceAsync(dev_map_C__);
        omp_print_data_map(dev_map_C__);
        /*************************************************************************************************************************************************************/
        /* Launch CUDA kernel ... */
        int _threads_per_block_ = xomp_get_maxThreadsPerBlock();
        int _num_blocks_ = xomp_get_max1DBlock(n - 1 - 0 + 1);
        /* in this example, this information could be provided by compiler analysis, but we can also retrive this from runtime as a more
         * general solution */
         long start_n, length_n;
        omp_loop_map_range(dev_map_A__, 0, -1, -1, &start_n, &length_n);
        /* the argu for this function should be the original pointer (x in this example) and the runtime should search and retrieve the
         * device map object
         */
        printf("device: %d, range: %d:%d\n", __ndev_i__, start_n, length_n);

        OUT__2__7117__<<<_num_blocks_,_threads_per_block_, 0, __dev_stream__[__ndev_i__]>>>(start_n, length_n,a,(double *)dev_map_A__->map_dev_ptr, (double *)__dev_map_B__->map_dev_ptr, (double *)__dev_map_C__->map_dev_ptr);

        /***************************************************************************************************************************************************/
        /****************** for each from and tofrom, we need call to DeviceToHost memcpy */
        omp_memcpyDeviceToHostAsync(__dev_map_C__);
    }

    omp_postACCKernel(__num_target_devices__, __num_mapped_variables__, __dev_stream__, (omp_data_map_t*)__data_maps__);
}

#if 0
/* multiple device */
/* A,B, C row-column partition */
void ompacc_matmul_mdev_3(REAL *A, REAL *B, REAL *C, int n)
{
    int i, j, k;
#pragma omp target device(*)=>(:)(:) map(from:C[0:n][0:n]>>(:)(:)), map(to:n,A[0:n]{0:n}>>(:){:},B{0:n}[0:n]>>{:}())
#pragma omp parallel for private(i,j,k) dist_iteration match_range C[]{}
    for (i = 0; i < n; i++)
#pragma omp parallel for private(i,j,k) dist_iteration match_range C{}[]
        for (k = 0; k < n; k++) {
            REAL c = 0.0;
            for (j = 0; j < n; j++)
                c += A[i * n + j] * B[j * n + k];
            C[i * n + k] = c;
        }
}
#endif

// Cannon's Matrix multiplication performs 2-D partitioned matrix-multiply.
// The implementation requires skewing.

void matmul_ompacc_mdev_v3(REAL *A, REAL *B, REAL *c,  int n)
{
}
