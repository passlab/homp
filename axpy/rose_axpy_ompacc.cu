#include "axpy.h"
/* standard one-dev support */
#include "libxomp.h" 
#include "xomp_cuda_lib_inlined.cu" 

__global__ void OUT__3__5904__(int n,double a,double *_dev_x,double *_dev_y)
{
  int _p_i;
  int _dev_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (_dev_i >= 0 && _dev_i <= n - 1) {
    _dev_y[_dev_i] += (a * _dev_x[_dev_i]);
  }
}

void axpy_ompacc(double *x,double *y,int n,double a)
{
  int i;
/* this one defines both the target device name and data environment to map to,
      I think here we need mechanism to tell the compiler the device type (could be multiple) so that compiler can generate the codes of different versions; 
      we also need to let the runtime know what the target device is so the runtime will chose the right function to call if the code are generated 
      #pragma omp target device (gpu0) map(x, y) 
   */
{
    double *_dev_x;
    int _dev_x_size = sizeof(double ) * (n - 0);
    _dev_x = ((double *)(xomp_deviceMalloc(_dev_x_size)));
    xomp_memcpyHostToDevice(((void *)_dev_x),((const void *)x),_dev_x_size);
    double *_dev_y;
    int _dev_y_size = sizeof(double ) * (n - 0);
    _dev_y = ((double *)(xomp_deviceMalloc(_dev_y_size)));
    xomp_memcpyHostToDevice(((void *)_dev_y),((const void *)y),_dev_y_size);
/* Launch CUDA kernel ... */
    int _threads_per_block_ = xomp_get_maxThreadsPerBlock();
    int _num_blocks_ = xomp_get_max1DBlock(n - 1 - 0 + 1);
    OUT__3__5904__<<<_num_blocks_,_threads_per_block_>>>(n,a,_dev_x,_dev_y);
    xomp_freeDevice(_dev_x);
    xomp_memcpyDeviceToHost(((void *)y),((const void *)_dev_y),_dev_y_size);
    xomp_freeDevice(_dev_y);
  }
}
/* version 1: use omp parallel, i.e. each host thread responsible for one dev */

__global__ void OUT__1__5904__(double a,int partsize,double *_dev_lx,double *_dev_ly)
{
  int _p_i;
  int _dev_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (_dev_i >= 0 && _dev_i <= partsize - 1) {
    _dev_ly[_dev_i] += (a * _dev_lx[_dev_i]);
  }
}

struct OUT__2__5904___data 
{
  void *x_p;
  void *y_p;
  void *n_p;
  void *a_p;
  void *ndev_p;
}
;
#if 0
/* version 2: leveraging the omp worksharing */
/* here we use devices, not device */
/* in this example, the y[0:n], and x[0:n] will be evenly distributed among the ndev devices, scalars such as a and n will each have a mapped copy in all the devices */
/* version 3: we should allow the following data mapping */
/* here we use devices, not device, map to three devices explicitly listed, and the data mapping is explicitly listed too */
/* if no explicitly-specified mapping deviice, they are mapping to all the devices, e.g. scalars such as a and n in this example will each have a mapped copy in all the devices */
/* version 4: the following data mapping is more dynamic */
/* here we use devices, not device, map to three devices explicitly listed, and the data mapping is explicitly listed too */
/* if no explicitly-specified mapping deviice, they are mapping to all the devices, e.g. scalars such as a and n in this example will each have a mapped copy in all the devices */
#endif
static void OUT__2__5904__(void *__out_argv);

void axpy_ompacc_mdev_1(double *x,double *y,int n,double a)
{
  int ndev = omp_get_num_devices();
  printf("There are %d devices available\n",ndev);
  double *lx;
  double *ly;
  struct OUT__2__5904___data __out_argv1__5904__;
  __out_argv1__5904__.ndev_p = ((void *)(&ndev));
  __out_argv1__5904__.a_p = ((void *)(&a));
  __out_argv1__5904__.n_p = ((void *)(&n));
  __out_argv1__5904__.y_p = ((void *)(&y));
  __out_argv1__5904__.x_p = ((void *)(&x));
  XOMP_parallel_start(OUT__2__5904__,&__out_argv1__5904__,1,ndev,"/data/yy8/2013-8-multiple-gpu-work/benchmarks/axpy/axpy_ompacc.c",23);
  XOMP_parallel_end("/data/yy8/2013-8-multiple-gpu-work/benchmarks/axpy/axpy_ompacc.c",47);
}

static void OUT__2__5904__(void *__out_argv)
{
  double **x = (double **)(((struct OUT__2__5904___data *)__out_argv) -> x_p);
  double **y = (double **)(((struct OUT__2__5904___data *)__out_argv) -> y_p);
  int *n = (int *)(((struct OUT__2__5904___data *)__out_argv) -> n_p);
  double *a = (double *)(((struct OUT__2__5904___data *)__out_argv) -> a_p);
  int *ndev = (int *)(((struct OUT__2__5904___data *)__out_argv) -> ndev_p);
  double *_p_lx;
  double *_p_ly;
  int i;
  int devid = omp_get_thread_num();
  cudaSetDevice(devid);
  int remain = ( *n %  *ndev);
  int esize = ( *n /  *ndev);
  int partsize;
  int starti;
  int endi;
/* each of the first remain dev has one more element */
  if (devid < remain) {
    partsize = (esize + 1);
    starti = (partsize * devid);
  }
  else {
    partsize = esize;
    starti = ((esize * devid) + remain);
  }
  endi = (starti + partsize);
  printf("dev %d range: %d-%d\n",devid,starti,endi);
  _p_lx = ( *x + starti);
  _p_ly = ( *y + starti);
{
    double *_dev_lx;
    int _dev_lx_size = sizeof(double ) * (partsize - 0);
    _dev_lx = ((double *)(xomp_deviceMalloc(_dev_lx_size)));
    xomp_memcpyHostToDevice(((void *)_dev_lx),((const void *)_p_lx),_dev_lx_size);
    double *_dev_ly;
    int _dev_ly_size = sizeof(double ) * (partsize - 0);
    _dev_ly = ((double *)(xomp_deviceMalloc(_dev_ly_size)));
    xomp_memcpyHostToDevice(((void *)_dev_ly),((const void *)_p_ly),_dev_ly_size);
/* Launch CUDA kernel ... */
    int _threads_per_block_ = xomp_get_maxThreadsPerBlock();
    int _num_blocks_ = xomp_get_max1DBlock(partsize - 1 - 0 + 1);
    OUT__1__5904__<<<_num_blocks_,_threads_per_block_>>>( *a,partsize,_dev_lx,_dev_ly);
    xomp_freeDevice(_dev_lx);
    xomp_memcpyDeviceToHost(((void *)_p_ly),((const void *)_dev_ly),_dev_ly_size);
    xomp_freeDevice(_dev_ly);
  }
}
