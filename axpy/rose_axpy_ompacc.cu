#include "axpy.h"
/* standard one-dev support */
#include "libxomp.h" 
#include "xomp_cuda_lib_inlined.cu" 

__global__ void OUT__3__4961__(void *__out_argv)
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
    OUT__3__4961__<<<_num_blocks_,_threads_per_block_>>>(n,a,_dev_x,_dev_y);
    xomp_freeDevice(_dev_x);
    xomp_memcpyDeviceToHost(((void *)y),((const void *)_dev_y),_dev_y_size);
    xomp_freeDevice(_dev_y);
  }
}
/* version 1: use omp parallel, i.e. each host thread responsible for one dev */

__global__ void OUT__1__4961__(double a,int partsize,double *_dev_x,double *_dev_y)
{
  int _p_i;
  int _dev_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (_dev_i >= 0 && _dev_i <= partsize - 1) {
    _dev_y[_dev_i] += (a * _dev_x[_dev_i]);
  }
}

struct OUT__2__4961___data 
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
static void OUT__2__4961__(void *__out_argv);

void axpy_mdev_v1(double *x,double *y,int n,double a)
{
  int ndev = omp_get_num_devices();
  struct OUT__2__4961___data __out_argv1__4961__;
  __out_argv1__4961__.ndev_p = ((void *)(&ndev));
  __out_argv1__4961__.a_p = ((void *)(&a));
  __out_argv1__4961__.n_p = ((void *)(&n));
  __out_argv1__4961__.y_p = ((void *)(&y));
  __out_argv1__4961__.x_p = ((void *)(&x));
  XOMP_parallel_start(OUT__2__4961__,&__out_argv1__4961__,1,ndev);
  XOMP_parallel_end();
}

static void OUT__2__4961__(void *__out_argv)
{
  int i;
  int devid = omp_get_thread_num();
  int remain = ( *__out_argv %  *__out_argv);
  int esize = ( *__out_argv /  *__out_argv);
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
{
    double *_dev_x;
    int _dev_x_size = sizeof(double ) * (endi - starti);
    _dev_x = ((double *)(xomp_deviceMalloc(_dev_x_size)));
    xomp_memcpyHostToDevice(((void *)_dev_x),((const void *)( *__out_argv)),_dev_x_size);
    double *_dev_y;
    int _dev_y_size = sizeof(double ) * (endi - starti);
    _dev_y = ((double *)(xomp_deviceMalloc(_dev_y_size)));
    xomp_memcpyHostToDevice(((void *)_dev_y),((const void *)( *__out_argv)),_dev_y_size);
/* Launch CUDA kernel ... */
    int _threads_per_block_ = xomp_get_maxThreadsPerBlock();
    int _num_blocks_ = xomp_get_max1DBlock(partsize - 1 - 0 + 1);
    OUT__1__4961__<<<_num_blocks_,_threads_per_block_>>>( *__out_argv,partsize,_dev_x,_dev_y);
    xomp_freeDevice(_dev_x);
    xomp_memcpyDeviceToHost(((void *)( *__out_argv)),((const void *)_dev_y),_dev_y_size);
    xomp_freeDevice(_dev_y);
  }
}
