#include "axpy.h"
/* standard one-dev support */
#include "libxomp.h" 
#include "xomp_cuda_lib_inlined.cu" 
#include "homp.h"

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

#if 0
/* version 2: leveraging the omp worksharing */
void axpy_mdev_v2(REAL* x, REAL* y, int n, REAL a) {
  int ndev = omp_get_num_devices();

#pragma omp target device (0:ndev) map(tofrom: y[0:n]>>(:)) map(to: x[0:n]>>(:),a,n)
#pragma omp parallel for shared(x, y, n, a) private(i)
/* in this example, the y[0:n], and x[0:n] will be evenly distributed among the ndev devices, scalars such as a and n will each have a mapped copy in all the devices */
  for (i = 0; i < n; ++i)
    y[i] += a * x[i];
}
#endif

void axpy_mdev_v2(double *x, double *y, int n,double a)
{
    int ndev = omp_get_num_devices();
  
    /* get number of target devices specified by the programmers */
    int __num_target_devices__; /*XXX: = runtime or compiler generated code */
    int __num_mapped_variables__; /* XXX: need compiler output */
     
    /* declare the streams used for the async launching
     * For each device, there is one stream
     */
    cudaStream_t __dev_stream__[__num_target_devices__];

    /* for each mapped variables on each device, we need a data_map for this,
     * our compiler should help to create an unique id for each of the mapped variables, which will
     * be used to access the map array
     *
     * in this example, x is 0, y is 1;
     */
    omp_data_map_t __data_maps__[__num_target_devices__][__num_mapped_variables__];
    int __ndev_i__;
    for (__ndev_i__ = 0; __ndev_i__<__num_target_devices__; __ndev_i__++) {
        cudaSetDevice(__ndev_i__);
        cudaStreamCreate(&__dev_stream__[__ndev_i__]);

        /***************** for each mapped variable has to and tofrom, if it has region mapped to this __ndev_i__ id, we need code here *******************************/
        omp_data_map_t * __dev_map_x__ = __data_maps__[__ndev_i__][0]; /* 0 is given by compiler here */
        __dev_map_x__->device_id = __ndev_i__; //omp_get_device(__ndev_i__)->sysid;
        __dev_map_x__->map_type = OMP_MAP_TO;  /* from compiler */
        __dev_map_x__->source_ptr = x;
        __dev_map_x__->dim_x = n;
        __dev_map_x__->dim_y = 1;
        __dev_map_x__->dim_z = 1;
        __dev_map_x__->sizeof_element = sizeof(double);

        __dev_map_x__->map_offset_x = __ndev_i__ * n/__num_target_devices__; /* chunking n into __ndev_i__ pieces, from compiler */
        __dev_map_x__->map_offset_y = 0;/* from compiler */
        __dev_map_x__->map_offset_z = 0;/* from compiler */

        __dev_map_x__->map_dim_x = n/__num_target_devices__;/* from compiler */
        __dev_map_x__->map_dim_y = 1;
        __dev_map_x__->map_dim_z = 1;

        omp_map_buffer(__dev_map_x__, 0);
        __dev_map_x__->stream = &__dev_stream__[__ndev_i__];

        omp_deviceMalloc_memcpyHostToDeviceAsync(__dev_map_x__);
        /*************************************************************************************************************************************************************/

		/***************************************************************** for y *********************************************************************/
        omp_data_map_t * __dev_map_y__ = __data_maps__[__ndev_i__][0]; /* 0 is given by compiler here */
        __dev_map_y__->device_id = __ndev_i__; //omp_get_device(__ndev_i__)->sysid;
		__dev_map_y__->map_type = OMP_MAP_TOFROM; /* from compiler */
		__dev_map_y__->source_ptr = y;
		__dev_map_y__->dim_x = n;
		__dev_map_y__->dim_y = 1;
		__dev_map_y__->dim_z = 1;
		__dev_map_y__->sizeof_element = sizeof(double);

		__dev_map_y__->map_offset_x = __ndev_i__ * n / __num_target_devices__; /* chunking n into __ndev_i__ pieces, from compiler */
		__dev_map_y__->map_offset_y = 0;/* from compiler */
		__dev_map_y__->map_offset_z = 0;/* from compiler */

		__dev_map_y__->map_dim_x = n / __num_target_devices__;/* from compiler */
		__dev_map_y__->map_dim_y = 1;
		__dev_map_y__->map_dim_z = 1;

		omp_map_buffer(__dev_map_y__, 0);
        __dev_map_y__->stream = &__dev_stream__[__ndev_i__];


		omp_deviceMalloc_memcpyHostToDeviceAsync(__dev_map_y__);
		/***************************************************************************************************************************************************************/
        /* Launch CUDA kernel ... */
        int _threads_per_block_ = xomp_get_maxThreadsPerBlock();
        int _num_blocks_ = xomp_get_max1DBlock(n - 1 - 0 + 1);
        OUT__3__5904__<<<_num_blocks_,_threads_per_block_, __dev_stream__[__ndev_i__])>>>(n,a,__dev_map_x__->map_dev_ptr,__dev_map_y__->map_dev_ptr);

        /***************************************************************************************************************************************************/
        /****************** for each from and tofrom, we need call to DeviceToHost memcpy */
        omp_memcpyDeviceToHostAsync(__dev_map_y__);
    }

    omp_postACCKernel(__num_target_devices__, __num_mapped_variables__, __dev_stream__, __data_maps__);
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
