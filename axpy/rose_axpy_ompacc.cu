#include "axpy.h"
/* standard one-dev support */
#include "libxomp.h" 
#include "xomp_cuda_lib_inlined.cu" 
#include "homp.h"

#if 0
void axpy_mdev_v2(REAL* x, REAL* y, int n, REAL a) {

#pragma omp target device (:) map(tofrom: y[0:n]>>(:)) map(to: x[0:n]>>(:),a,n)
#pragma omp parallel for shared(x, y, n, a) private(i) map_range x[:]
/* in this example, the y[0:n], and x[0:n] will be evenly distributed among the ndev devices, scalars such as a and n will each have a mapped copy in all the devices */
  for (i = 0; i < n; ++i)
    y[i] += a * x[i];
}

/* NOTE: the compiler needs to do the analysis for multiple pragma(s) and loop nest. The x[:] in the mapped_range x[:] should
 * be in the previous pragma's map clause
 *
 * Thus this requires the runtime to keep track of the mapped variables and all the details. In some examples, those information could
 * be provided by code-generation of compiler. but in other cases, e.g. the orphaned directive, one need to retrieve from the runtime
 * to get those information. Thus in current implementation, we simply keep all the information in the runtime, regardless of using it
 * or not.
 */
#endif

__global__ void OUT__3__5904__(int start_n, int len_n,double a,double *_dev_x,double *_dev_y)
{
  int _dev_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (_dev_i >= start_n && _dev_i <= start_n + len_n  - 1) {
    _dev_y[_dev_i] += (a * _dev_x[_dev_i]);
  }
}


void axpy_ompacc_mdev_v2(double *x, double *y, int n,double a)
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
     * in this example, x is 0, y is 1;
     */
    omp_data_map_t __data_maps__[__num_target_devices__][__num_mapped_variables__];
    int __ndev_i__;
    for (__ndev_i__ = 0; __ndev_i__<__num_target_devices__; __ndev_i__++) {
        cudaSetDevice(__ndev_i__);
        cudaStreamCreate(&__dev_stream__[__ndev_i__]);

        /***************** for each mapped variable has to and tofrom, if it has region mapped to this __ndev_i__ id, we need code here *******************************/
        omp_data_map_t * __dev_map_x__ = &__data_maps__[__ndev_i__][0]; /* 0 is given by compiler here */
        __dev_map_x__->device_id = __ndev_i__; //omp_get_device(__ndev_i__)->sysid;
        __dev_map_x__->map_type = OMP_MAP_TO;  /* from compiler */
        __dev_map_x__->source_ptr = x;
        __dev_map_x__->dim[0] = n;
        __dev_map_x__->dim[1] = 1;
        __dev_map_x__->dim[2] = 1;
        __dev_map_x__->sizeof_element = sizeof(double);

        __dev_map_x__->map_offset[0] = __ndev_i__ * n/__num_target_devices__; /* chunking n into __ndev_i__ pieces, from compiler */
        __dev_map_x__->map_offset[1] = 0;/* from compiler */
        __dev_map_x__->map_offset[2] = 0;/* from compiler */

        __dev_map_x__->map_dim[0] = n/__num_target_devices__;/* from compiler */
        __dev_map_x__->map_dim[1] = 1;
        __dev_map_x__->map_dim[2] = 1;

        omp_map_buffer(__dev_map_x__, 0);
        __dev_map_x__->stream = &__dev_stream__[__ndev_i__];

        omp_deviceMalloc_memcpyHostToDeviceAsync(__dev_map_x__);
        /*************************************************************************************************************************************************************/

		/***************************************************************** for y *********************************************************************/
        omp_data_map_t * __dev_map_y__ = &__data_maps__[__ndev_i__][0]; /* 0 is given by compiler here */
        __dev_map_y__->device_id = __ndev_i__; //omp_get_device(__ndev_i__)->sysid;
		__dev_map_y__->map_type = OMP_MAP_TOFROM; /* from compiler */
		__dev_map_y__->source_ptr = y;
		__dev_map_y__->dim[0] = n;
		__dev_map_y__->dim[1] = 1;
		__dev_map_y__->dim[2] = 1;
		__dev_map_y__->sizeof_element = sizeof(double);

		__dev_map_y__->map_offset[0] = __ndev_i__ * n / __num_target_devices__; /* chunking n into __ndev_i__ pieces, from compiler */
		__dev_map_y__->map_offset[1] = 0;/* from compiler */
		__dev_map_y__->map_offset[2] = 0;/* from compiler */

		__dev_map_y__->map_dim[0] = n / __num_target_devices__;/* from compiler */
		__dev_map_y__->map_dim[1] = 1;
		__dev_map_y__->map_dim[2] = 1;

		omp_map_buffer(__dev_map_y__, 0);
        __dev_map_y__->stream = &__dev_stream__[__ndev_i__];

		omp_deviceMalloc_memcpyHostToDeviceAsync(__dev_map_y__);
		/***************************************************************************************************************************************************************/
        /* Launch CUDA kernel ... */
        int _threads_per_block_ = xomp_get_maxThreadsPerBlock();
        int _num_blocks_ = xomp_get_max1DBlock(n - 1 - 0 + 1);
        /* in this example, this information could be provided by compiler analysis, but we can also retrive this from runtime as a more
         * general solution */
        int start_n, length_n;
        omp_loop_map_range(__dev_map_x__, 1, -1, -1, &start_n, &length_n);
        /* the argu for this function should be the original pointer (x in this example) and the runtime should search and retrieve the
         * device map object
         */

        OUT__3__5904__<<<_num_blocks_,_threads_per_block_, 0, __dev_stream__[__ndev_i__]>>>(start_n, length_n,a,(double *)__dev_map_x__->map_dev_ptr, (double *)__dev_map_y__->map_dev_ptr);

        /***************************************************************************************************************************************************/
        /****************** for each from and tofrom, we need call to DeviceToHost memcpy */
        omp_memcpyDeviceToHostAsync(__dev_map_y__);
    }

    omp_postACCKernel(__num_target_devices__, __num_mapped_variables__, __dev_stream__, (omp_data_map_t*)__data_maps__);
}
