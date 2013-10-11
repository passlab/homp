#include "axpy.h"
/* standard one-dev support */
#include "libxomp.h" 
#include "xomp_cuda_lib_inlined.cu" 
#include "homp.h"

#if 0
void axpy_mdev_v2(REAL* x, REAL* y,  long n, REAL a) {

#pragma omp target device (:) map(tofrom: y[0:n]>>(:)) map(to: x[0:n]>>(:),a,n)
#pragma omp parallel for shared(x, y, n, a) private(i) dist_iteration match_range x[:]
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

__global__ void OUT__3__5904__( long start_n,  long len_n,double a,double *_dev_x,double *_dev_y)
{
   long _dev_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (_dev_i >= start_n && _dev_i <= start_n + len_n  - 1) {
    _dev_y[_dev_i] += (a * _dev_x[_dev_i]);
  }
}


void axpy_ompacc_mdev_v2(double *x, double *y,  long n,double a)
{
    /* get number of target devices specified by the programmers */
    int __num_target_devices__ = omp_get_num_active_devices(); /*XXX: = runtime or compiler generated code */
    
    printf("use %d target devices\n", __num_target_devices__);

	omp_device_t *__target_devices__[__num_target_devices__];
	/**TODO: compiler generated code or runtime call to init the __target_devices__ array */
	int __i__;
	for (__i__ = 0; __i__ < __num_target_devices__; __i__++) {
		__target_devices__[__i__] = &omp_devices[__i__]; /* currently this is simple a copy of the pointer */
	}
	/**TODO: compiler generated code or runtime call to init the topology */
	int __top_ndims__ = 1;
	int __top_dims__[__top_ndims__];
	omp_factor(__num_target_devices__, __top_dims__, __top_ndims__);
	int __top_periodic__[__top_ndims__]; __top_periodic__[0] = 0;
	omp_grid_topology_t __topology__={__num_target_devices__, __top_ndims__, __top_dims__, __top_periodic__};
	omp_grid_topology_t *__topp__ = &__topology__;

	int __num_mapped_variables__ = 2; /* XXX: need compiler output */

	omp_stream_t __dev_stream__[__num_target_devices__]; /* need to change later one for omp_stream_t struct */
	omp_data_map_info_t __data_map_infos__[__num_mapped_variables__];

	omp_data_map_info_t * __info__ = &__data_map_infos__[0];
	omp_data_map_init_info(__info__, __topp__, x, sizeof(double), OMP_MAP_TO, n, 1, 1);
	__info__->maps = (omp_data_map_t **)alloca(sizeof(omp_data_map_t *) * __num_target_devices__);

	__info__ = &__data_map_infos__[1];
	omp_data_map_init_info(__info__, __topp__, y, sizeof(double), OMP_MAP_TOFROM, n, 1, 1);
	__info__->maps = (omp_data_map_t **)alloca(sizeof(omp_data_map_t *) * __num_target_devices__);

	omp_data_map_t __data_maps__[__num_target_devices__][__num_mapped_variables__];
	for (__i__ = 0; __i__ < __num_target_devices__; __i__++) {
		omp_device_t * __dev__ = __target_devices__[__i__];
		omp_set_current_device(__dev__);
		omp_init_stream(__dev__, &__dev_stream__[__i__]);

		/***************** for each mapped variable has to and tofrom, if it has region mapped to this __ndev_i__ id, we need code here *******************************/
		omp_data_map_t * __dev_map_x__ = &__data_maps__[__i__][0]; /* 0 is given by compiler here */
		omp_data_map_init_map(__dev_map_x__, &__data_map_infos__[0], __i__, __dev__, &__dev_stream__[__i__]);
		omp_data_map_do_even_map(__dev_map_x__, 0, __topp__, 0, __i__);

		omp_map_buffer(__dev_map_x__, 0); /* even a 2-d array, but since we are doing row-major partition, no need to marshalled data */

		omp_memcpyHostToDeviceAsync(__dev_map_x__);
		omp_print_data_map(__dev_map_x__);
		/*************************************************************************************************************************************************************/

		/***************************************************************** for u *********************************************************************/
		omp_data_map_t * __dev_map_y__ = &__data_maps__[__i__][1]; /* 1 is given by compiler here */
		omp_data_map_init_map(__dev_map_y__, &__data_map_infos__[1], __i__, __dev__, &__dev_stream__[__i__]);

		omp_data_map_do_even_map(__dev_map_y__, 0, __topp__, 0, __i__);

		omp_map_buffer(__dev_map_y__, 0); /* column major, marshalling needed */

		omp_memcpyHostToDeviceAsync(__dev_map_y__);
		omp_print_data_map(__dev_map_y__);

		/***************************************************************************************************************************************************************/
        /* Launch CUDA kernel ... */
        int _threads_per_block_ = xomp_get_maxThreadsPerBlock();
        int _num_blocks_ = xomp_get_max1DBlock(n - 1 - 0 + 1);
        /* in this example, this information could be provided by compiler analysis, but we can also retrive this from runtime as a more
         * general solution */
        long start_n, length_n;
        omp_loop_map_range(__dev_map_x__, 0, -1, -1, &start_n, &length_n);
        /* the argu for this function should be the original pointer (x in this example) and the runtime should search and retrieve the
         * device map object
         */
        printf("device: %d, range: %d:%d\n", __i__, start_n, length_n);

        OUT__3__5904__<<<_num_blocks_,_threads_per_block_, 0, __dev_stream__[__i__].systream.cudaStream>>>(start_n, length_n,a,(double *)__dev_map_x__->map_dev_ptr, (double *)__dev_map_y__->map_dev_ptr);

        /***************************************************************************************************************************************************/
        /****************** for each from and tofrom, we need call to DeviceToHost memcpy */
        omp_memcpyDeviceToHostAsync(__dev_map_y__);
    }

    omp_sync_cleanup(__num_target_devices__, __num_mapped_variables__, __dev_stream__, &__data_maps__[0][0]);

}
