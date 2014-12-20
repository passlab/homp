#include "axpy.h"
/* standard one-dev support */
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

#if defined (DEVICE_NVGPU_SUPPORT)
__global__ void OUT__3__5904__( long start_n,  long len_n,REAL a,REAL *_dev_x,REAL *_dev_y)
{
   long _dev_i = blockDim.x * blockIdx.x + threadIdx.x;
  if (_dev_i >= start_n && _dev_i <= start_n + len_n  - 1) {
    _dev_y[_dev_i] += (a * _dev_x[_dev_i]);
  }
}
#endif

struct OUT__3__5904__other_args {
	REAL a;
	long n;
	REAL *x;
	REAL *y;
};

/* called by the helper thread */
void OUT__3__5904__launcher (omp_offloading_t * off, void *args) {
    struct OUT__3__5904__other_args * iargs = (struct OUT__3__5904__other_args*) args; 
    long start_n, length_n;
    REAL a = iargs->a;
    REAL n = iargs->n;
    //omp_offloading_info_t * off_info = off->off_info;
    //printf("off: %X, off_info: %X, devseqid: %d\n", off, off_info, off->devseqid);
    omp_data_map_t * map_x = omp_map_get_map(off, iargs->x, -1);
    //omp_data_map_t * map_x = &off_info->data_map_info[0].maps[off->devseqid]; /* 0 means the map X */
    omp_data_map_t * map_y = omp_map_get_map(off, iargs->y, -1);
    //omp_data_map_t * map_y = &off_info->data_map_info[1].maps[off->devseqid]; /* 1 means the map Y */

    //printf("x: %X, x2: %X, y: %X, y2: %X\n", map_x, map_x2, map_y, map_y2);

    //omp_print_data_map(map_x);
    //omp_print_data_map(map_y);

    REAL * x = (REAL *)map_x->map_dev_ptr;
    REAL * y = (REAL *)map_y->map_dev_ptr;
    
    omp_loop_map_range(map_x, 0, -1, -1, &start_n, &length_n);
//    printf("devseqid: %d, start_n: %d, length_n: %d, x: %X, y: %X\n", off->devseqid, start_n, length_n, x, y);
    
	omp_device_type_t devtype = off->dev->type;
#if defined (DEVICE_NVGPU_SUPPORT)
	if (devtype == OMP_DEVICE_NVGPU) {
		int threads_per_team = omp_get_optimal_threads_per_team(off->dev);
		int teams_per_league = omp_get_optimal_teams_per_league(off->dev, threads_per_team, length_n);
        OUT__3__5904__<<<teams_per_league,threads_per_team, 0, off->stream->systream.cudaStream>>>(start_n, length_n,a,x,y);
	} else
#endif
	if (devtype == OMP_DEVICE_THSIM) {
		int i;
#pragma omp parallel for shared(y, x, a, start_n, length_n) private(i)
		for (i=start_n; i<start_n + length_n; i++) {
			y[i] += a*x[i];
	//		printf("x[%d]: %f, y[%d]: %f\n", i, x[i], i, y[i]);
		}
	} else {
		fprintf(stderr, "device type is not supported for this call\n");
		abort();
	}
}

REAL axpy_ompacc_mdev_v2(REAL *x, REAL *y,  long n,REAL a)
{
	double ompacc_time = read_timer_ms(); //read_timer_ms();
	
    /* get number of target devices specified by the programmers */
    int __num_target_devices__ = omp_get_num_active_devices(); /*XXX: = runtime or compiler generated code */
    
	omp_device_t *__target_devices__[__num_target_devices__];
	/**TODO: compiler generated code or runtime call to init the __target_devices__ array */
	int __i__;
	for (__i__ = 0; __i__ < __num_target_devices__; __i__++) {
		__target_devices__[__i__] = &omp_devices[__i__]; /* currently this is simple a copy of the pointer */
	}
	/**TODO: compiler generated code or runtime call to init the topology */
	omp_grid_topology_t __top__;	
	int __top_ndims__ = 1;
	int __top_dims__[__top_ndims__];
	int __top_periodic__[__top_ndims__]; 
	int __id_map__[__num_target_devices__];
	omp_grid_topology_init_simple (&__top__, __target_devices__, __num_target_devices__, __top_ndims__, __top_dims__, __top_periodic__, __id_map__);

	int __num_mapped_array__ = 2; /* XXX: need compiler output */

	omp_data_map_info_t __data_map_infos__[__num_mapped_array__];
		
	omp_data_map_info_t * __info__ = &__data_map_infos__[0];
	long x_dims[1]; x_dims[0] = n;
	omp_data_map_t x_maps[__num_target_devices__];
	omp_data_map_dist_t x_dist[1];
	omp_data_map_init_info_straight_dist("x", __info__, &__top__, x, 1, x_dims, sizeof(REAL), x_maps, OMP_DATA_MAP_TO, OMP_DATA_MAP_AUTO, x_dist, OMP_DATA_MAP_DIST_EVEN);

	__info__ = &__data_map_infos__[1];
	long y_dims[1]; y_dims[0] = n;
	omp_data_map_t y_maps[__num_target_devices__];
	omp_data_map_dist_t y_dist[1];
	omp_data_map_init_info_straight_dist("y", __info__, &__top__, y, 1, y_dims, sizeof(REAL), y_maps, OMP_DATA_MAP_TOFROM, OMP_DATA_MAP_AUTO, y_dist, OMP_DATA_MAP_DIST_EVEN);
	
	struct OUT__3__5904__other_args args;
	args.a = a;
	args.n = n;
	args.x = x;
	args.y = y;
	omp_offloading_info_t __offloading_info__;
	__offloading_info__.offloadings = (omp_offloading_t *) alloca(sizeof(omp_offloading_t) * __num_target_devices__);
	/* we use universal args and launcher because axpy can do it */
	omp_offloading_init_info ("axpy kernel", &__offloading_info__, &__top__, __target_devices__, 0, OMP_OFFLOADING_DATA_CODE, __num_mapped_array__, __data_map_infos__, OUT__3__5904__launcher, &args);

#if 0
	/* we could specify dev-specific args and kernel_launcher */
	struct OUT__3__5904__other_args args[__num_target_devices__];
	for (__i__ = 0; __i__ < __num_target_devices__; __i__++) {
		args[i].a = a;
		args[i].n = n;
		__offloading_info__.offloadings[i].args = &args[i];
		__offloading_info__.offloadings[i].kernel_launcher = OUT__3__5904__launcher;
	}
#endif
	
	/*********** NOW notifying helper thread to work on this offload ******************/
#if DEBUG_MSG
	 printf("=========================================== offloading to %d targets ==========================================\n", __num_target_devices__);
#endif
	/* here we do not need sync start */
	omp_offloading_start(&__offloading_info__);
	ompacc_time = read_timer_ms() - ompacc_time;
	omp_offloading_clear_report_info(&__offloading_info__);
	double cpu_total = ompacc_time;
	return cpu_total;
}
