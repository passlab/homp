#include "sum.h"
#include "homp.h"

#if 0
/* v1: explicit distribution of both data and loop:
 * the y[0:n], and x[0:n] will be evenly distributed among the ndev devices,
 * scalars such as a and n will each have a mapped copy in all the devices, loop will also be evenly distributed */
REAL sum_mdev_v1(long n, REAL* x, REAL* y) {
	REAL result;
#pragma omp parallel target device (*) map(tofrom: y[0:n] distribute(BLOCK)) map(to: x[0:n] distribute(BLOCK),n) map(from:result)
#pragma omp parallel for shared(x, y, n) distribute(BLOCK) reduction(+:result)
	for (i = 0; i < n; ++i)
    	result += y[i] * x[i];
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

struct sum_dev_kernel_args {
	REAL *result;
	long n;
	REAL *x;
	REAL *y;
};

/**
 * The demultiplexer for multiple device, called by the helper thread
 **/
static void sum_dev_kernel_demux(omp_offloading_t *off, void *args) {
    struct sum_dev_kernel_args * iargs = (struct sum_dev_kernel_args*) args;
    long start_n, length_n;
    REAL result; /* reduction var array */
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
    
    long abs_off = omp_loop_get_range(off, 0, &start_n, &length_n);

	//long omp_loop_get_range(omp_offloading_t * off, int loop_depth, long * start, long* length) {
    //printf("devseqid: %d, start_n: %d (%d), length_n: %d, x: %X, y: %X\n", off->devseqid, start_n, abs_off, length_n, x, y);
	omp_device_type_t devtype = off->dev->type;
	if (devtype == OMP_DEVICE_NVGPU) {
#if defined (DEVICE_NVGPU_CUDA_SUPPORT)
		sum_nvgpu_cuda_wrapper(off, start_n, length_n, x, y, &result);
#endif
	} else if (devtype == OMP_DEVICE_ITLMIC) { /* TODO with OpenCL */
#if defined(DEVICE_ITLMIC_SUPPORT)
		sum_itlmic_wrapper(off, start_n, length_n, x, y, &result);
#endif
	} else if (devtype == OMP_DEVICE_THSIM || devtype == OMP_DEVICE_HOSTCPU) {
		sum_cpu_omp_wrapper(off, start_n, length_n, x, y, &result);
	} else {
		fprintf(stderr, "device type is not supported for this call\n");
		abort();
	}
	iargs->result[off->devseqid] += result;
}

double sum_ompacc_mdev(int ndevs, int *targets, long n, REAL *x, REAL *y, REAL *result) {
	double ompacc_init_time = read_timer_ms();

	omp_grid_topology_t * __top__ = omp_grid_topology_init(ndevs, targets, 1);
	/* init other infos (dims, periodic, idmaps) of top if needed */

	int __num_maps__ = 2; /* XXX: need compiler output */
	int i;

	/* we use universal args and launcher because sum can do it */
	struct sum_dev_kernel_args args;
	REAL device_result[ndevs];
	for (i=0; i<ndevs; i++) device_result[i] = 0.0;
	args.result = device_result;
	args.n = n;
	args.x = x;
	args.y = y;
	omp_offloading_info_t *__off_info__ = omp_offloading_init_info("sum_kernel", __top__, 1, OMP_OFFLOADING_DATA_CODE,
																   __num_maps__, sum_dev_kernel_demux, &args, 1);
	omp_offloading_append_profile_per_iteration(__off_info__, 2, 2, 1);

	omp_data_map_info_t *__x_map_info__ = &__off_info__->data_map_info[0];
	omp_data_map_init_info("x", __x_map_info__, __off_info__, x, 1, sizeof(REAL), OMP_DATA_MAP_TO, OMP_DATA_MAP_AUTO);
	omp_data_map_info_set_dims_1d(__x_map_info__, n);

	omp_data_map_info_t *__y_map_info__ = &__off_info__->data_map_info[1];
	omp_data_map_init_info("y", __y_map_info__, __off_info__, y, 1, sizeof(REAL), OMP_DATA_MAP_TO, OMP_DATA_MAP_AUTO);
	omp_data_map_info_set_dims_1d(__y_map_info__, n);

#if 0
    /* test BLOCK policy */
	omp_data_map_dist_init_info(__x_map_info__, 0, OMP_DIST_POLICY_BLOCK, 0, n, 0, 0);
	omp_data_map_dist_init_info(__y_map_info__, 0, OMP_DIST_POLICY_BLOCK, 0, n, 0, 0);
	omp_loop_dist_init_info(__off_info__, 0, OMP_DIST_POLICY_BLOCK, 0, n, 0, 0);
	printf("version 2: BLOCK dist policy for x, y, and loop\n");

	/* test loop binding to data */
	omp_data_map_dist_init_info(__x_map_info__, 0, OMP_DIST_POLICY_BLOCK, 0, n, 0, 0);
	//omp_data_map_dist_init_info(__y_map_info__, 0, OMP_DIST_POLICY_BLOCK, 0, n, 0);
	omp_data_map_dist_align_with_data_map(__y_map_info__, 0, 0, __x_map_info__, 0);
	omp_loop_dist_align_with_data_map(__off_info__, 0, 0, __x_map_info__, 0);
	printf("version 3: BLOCK dist policy for x and y, and loop dist binds with x\n");
#endif

	omp_loop_dist_init_info(__off_info__, 0, LOOP_DIST_POLICY, 0, n, LOOP_DIST_CHUNK_SIZE, 0);
	omp_data_map_dist_align_with_loop(__x_map_info__, 0, 0, __off_info__, 0);
	omp_data_map_dist_align_with_loop(__y_map_info__, 0, 0, __off_info__, 0);

	/*********** NOW notifying helper thread to work on this offload ******************/
#if DEBUG_MSG
	 printf("=========================================== offloading to %d targets ==========================================\n", __num_targets__);
#endif
	ompacc_init_time = read_timer_ms() - ompacc_init_time;
	//  printf("init time: %fs\n", ompacc_init_time);
	/* here we do not need sync start */
	double off_total = read_timer_ms();
	int total_its = 1;
	for (i =0; i < total_its; i++) {
		omp_offloading_start(__off_info__);
	}
	*result = 0.0;
	for (i = 0; i<ndevs; i++) {
		*result += device_result[i];
	}

	off_total = (read_timer_ms() - off_total)/total_its;
#if defined (OMP_BREAKDOWN_TIMING)
	omp_print_map_info(__x_map_info__);
	omp_print_map_info(__y_map_info__);
	omp_offloading_info_report_profile(__off_info__);
#endif

	omp_offloading_fini_info(__off_info__);
	omp_grid_topology_fini(__top__);
	off_total += ompacc_init_time;
	return off_total;
}



