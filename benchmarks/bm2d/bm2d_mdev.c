#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include "homp.h"
#include "bm2d.h"

void bm2d_omp_mdev_launcher(omp_offloading_t * off, void *args) {
    struct bm2d_off_args * iargs = (struct bm2d_off_args*) args;
    long n = iargs->n;
    long m = iargs->m;
    int maxwin = iargs->maxwin;
    int num_its = iargs->num_its;
    long u_dimX = iargs->u_dimX;
    long u_dimY = iargs->u_dimY;
    int coeff_dimX = iargs->coeff_dimX;

    omp_data_map_t * map_u = omp_map_get_map(off, iargs->u, -1); /* 1 is for the map u */
    omp_data_map_t * map_uold = omp_map_get_map(off, iargs->uold, -1); /* 2 is for the map uld */
    omp_data_map_t * map_coeff = omp_map_get_map(off, iargs->coeff, -1); /* 2 is for the map uld */

    REAL * u = (REAL*) map_u->map_dev_wextra_ptr;
    REAL * uold = (REAL*) map_uold->map_dev_wextra_ptr;
    REAL *coeff = (REAL*) map_coeff->map_dev_wextra_ptr;
    coeff = coeff + (2*maxwin+1) * maxwin + maxwin; /* TODO this should be a call to map a host-side address to dev-side address*/

#if CORRECTNESS_CHECK
    printf("kernel launcher: u: %X, uold: %X\n", u, uold);
    print_array("u in device: ", "udev", u, n, m);
    print_array("uold in device: ", "uolddev", uold, n, m);
#endif

    long offset;
    long start;
    long len;
    offset = omp_loop_get_range(off, 0, &start, &len);

#if 0
    /* col-wise dist */
    omp_loop_get_range(off, 0, &start, &len);
    /* row/col-wise dist */
    omp_loop_get_range(off, 0, &start, &len); /* todo */
    omp_loop_get_range(off, 0, &start, &len); /* todo */
#endif

    omp_device_type_t devtype = off->dev->type;
    void (*bm2d_kernel_wrapper)(omp_offloading_t *off, long start_n, long len_n, long n, long m, long u_dimX,
                                long u_dimY, REAL *u, REAL *uold, int maxwin, int coeff_dimX, REAL *coeff);
    if (devtype == OMP_DEVICE_NVGPU) {
#if defined (DEVICE_NVGPU_CUDA_SUPPORT)
        bm2d_kernel_wrapper = bm2d_nvgpu_cuda_wrapper;
#endif
    } else if (devtype == OMP_DEVICE_ITLMIC) {
#if defined(DEVICE_ITLMIC_SUPPORT)
        bm2d_kernel_wrapper = bm2d_itlmic_wrapper;
#endif
    } else if (devtype == OMP_DEVICE_THSIM || devtype == OMP_DEVICE_HOSTCPU) {
        bm2d_kernel_wrapper = bm2d_cpu_omp_wrapper;
    } else {
        fprintf(stderr, "device type is not supported for this call\n");
        abort();
    }

    REAL * uu;
    REAL * uuold;
    long it = iargs->it_num; /* iteration */
    if (it % 2 == 0) {
        uu = u;
        uuold = uold;
    } else {
        uu = uold;
        uuold = u;
    }
    bm2d_kernel_wrapper(off, start, len, n, m, u_dimX, u_dimY, uu, uuold, maxwin, coeff_dimX, coeff);
}

double bm2d_omp_mdev(int ndevs, int *targets, long n, long m, REAL *u, int maxwin, REAL *coeff, int num_its) {
    long u_dimX = n + 2 * maxwin;
    long u_dimY = m + 2 * maxwin;
    int coeff_dimX = 2*maxwin+1;
    REAL * coeff_center = coeff + (2*maxwin+1) * maxwin + maxwin; /* let coeff point to the center element */
    REAL *uold = (REAL *) omp_unified_malloc(sizeof(REAL) * u_dimX * u_dimY);
    memcpy(uold, u, sizeof(REAL)*u_dimX * u_dimY);
    //print_array("Before offloading", "u", u, u_dimX, u_dimY);

    double off_init_time = read_timer_ms();

    /**************************************** dist-specific *****************************************/
    int __top_ndims__ = 1;
    /* TODO: to use row/col-wise dist, __top_ndims__ should be set to 2 */
    omp_grid_topology_t * __top__ = omp_grid_topology_init(ndevs, targets, __top_ndims__);
    /* init other infos (dims, periodic, idmaps) of top if needed */

    int __num_maps__ = 3; /* u, uold and the coeff */ /* XXX: need compiler output */

    /* stencil kernel offloading */
    struct bm2d_off_args off_args;
    off_args.n = n; off_args.m = m; off_args.u = u; off_args.maxwin = maxwin; off_args.coeff = coeff; off_args.num_its = num_its;
    off_args.uold = uold; off_args.coeff_center = coeff_center; off_args.coeff_dimX = coeff_dimX; off_args.u_dimX = u_dimX; off_args.u_dimY = u_dimY;
    omp_offloading_info_t * __off_info__ = omp_offloading_init_info("bm2d_kernel", __top__, 1, OMP_OFFLOADING_DATA_CODE, __num_maps__,
                                     bm2d_omp_mdev_launcher, &off_args, 1);
    omp_offloading_append_profile_per_iteration(__off_info__, 13*u_dimY, 7, 1);

    //printf("data copy off: %X, bm2d off: %X\n", __copy_data_off_info__, __off_info__);

    /* u map info */
    omp_data_map_info_t *__u_map_info__ = &__off_info__->data_map_info[0];
    omp_data_map_init_info("u", __u_map_info__, __off_info__, u, 2, sizeof(REAL), OMP_DATA_MAP_TOFROM, OMP_DATA_MAP_AUTO);
    omp_data_map_info_set_dims_2d(__u_map_info__, u_dimX, u_dimY);

    /* uold map info */
    omp_data_map_info_t *__uold_map_info__ = &__off_info__->data_map_info[1];
    omp_data_map_init_info("uold", __uold_map_info__, __off_info__, uold, 2, sizeof(REAL), OMP_DATA_MAP_TO, OMP_DATA_MAP_AUTO);
    omp_data_map_info_set_dims_2d(__uold_map_info__, u_dimX, u_dimY);

    /* coeff map info */
    omp_data_map_info_t *__coeff_map_info__ = &__off_info__->data_map_info[2];
    omp_data_map_init_info("coeff", __coeff_map_info__, __off_info__, coeff, 2, sizeof(REAL), OMP_DATA_MAP_TO, OMP_DATA_MAP_AUTO);
    omp_data_map_info_set_dims_2d(__coeff_map_info__, coeff_dimX, coeff_dimX);

    omp_data_map_dist_init_info(__coeff_map_info__, 0, OMP_DIST_POLICY_FULL, 0, coeff_dimX, 0, 0);
    omp_data_map_dist_init_info(__coeff_map_info__, 1, OMP_DIST_POLICY_FULL, 0, coeff_dimX, 0, 0);
    /**************************************** dist-specific *****************************************/

    /* row-wise distribution */
#if 0
    /* BLOCK_BLOCK */
    omp_data_map_dist_init_info(__u_map_info__, 0, OMP_DIST_POLICY_BLOCK, maxwin, n, 0, 0);
    omp_loop_dist_init_info(__off_info__, 0, OMP_DIST_POLICY_BLOCK, 0, n, 0, 0);
    //printf("BLOCK dist policy for arrays and loop dist\n");
    /* BLOCK_ALIGN */
    omp_data_map_dist_init_info(__u_map_info__, 0, OMP_DIST_POLICY_BLOCK, maxwin, n, 0, 0);
    omp_loop_dist_align_with_data_map(__off_info__, 0, 0, __u_map_info__, 0);
    //printf("BLOCK dist policy for arrays, and loop dist align with array A row dist\n");
#endif

    /* AUTO_ALIGN */
    omp_loop_dist_init_info(__off_info__, 0, LOOP_DIST_POLICY, 0, n, LOOP_DIST_CHUNK_SIZE, 0);
    omp_data_map_dist_align_with_loop(__u_map_info__, 0, maxwin, __off_info__, 0);
    //printf("AUTO dist policy for loop dist and array align with loops\n");

    /* used by all row-wise dist */
    omp_data_map_dist_init_info(__u_map_info__, 1, OMP_DIST_POLICY_FULL, 0, u_dimY, 0, 0);
    omp_map_add_halo_region(__u_map_info__, 0, maxwin, maxwin, OMP_DIST_HALO_EDGING_REFLECTING);
    omp_data_map_dist_align_with_data_map_with_halo(__uold_map_info__, OMP_ALL_DIMENSIONS, OMP_ALIGNEE_OFFSET, __u_map_info__, OMP_ALL_DIMENSIONS);

#if 0
    /* col-wise distribution */
    omp_data_map_dist_init_info(__u_map_info__, 0, OMP_DIST_POLICY_FULL, maxwin, n, 0, 0);
    omp_data_map_dist_init_info(__u_map_info__, 1, OMP_DIST_POLICY_BLOCK, maxwin, n, 0, 0);
    omp_map_add_halo_region(__u_map_info__, 0, maxwin, maxwin, OMP_DIST_HALO_EDGING_REFLECTING);
    omp_data_map_dist_align_with_data_map_with_halo(__uold_map_info__, OMP_ALL_DIMENSIONS, 0, __u_map_info__, OMP_ALL_DIMENSIONS);
    omp_loop_dist_init_info(__off_info__, 1, OMP_DIST_POLICY_BLOCK, 0, m, 0, 0);

    /* row/col-wise distribution */
    omp_data_map_dist_init_info(__u_map_info__, 0, OMP_DIST_POLICY_BLOCK, maxwin, n, 0, 0);
    omp_data_map_dist_init_info(__u_map_info__, 1, OMP_DIST_POLICY_BLOCK, maxwin, n, 0, 1);
    omp_map_add_halo_region(__u_map_info__, 0, maxwin, maxwin, OMP_DIST_HALO_EDGING_REFLECTING);
    omp_map_add_halo_region(__u_map_info__, 1, maxwin, maxwin, OMP_DIST_HALO_EDGING_REFLECTING);
    omp_data_map_dist_align_with_data_map_with_halo(__uold_map_info__, OMP_ALL_DIMENSIONS, 0, __u_map_info__, OMP_ALL_DIMENSIONS);
    omp_loop_dist_init_info(__off_info__, 0, OMP_DIST_POLICY_BLOCK, 0, n, 0, 0);
    omp_loop_dist_init_info(__off_info__, 1, OMP_DIST_POLICY_BLOCK, 0, m, 0, 1);
#endif

    /************************************************************************************************/
    off_init_time = read_timer_ms() - off_init_time;
    /*********** NOW notifying helper thread to work on this offload ******************/
#if DEBUG_MSG
	printf("=========================================== offloading to %d targets ==========================================\n", __num_target_devices__);
#endif
    double off_kernel_time = read_timer_ms();
    int it;
    int num_runs = 10;
    for (it=0; it< num_runs; it++) {
        int i;
        for (i = 0; i < num_its; i++) {
            off_args.it_num = i;
            omp_offloading_start(__off_info__);
        }
    }
    off_kernel_time = (read_timer_ms() - off_kernel_time)/num_runs;
    omp_print_map_info(__u_map_info__);
    omp_print_map_info(__uold_map_info__);
    omp_print_map_info(__coeff_map_info__);
#if defined (OMP_BREAKDOWN_TIMING)
    omp_offloading_info_report_profile(__off_info__, num_runs);
#endif
    omp_offloading_fini_info(__off_info__);
    omp_grid_topology_fini(__top__);

    omp_unified_free(uold);

    return off_kernel_time;
}
