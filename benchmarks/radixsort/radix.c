#include<string.h>
#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<string.h>
#include<math.h>
#include "homp.h"

#define FILENAME "a480.dat"
#define NUMBERS 480
#define DEBUG 1
#define REAL float

#if 0
/* v2: explicit distribution of both data and loop:
 * the y[0:n], and x[0:n] will be evenly distributed among the ndev devices,
 * scalars such as a and n will each have a mapped copy in all the devices, loop will also be evenly distributed */
void axpy_mdev_v2(REAL* x, REAL* y, long n, REAL a) {
#pragma omp target device (*) map(tofrom: y[0:n] dist_data(BLOCK)) map(to: x[0:n] dist_data(BLOCK),a,n)
#pragma omp parallel for shared(x, y, n, a) dist_iteration(BLOCK)
	for (i = 0; i < n; ++i)
	y[i] += a * x[i];
}

/* v3: block distribute array x and y and let the loop distribution aligh with x
 */
void axpy_mdev_v3(REAL* x, REAL* y, long n, REAL a) {
#pragma omp target device (*) map(tofrom: y[0:n] dist_data(BLOCK)) map(to: x[0:n] dist_data(BLOCK),a,n)
#pragma omp parallel for shared(x, y, n, a) dist_iteration(ALIGN(x))
	for (i = 0; i < n; ++i)
	y[i] += a * x[i];
}

/* v4: AUTO-distribute the loop iteration and let the distribution of array x and y to be aligned with loop distribution.
 */
void axpy_mdev_v4(REAL* x, REAL* y, long n, REAL a) {
#pragma omp target device (*) map(tofrom: y[0:n] dist_data(ALIGN)) map(to: x[0:n] dist_data(ALIGN),a,n)
#pragma omp parallel for shared(x, y, n, a) dist_iteration(AUTO)
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

extern int sort_mdev_v = 3;





//#define FILENAME "a480.dat"
//#define NUMBERS 480

//#define FILENAME "a4095.dat"
//#define NUMBERS 4095



REAL sort_ompacc_mdev(int *bufferInt);

int *r_values;
int *d_values;
int *t_values;

int *d_split;
int *d_e;
int *d_f;
int *d_t;
struct sort_args {
    int *bufferInt;
};

#if defined (DEVICE_NVGPU_SUPPORT)
#include "xomp_cuda_lib_inlined.cu"
__global__ void sort(long start_n, long length_n, int *bufferInt) {
	printf("Inside kernel \n");
	int _p_i;
	long _dev_lower;
	long _dev_upper;
	long _dev_loop_chunk_size;
	long _dev_loop_sched_index;
	long _dev_loop_stride;
	long _dev_thread_num = getCUDABlockThreadCount(1);
	long _dev_thread_id = getLoopInexFromCUDAVariables(1);
	//int x = (blockIdx.x * blockDim.x) + threadIdx.x;

	XOMP_static_sched_init(start_n, start_n + length_n - 1, 1, 1,
			_dev_thread_num, _dev_thread_id,
			&_dev_loop_chunk_size, &_dev_loop_sched_index,
			&_device_loop_stride);

	while (XOMP_static_sched_next
			(&_dev_loop_sched_index, start_n + length_n - 1, 1,
					_dev_loop_stride, _dev_loop_chunk_size, _dev_thread_num,
					_dev_thread_id, &_dev_lower, &_dev_upper))

	int b[NUMBERS], m = 0, exp = 1;
	for (_p_i = _dev_lower, _p_i <= _dev_upper; _p_i += 1) {
		for (x = 0; x < NUMBERS; x++) {
			{
				if (bufferInt[x] > m)
				m = bufferInt[x];
			}
			while (m / exp > 0) {
				int bucket[NUMBERS] = {0};

				for (x = 0; x < NUMBERS; x++)
				bucket[bufferInt[x] / exp % 10]++;
				__syncthreads();
				for (x = 1; x < NUMBERS; x++)
				bucket[x] += bucket[x - 1];
				__syncthreads();
				for (x = NUMBERS - 1; x >= 0; x--)
				b[--bucket[bufferInt[x] / exp % 10]] = bufferInt[x];

				__syncthreads();
				for (x = 0; x < NUMBERS; x++)
				bufferInt[x] = b[x];

				__syncthreads();
				exp *= 10;
			}
		}
	}
}
#endif


//Launcher function
void sort_launcher(omp_offloading_t *off, void *args) {
    printf("Inside launcher function \n");
    struct sort_args *argss = (struct sort_args *) args;
    long start_n, length_n;
    int *bufferInt = argss->bufferInt;

    omp_data_map_t *map_bufferInt = omp_map_get_map(off, argss->bufferInt, -1);

    //LatLong *locations=(LatLong *)map_distances->map_dev_ptr;
    bufferInt = (int *) map_bufferInt->map_dev_ptr;

    omp_loop_get_range(off, 0, &start_n, &length_n);

    omp_device_type_t devtype = off->dev->type;


#if defined (DEVICE_NVGPU_SUPPORT)
	if (devtype == OMP_DEVICE_NVGPU) {
	int threads_per_team =
		omp_get_optimal_threads_per_team(off->dev);
		int teams_per_league =
		omp_get_optimal_teams_per_league(off->dev, threads_per_team,
				length_n);

		sort <<< teams_per_league, threads_per_team, 0,
		off->stream->systream.cudaStream >>> (start_n, length_n,bufferInt);

		fprintf("device type is GPU \n");
	}
	else
#endif

    if (devtype == OMP_DEVICE_THSIM) {
        int i;
//#pragma omp parallel for shared (bufferInt)private(i)
//serial code starts here:
        printf("Inside serial sort function \n");

        // int *bufferInt = NULL;
        char tmpLine[20];        //to store the contents of the line we read from the file
        FILE *handler = fopen(FILENAME, "r");
        if (handler == NULL) {
            printf("Unable to open file\n");
            exit(-1);
        }
        bufferInt = (int *) malloc(sizeof(int) * (NUMBERS));

        for (i = 0; i < NUMBERS; i++) {
            fgets(tmpLine, 20, handler);
            bufferInt[i] = atoi(tmpLine);
        }

#if DEBUG == 1
        printf("Printing output\n");
        //printing output to verify
        for (i = 0; i < NUMBERS; i++) {
            //      printf("%d \n ", bufferInt[i]);
        }
        printf("\n");
#endif
        //sorting starts here
        int b[NUMBERS], m = 0, exp = 1;

        for (i = 0; i < NUMBERS; i++) {
            if (bufferInt[i] > m)
                m = bufferInt[i];
        }
        while (m / exp > 0) {
            int bucket[NUMBERS] = {0};

            for (i = 0; i < NUMBERS; i++)
                bucket[bufferInt[i] / exp % 10]++;
            for (i = 1; i < NUMBERS; i++)
                bucket[i] += bucket[i - 1];
            for (i = NUMBERS - 1; i >= 0; i--)
                b[--bucket[bufferInt[i] / exp % 10]] = bufferInt[i];
            for (i = 0; i < NUMBERS; i++)
                bufferInt[i] = b[i];
            exp *= 10;
        }
        for (i = 0; i < NUMBERS; i++) {
            //    printf("After sorting is %d \n",bufferInt[i]);
            if (i == 0) {
                printf("Minimum value of array is %d \n", bufferInt[i]);
            }
            if (i == NUMBERS - 1) {
                printf("Maximum value of array is %d \n", bufferInt[i]);
            }
            //Median Value calculation
            if (i == NUMBERS / 2) {
                if (i % 2 == 0) {
                    float temp = bufferInt[i] + bufferInt[i - 1];
                    printf("median value is %f \n", temp / 2);
                }
                else {
                    int temp1 = bufferInt[i];
                    printf("Median value is %d \n", temp1);
                }
            }
        }

        //serial code ends here


    }
    else {
        printf("device type is not supported for this call \n");
        abort();
    }

    printf("ending launcher function \n");
}


//sort_ompacc_mdev function
REAL sort_ompacc_mdev(int *bufferInt) {
    printf("Inside sort_ompacc_mdev function \n");
    double ompacc_time = read_timer_ms();
    int __num_target_devices__ = omp_get_num_active_devices();

    omp_device_t *__target_devices__[__num_target_devices__];

    int __i__;
    for (__i__ = 0; __i__ < __num_target_devices__; __i__++) {
        __target_devices__[__i__] = &omp_devices[__i__];
    }
    printf("Initializing grid \n");
    omp_grid_topology_t __top__;
    int __top_ndims__ = 1;
    int __top_dims__[__top_ndims__];
    int __top_periodic__[__top_ndims__];
    int __id_map__[__num_target_devices__];
    omp_grid_topology_init_simple(&__top__, __target_devices__, __num_target_devices__, __top_ndims__, __top_dims__, __top_periodic__, __id_map__);


    printf("Before mapping array \n");
    int __num_mapped_array__ = 1;
    omp_data_map_info_t __data_map_infos__[__num_mapped_array__];

    omp_offloading_info_t __offloading_info__;
    __offloading_info__.offloadings = (omp_offloading_t *) alloca(sizeof(omp_offloading_t) * __num_target_devices__);
    printf("Before defining structure \n");
    struct sort_args args;
    args.bufferInt = bufferInt;

    __offloading_info__.per_iteration_profile.num_fp_operations = 1;
    __offloading_info__.per_iteration_profile.num_load = 2;
    __offloading_info__.per_iteration_profile.num_store = 1;
    omp_dist_info_t loop_nest_dist[1];

    printf("Befor calling the launcher function \n");

    omp_offloading_init_info("Sorting Kernel", &__offloading_info__, &__top__, __target_devices__, 1, OMP_OFFLOADING_DATA_CODE,
                             __num_mapped_array__, __data_map_infos__, sort_launcher, &args, loop_nest_dist, 1);

    //a for loop for multiple arrays
    omp_data_map_info_t *__info__ = &__data_map_infos__[0];

    long bufferInt_dims[1];
    bufferInt_dims[0] = NUMBERS;

    omp_data_map_t bufferInt_maps[__num_target_devices__];
    omp_dist_info_t bufferInt_dist[1];

    __info__ = &__data_map_infos__[1];
    omp_data_map_init_info("bufferInt", __info__, &__offloading_info__, bufferInt, 1, bufferInt_dims, sizeof(int *), bufferInt_maps,
                           OMP_DATA_MAP_TO, OMP_DATA_MAP_AUTO, bufferInt_dist);


    printf("Before defining mdev types \n");
    if (sort_mdev_v == 3) {
        printf("Entering sort_mdev_v \n");
        omp_dist_init_info(&bufferInt_dist[0], OMP_DIST_POLICY_BLOCK, 0, NUMBERS, 0);
        omp_align_dist_init_info(&loop_nest_dist[0], OMP_DIST_POLICY_ALIGN, &__data_map_infos__[0], OMP_DIST_TARGET_DATA_MAP, 0);
        printf("VERSION 3: BLOCK dist policy for distances and location\n");
        printf("Exited sort_mdev_v version 3 \n");
    } else if (sort_mdev_v == 4) {
        omp_dist_init_info(&loop_nest_dist[0], OMP_DIST_POLICY_AUTO, 0, NUMBERS, 0);
        omp_align_dist_init_info(&bufferInt_dist[0], OMP_DIST_POLICY_ALIGN, &__offloading_info__, OMP_DIST_TARGET_LOOP_ITERATION, 0);
        printf("Version 4: Auto policy for loop and distances and location with loop dist \n");
    } else {
        omp_dist_init_info(&bufferInt_dist[0], OMP_DIST_POLICY_BLOCK, 0, NUMBERS, 0);
        omp_dist_init_info(&loop_nest_dist[0], OMP_DIST_POLICY_BLOCK, 0, NUMBERS, 0);
    }
#if DEBUG_MSG
	 printf("=========================================== offloading to %d targets ==========================================\n", __num_target_devices__);
#endif
    /* here we do not need sync start */
    printf("before omp_offloading_start function \n");

    int it;
    int total_its = 20;
    for (it = 0; it < total_its; it++)
        omp_offloading_start(&__offloading_info__, it == total_its - 1);
    omp_offloading_fini_info(&__offloading_info__);
    ompacc_time = read_timer_ms() - ompacc_time;
#if defined (OMP_BREAKDOWN_TIMING)
    printf("Before profiling thingy \n");
	omp_offloading_info_report_profile(&__offloading_info__);
#endif

    double cpu_total = ompacc_time;
    printf("After calculating cpu_total \n");
    return cpu_total;

}


int
main() {

    printf("Inside main function \n");
    int *bufferInt = NULL;
    char tmpLine[20];        //to store the contents of the line we read     from the file
    int i;

    FILE *handler = fopen(FILENAME, "r");
    if (handler == NULL) {
        printf("Unable to open file\n");
        exit(-1);
    }
    bufferInt = (int *) malloc(sizeof(int) * (NUMBERS));

    for (i = 0; i < NUMBERS; i++) {
        fgets(tmpLine, 20, handler);
        bufferInt[i] = atoi(tmpLine);
    }

    omp_init_devices();
    REAL omp_time = read_timer_ms();
    REAL sort_kernel_time = sort_ompacc_mdev(bufferInt);
    omp_time = (read_timer_ms() - omp_time);
    omp_fini_devices();

/*

#if DEBUG == 1
#endif
   float time_data, time_kernel, time_total;

   int *d_bufferInt = NULL;
  int *d_sortedArray = NULL;
  //srand((unsigned) time(NULL));
  size_t bytes = NUMBERS * sizeof (int);
  cudaMalloc (&d_bufferInt, bytes);
  cudaMalloc (&d_sortedArray, bytes);
  cudaMemcpy (d_bufferInt, bufferInt, bytes, cudaMemcpyHostToDevice);
  sort <<< 1, 16 >>> (d_bufferInt);

  cudaMemcpy (bufferInt, d_bufferInt, bytes, cudaMemcpyDeviceToHost);
  printf ("Data Transfer Time %f ms \n", time_data);
  printf ("Kernel Execution Time %f ms \n", time_kernel);
  printf ("Total Execution Time %f ms \n", time_total);
  for (i = 0; i < NUMBERS; i++)
    {
      //printf("After sorting is %d \n", bufferInt[i]);
      if (i == 0)
	{
	  printf ("Minimum value of array is %d \n", bufferInt[i]);
	}
      if (i == NUMBERS - 1)
	{
	  printf ("Maximum value of array is %d \n", bufferInt[i]);
	}
      //Median Value calculation
      if (i == NUMBERS / 2)
	{
	  if (i % 2 == 0)
	    {
	      float temp = bufferInt[i] + bufferInt[i - 1];
	      printf ("median value is %f \n", temp / 2);
	    }
	  else
	    {
	      int temp1 = bufferInt[i];
	      printf ("Median value is %d \n", temp1);
	    }
	}
    }

  cudaFree (d_bufferInt);
  cudaFree (d_sortedArray);

  free (bufferInt);
*/
    printf("ending main function \n");
    return (0);

}
