/*
 * homp.h
 *
 *  Created on: Sep 16, 2013
 *      Author: yy8
 */

#ifndef __HOMP_H__
#define __HOMP_H__

#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#if defined (DEVICE_NVGPU_SUPPORT)
#include <cuda.h>
#include <cuda_runtime.h>
#endif

/**************************** OpenMP 4.0 standard support *******************************/
extern int default_device_var; /* -1 means no device, the runtime should be initialized this to be 0 if there is at least one device */
/* the OMP_DEFAULT_DEVICE env variable is also defined in 4.0, which set the default-device-var icv */
extern void omp_set_default_device(int device_num );
extern int omp_get_default_device(void);
extern int omp_get_num_devices();

typedef struct omp_data_map omp_data_map_t;
typedef struct omp_offloading_info omp_offloading_info_t;

/**
 * multiple device support
 * the following should be a list of name agreed with vendors

 * NOTES: should we also have a device version number, e.g. a type of device
 * may have mulitple generations, thus versions.
 */
typedef enum omp_device_type {
   OMP_DEVICE_NVGPU,      /* NVIDIA GPGPUs */
   OMP_DEVICE_ITLMIC,     /* Intel MIC */
   OMP_DEVICE_TIDSP,      /* TI DSP */
   OMP_DEVICE_AMPU,       /* AMD APUs */
   OMP_DEVICE_REMOTE,	  /* a remote node */
   OMP_DEVICE_LOCALTH,	  /* a new thread of the same process, e.g. pthread */
   OMP_DEVICE_LOCALPS,	  /* a new process in the same node, e.g. a new process created using fork) */
   OMP_NUM_DEVICE_TYPES,  /* the total number of types of supported devices */
} omp_device_type_t;

extern char * omp_device_type_name[];
/**
 ********************* Runtime notes ***********************************************
 * runtime may want to have internal array to supports the programming APIs for mulitple devices, e.g.
 *
 * We have a pthread managing a accelerator device, i.e. responsible of set up the GPU,
 * launching calls for memory allocation, kernel launching as well as timing.
 * A sequence of calls in one offloading involve.
 * set up stream and event for queuing and sync purpose, we currently only one stream per
 * device at a time, i.e. multiple concurrent streams are not supported.
 * 1. compute data mapping region for each variables, allocate device memory to store mapped data
 * 2. copy data from host to device (could happen while allocating device memory)
 * 3. launch the kernel that will work on the data
 * 4. if any, do data exchange between host and devices, and between devices
 * 5. more kernel launch and data exchange
 * 6. copy data from device to host, deallocate memory that will not be used
 *
 * Between each of the above steps, a barrier may be needed.
 * The thread will can be used to a accelerator
 *
 * The helper thread use the offload_queue to keep track of a list of
 * offloading request to this device, see struct_offloading_dev_t struct definition
 */
typedef struct omp_device {
	int id; /* the id from omp view */
	char * sysid; /* the handle from the system view, e.g. 
			 device id for NVGPU cudaSetDevice(sysid), 
			 or pthread_t for LOCALTH. Need type casting to become device-specific id */
	omp_device_type_t type;
	int status;
	struct omp_device * next; /* the device list */
	omp_offloading_info_t * offload_info;
	volatile int notification_counter; /* the counter will be -1 if nothing to do, the master thread (who offloads) will reset this
	counter to the value of the device id the master thread is helping, so the other helper threads for other dev will know where to retrieve
	the offload_info */

	omp_data_map_t ** resident_data_maps; /* a link-list or an array for resident data maps (data maps cross multiple offloading region */

	pthread_t helperth;

} omp_device_t;

typedef enum OMP_OFFLOADING_STEPS {
	OMP_OFFLOADING_INIT,      /* initialization of device, e.g. stream, host barrier, etc */
	OMP_OFFLOADING_MAPMEM,    /* compute data map and allocate memory/buffer on host and device */
	OMP_OFFLOADING_COPYTO,    /* copy data from host to device */
	OMP_OFFLOADING_KERNEL,    /* kernel execution */
	OMP_OFFLOADING_EXCHANGE,  /* p2p (dev2dev) and h2d (host2dev) data exchange */
	OMP_OFFLOADING_COPYFROM,  /* copy data from dev to host */
	OMP_OFFLOADING_COMPLETE,  /* make grace complete, e.g. deallocate memory and turn off dev */
	OMP_OFFLOADING_NUM_STEPS, /* total number of steps */
} OMP_OFFLOADING_STEP_t;


#define OMP_DEV_STREAM_NUM_EVENTS 6

/**
 * each stream also provide a limited number of event objects for collecting timing
 * information.
 */
typedef struct omp_dev_stream {
	omp_device_t * dev;
	union {
#if defined (DEVICE_NVGPU_SUPPORT)
		cudaStream_t cudaStream;
#endif
		void * myStream;
	} systream;

	/* we organize them as set of events */
#if defined (DEVICE_NVGPU_SUPPORT)
	cudaEvent_t start_event[OMP_DEV_STREAM_NUM_EVENTS];
	cudaEvent_t stop_event[OMP_DEV_STREAM_NUM_EVENTS];
#endif
	float start_time[OMP_DEV_STREAM_NUM_EVENTS];
	float stop_time[OMP_DEV_STREAM_NUM_EVENTS];
	float elapsed[OMP_DEV_STREAM_NUM_EVENTS];
} omp_dev_stream_t;

/**
 ********************** Compiler notes *********************************************
 * The recommended compiler flag name to output the
 * list of supported device is "-omp-device-types". The use of "-omp-device-types"
 * could be also as follows to restrict the compiler to only support
 * compilation for the listed device types: -omp-device-types="TYPE1,TYPE2,...,TYPE3"
 */

/* APIs to support multiple devices: */
extern char * omp_supported_device_types(); /* return a list of devices supported by the compiler in the format of TYPE1:TYPE2 */
extern omp_device_type_t omp_get_device_type(int devid);
extern char * omp_get_device_type_as_string(int devid);
extern int omp_get_num_devices_of_type(omp_device_type_t type); /* current omp has omp_get_num_devices(); */
extern int omp_get_devices(omp_device_type_t type, int *devnum_array, int ndev); /* return a list of devices of the specified type */
extern omp_device_t * omp_get_device(int id);
extern omp_device_t * omp_devices; /* an array of all device objects */
extern int omp_num_devices;

typedef struct omp_grid_topology_idmap {
	int devid; /* the cooresponding devid */
	int seqid; /* the seq id in the topology */
} omp_grid_topology_idmap_t;

extern void omp_grid_topology_init_simple (omp_grid_topology_idmap_t * idmap, int nnodes, int ndims, int *dims, int *periodic, omp_grid_topology_idmap_t * idmap);

/* a topology of devices, or threads or teams */
typedef struct omp_grid_topology {
	 int nnodes;     /* Product of dims[*], gives the size of the topology */
	 int ndims;
	 int *dims;
	 int *periodic;
	 omp_grid_topology_idmap_t * idmap; /* it is an array of nnodes of elements */
} omp_grid_topology_t;

/* APIs to support data/array mapping and distribution */
typedef enum omp_data_map_type {
	OMP_DATA_MAP_TO,
	OMP_DATA_MAP_FROM,
	OMP_DATA_MAP_TOFROM,
	OMP_DATA_MAP_ALLOC,
} omp_data_map_type_t;

typedef enum omp_data_map_dist_type {
	OMP_DATA_MAP_DIST_EVEN,
	OMP_DATA_MAP_DIST_FULL,
	OMP_DATA_MAP_DIST_FIX, /* user defined */
	OMP_DATA_MAP_DIST_CHUNK, /* with chunk size, and cyclic */
} omp_data_map_dist_type_t;

typedef struct omp_data_map_dist {
	long start; /* the start index for the dim of the original array */
	long end;   /* the end index for the dim of the original array */
	int halo_left; /* the left halo, # of elements */
	int halo_right; /* the right halo, # of elements */
	int halo_cyclic;
	omp_data_map_dist_type_t type; /* the dist type */
	int topdim; /* which top dim to apply dist, for dist_even, copy*/
} omp_data_map_dist_t;

#define OMP_NUM_ARRAY_DIMENSIONS 3

/* for each mapped host array, we have one such object */
typedef struct omp_data_map_info {
    omp_grid_topology_t * top;
	void * source_ptr;
	int num_dims;
	long * dims;

	int sizeof_element;
	omp_data_map_type_t map_type; /* the map type, to, from, or tofrom */

	/*
	 * The dist array maintains info on how the array should be distributed among dev topology, including
	 * the halo region info.
	 *
	 * the halo region: halo region is considered out-of-bound access of the main array region,
	 * thus the index could be -1, -2, or larger than the dimensions. Our memory allocation and pointer
	 * arithmetic will make sure we do not go out of memory bound
	 *
	 * In each dimension, we may have halo region.
	 */
	omp_data_map_dist_t *dist;

	/* a quick flag to tell whether this is halo region or not in this map,
	 * otherwise, we have to iterate the halo_region array to see whether this is one or not */
	short has_halo_region;

	omp_data_map_t ** maps; /* a list of data maps of this array */
} omp_data_map_info_t;

/* for each device, we maintain a list such objects, each for one mapped array */
struct omp_data_map {
	omp_data_map_info_t * info;
    omp_device_t * dev;

	long map_dim[OMP_NUM_ARRAY_DIMENSIONS]; /* the dimensions for the mapped region */
	/* the offset of each dimension from the original array for the mapped region (not the mem region)*/
	long map_offset[OMP_NUM_ARRAY_DIMENSIONS];
	void * map_dev_ptr; /* the mapped buffer on device, only for the mapped array region (not including halo region) */
	long map_size; // = map_dim[0] * map_dim[1] * map_dim[2] * sizeof_element;

	omp_data_map_halo_region_mem_t halo_mem [OMP_NUM_ARRAY_DIMENSIONS];
	long mem_dim[OMP_NUM_ARRAY_DIMENSIONS]; /* the dimensions for the mem region for both mapped region and halo region */
	void * mem_dev_ptr; /* the mapped buffer on device, for the mapped array region plus halo region */
	long mem_size; // = mem_dim[0] * mem_dim[1] * mem_dim[2] * sizeof_element;

    void * map_buffer; /* the mapped buffer on host. This pointer is either the
	offset pointer from the source_ptr, or the pointer to the marshalled array subregions */
	int marshalled_or_not;

	omp_dev_stream_t * stream; /* the stream operations of this data map are registered with, mostly it will be the stream created for an offloading */
};

/**
 * in each dimension, halo region have left and right halo region and also a flag for cyclic halo or not,
 */
typedef struct omp_data_map_halo_region_info {
	int left; /* element size */
	int right; /* element size */
	short cyclic;
	int top_dim; /* which dimension of the device topology this halo region is related to */
} omp_data_map_halo_region_info_t;

typedef struct omp_data_map_halo_region_mem {
	/* the in/out pointer is the buffer for the halo regions.
	 * The in ptr is the buffer for halo region that will be copied in,
	 * and the out is for those that will be copied out.
	 * In implementation, we put the in and out buffer into one mem space for each left and right halo region
	 */
	omp_data_map_t *left_map; /* left data map */
	omp_data_map_t *right_map; /* left data map */
	void * left_in_ptr;
	long left_in_size; /* for pull update, == right_out_size if push protocol is used */
	void * left_out_ptr;
	void * right_in_ptr;
	long right_in_size; /* for pull update, == left_out_size if push protocol is used */
	void * right_out_ptr;
} omp_data_map_halo_region_mem_t;

/**
  * info per offloading
  *
  * For each offloading to one or multiple devices, we will maintain an object of omp_offloading_info
  * that keeps track of the topology of target devices, the mapped variables and other info.
  *
  * The barrier is used for syncing target devices
  */
typedef struct omp_offloading_info {
	/************** per-offloading var, shared by all target devices ******/
	omp_grid_topology_t * top; /* num of target devices are in this object */
	omp_device_t ** targets; /* a list of target devices */

	int num_mapped_vars;
	omp_data_map_info_t * data_map_info; /* an entry for each mapped variable */
	omp_offloading_t * dev_offloadings; /* a list of dev-specific offloading objects */

	void *(*kernel)(void *); /* the same kernel to be called by each of the target device, if kernel == NULL, we are just offloading data */

	/* the parcipating barrier */
	pthread_barrier_t barrier;

} omp_offloading_info_t;

/**
 * info for per device
 *
 * For each offloading, we have this object to keep track per-device info, e.g. the stream used, mapped region of
 * a variable (array or scalar) and kernel info.
 *
 * The next pointer is used to form the offloading queue (see omp_device struct)
 */
typedef struct omp_offloading {
	/* per-offloading info */
	omp_offloading_info_t * off_info;

	/************** per device var ***************/	
	omp_dev_stream_t stream;

	/* the map for a variable on this device */
	omp_data_map_t ** data_maps; /* the data maps used only for this specific offloading */

	/* kernel info */
	int X1, Y1, Z1; /* the first level kernel thread configuration, e.g. CUDA blockDim */
	int X2, Y2, Z2; /* the second level kernel thread config, e.g. CUDA gridDim */
	void ** para;
	void *(*kernel)(void *); /* device specific kernel, if any */

	struct omp_offloading * next; /* the link to form offloading queue */
	struct omp_offloading * prev;
} omp_offloading_t;

extern void omp_offloading_init_info(omp_offloading_info_t * info, omp_grid_topology_t * top, omp_device_t **targets, int num_mapped_vars,
		omp_data_map_info_t * data_map_info, void *(*kernel)(void *));

/** temp solution */
typedef struct omp_reduction_float {
				float result;
				float *input;
				int num;
				int opers;
} omp_reduction_float_t;

extern int omp_init_devices(); /* return # of devices initialized */
extern int omp_get_num_active_devices();
extern int omp_set_current_device_dev(omp_device_t * d); /* return the current device id */
extern int omp_set_current_device(int id); /* return the current device id */

extern void omp_init_stream(omp_device_t * d, omp_dev_stream_t * stream);
extern void omp_stream_start_event_record(omp_dev_stream_t * stream, int event);
extern void omp_stream_stop_event_record(omp_dev_stream_t * stream, int event);
extern float omp_stream_event_elapsed_ms(omp_dev_stream_t * stream, int event);
extern float omp_stream_event_elapsed_accumulate_ms(omp_dev_stream_t * stream, int event);

extern void omp_topology_print(omp_grid_topology_t * top);
extern void omp_data_map_init_info(omp_data_map_info_t *info, omp_grid_topology_t * top, void * source_ptr, int num_dims, long* dims, int sizeof_element,
		omp_data_map_type_t * map_type, omp_data_map_dist_t * dist);
extern void omp_data_map_init_info_dist_straight(omp_data_map_info_t *info, omp_grid_topology_t * top, void * source_ptr, int num_dims, long* dims, int sizeof_element,
		omp_data_map_type_t * map_type, omp_data_map_dist_t * dist, omp_data_map_dist_type_t * dist_type) ;
extern void omp_data_map_init_map(omp_data_map_t *map, omp_data_map_info_t * info, omp_device_t * dev,	omp_dev_stream_t * stream);
extern void omp_data_map_do_even_dist(omp_data_map_t *map, int dim, omp_grid_topology_t *top, int topdim, int devid);
extern void omp_print_data_map(omp_data_map_t * map);

extern void omp_map_marshal(omp_data_map_t * map);
extern void omp_map_unmarshal(omp_data_map_t * map);
extern void omp_map_add_halo_region(omp_data_map_info_t * info, int dim, int left, int right, int cyclic);
extern void omp_map_init_add_halo_region(omp_data_map_t * map, int dim, int left, int right, int cyclic);
extern void omp_halo_region_pull(omp_data_map_t * map, int dim, int from_left_right);
extern void omp_halo_region_pull_async(omp_data_map_t * map, int dim, int from_left_right);
extern void omp_map_buffer_malloc(omp_data_map_t * map);

extern void omp_sync_stream(int num_devices, omp_dev_stream_t dev_stream[], int destroy_stream);
extern void omp_sync_cleanup(int num_devices, int num_maps, omp_dev_stream_t dev_stream[], omp_data_map_t data_map[]);

#if defined (DEVICE_NVGPU_SUPPORT)
extern void xomp_beyond_block_reduction_float_stream_callback(cudaStream_t stream,  cudaError_t status, void* userData );
#endif

/**
 * return the mapped range index from the iteration range of the original array
 * e.g. A[128], when being mapped to a device for A[64:64] (from 64 to 128), then, the range 100 to 128 in the original A will be
 * mapped to 36 to 64 in the mapped region of A
 *
 * @param: omp_data_map_t * map: the mapped variable, we should use the original pointer and let the runtime retrieve the map
 * @param: int dim: which dimension to retrieve the range
 * @param: int start: the start index from the original array, if start is -1, use the map_offset_<dim>, which will simply cause
 * 					the function return 0 for obvious reasons
 * @param: int length: the length of the range, if -1, use the mapped dim from the start
 * @param: int * map_start: the mapped start index in the mapped range, if return <0 value, wrong input
 * @param: int * map_length: normally just the length, if lenght == -1, use the map_dim[dim]
 *
 * NOTE: the mapped range must be a subset of the range of the specified map in the specified dim
 *
 */
extern long omp_loop_map_range (omp_data_map_t * map, int dim, long start, long length, long * map_start, long * map_length);
/*
 * marshalled the array region of the source array, and copy data to to its new location (map_buffer)
 */
extern void omp_map_malloc_dev(omp_data_map_t * map);
extern void omp_map_memcpy_to(omp_data_map_t * map);
extern void omp_map_memcpy_to_async(omp_data_map_t * map);
extern void omp_map_memcpy_from(omp_data_map_t * map);
extern void omp_map_memcpy_from_async(omp_data_map_t * map) ;
extern void omp_map_memcpy_DeviceToDevice(omp_data_map_t * dst, omp_data_map_t * src, int size);
extern void omp_map_memcpy_DeviceToDeviceAsync(omp_data_map_t * dst, omp_data_map_t * src, int size);
/* extern void omp_postACCKernel(int num_devices, int num_maps, cudaStream_t dev_stream[num_devices], omp_data_map_t data_map[num_devices][num_maps]);
*/
/*  factor input n into dims number of numbers (store into factor[]) whose multiplication equals to n */
extern void omp_factor(int n, int factor[], int dims);

extern void devcall_errchk(int code, char *file, int line, int abort);
#define devcall_assert(ecode) { devcall_errchk((ecode), __FILE__, __LINE__, 1); }

#ifdef __cplusplus
 }
#endif

#endif /* __HOMP_H__ */
