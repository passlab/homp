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

#include <cuda.h>
#include <cuda_runtime.h>

/**************************** OpenMP 4.0 standard support *******************************/
extern int default_device_var; /* -1 means no device, the runtime should be initialized this to be 0 if there is at least one device */
/* the OMP_DEFAULT_DEVICE env variable is also defined in 4.0, which set the default-device-var icv */
extern void omp_set_default_device(int device_num );
extern int omp_get_default_device(void);
extern int omp_get_num_devices();

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
   OMP_NUM_DEVICE_TYPES,  /* the total number of types of supported devices */
} omp_device_type_t;

extern char * omp_device_type_name[];
/**
 ********************* Runtime notes ***********************************************
 * runtime may want to have internal array to supports the programming APIs for mulitple devices, e.g.
 */
typedef struct omp_device {
	int id; /* the id from omp view */
	int sysid; /* the id from the system view, used for call like cudaSetDevice(sysid) */
	omp_device_type_t type;
	int status;
	struct omp_device * next; /* the device list */
} omp_device_t;

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

/* APIs to support data/array mapping and distribution */
typedef enum omp_map_type {
	OMP_MAP_TO,
	OMP_MAP_FROM,
	OMP_MAP_TOFROM,
	OMP_MAP_ALLOC,
} omp_map_type_t;

typedef enum omp_map_dist_type {
	OMP_MAP_DIST_EVEN,
	OMP_MAP_DIST_COPY,
	OMP_MAP_DIST_FIX, /* user defined */
	OMP_MAP_DIST_CHUNK, /* with chunk size, and cyclic */
} omp_map_dist_type_t;

/* a topology of devices, or threads or teams */
typedef struct omp_grid_topology {
	 int nnodes;     /* Product of dims[*], gives the size of the topology */
	 int ndims;
	 int *dims;
	 int *periodic;
} omp_grid_topology_t;

#define OMP_STREAM_NUM_EVENTS 6
/**
 * each stream also provide a limited number of event objects for collecting timing
 * information.
 */
typedef struct omp_stream {
	omp_device_t * dev;
	union {
		cudaStream_t cudaStream;
		void * myStream;
	} systream;

	/* we orgainze them as set of events */
	cudaEvent_t start_event[OMP_STREAM_NUM_EVENTS];
	cudaEvent_t stop_event[OMP_STREAM_NUM_EVENTS];
	float elapsed[OMP_STREAM_NUM_EVENTS];
} omp_stream_t;
typedef struct omp_data_map omp_data_map_t;

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

#define OMP_NUM_ARRAY_DIMENSIONS 3
/* for each mapped host array, we have one such object */

typedef struct omp_data_map_info {
    omp_grid_topology_t * top;
	void * source_ptr;
	long dim[OMP_NUM_ARRAY_DIMENSIONS]; /* dimensions for the original array */
	int sizeof_element;
	omp_map_type_t map_type; /* the map type, to, from, or tofrom */

	/* the halo region: halo region is considered out-of-bound access of the main array region,
	 * thus the index could be -1, -2, or larger than the dimensions. Our memory allocation and pointer
	 * arithmetic will make sure we do not go out of memory bound
	 *
	 * In each dimension, we may have halo region.
	 */
	omp_data_map_halo_region_info_t halo_region[OMP_NUM_ARRAY_DIMENSIONS];
	/* a quick flag to tell whether this is halo region or not in this map,
	 * otherwise, we have to iterate the halo_region array to see whether this is one or not */
	short has_halo_region;

	omp_data_map_t ** maps; /* a list of data maps of this array */
} omp_data_map_info_t;

/* for each device, we maintain such an object */
struct omp_data_map {
	omp_data_map_info_t * info;
    omp_device_t * dev;
    int devsid; /* the linear id of this data environment mapping */

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
	offsetted pointer from the source_ptr, or the pointer to the marshalled array subregions */
	int marshalled_or_not;

	omp_stream_t * stream; /* the stream operations of this data map are registered with */
};

/** temp solution */
typedef struct omp_reduction_float {
				float * result;
				float *input;
				int num;
				int opers;
} omp_reduction_float_t;

extern void omp_init_devices();
extern int omp_get_num_active_devices();
extern void omp_set_current_device(omp_device_t * d);

extern void omp_init_stream(omp_device_t * d, omp_stream_t * stream);
extern void omp_stream_start_event_record(omp_stream_t * stream, int event);
extern void omp_stream_stop_event_record(omp_stream_t * stream, int event);
extern float omp_stream_event_elapsed_ms(omp_stream_t * stream, int event);

extern void omp_topology_print(omp_grid_topology_t * top);
extern void omp_data_map_init_info(omp_data_map_info_t *info, omp_grid_topology_t * top, void * source_ptr, int sizeof_element,
		omp_map_type_t map_type, long dim0, long dim1, long dim2);
extern void omp_data_map_init_map(omp_data_map_t *map, omp_data_map_info_t * info, int devsid, omp_device_t * dev,	omp_stream_t * stream);
extern void omp_data_map_do_even_map(omp_data_map_t *map, int dim, omp_grid_topology_t *top, int topdim, int devsid);
extern void omp_print_data_map(omp_data_map_t * map);

extern void omp_data_map_marshal(omp_data_map_t * map);
extern void omp_data_map_unmarshal(omp_data_map_t * map);
extern void omp_map_add_halo_region(omp_data_map_info_t * info, int dim, int left, int right, int cyclic, int top_dim);
extern void omp_map_init_add_halo_region(omp_data_map_t * map, int dim, int left, int right, int cyclic);
extern void omp_halo_region_pull(omp_data_map_t * map, int dim, int from_left_right);
extern void omp_halo_region_pull_async(omp_data_map_t * map, int dim, int from_left_right);
extern void omp_map_buffer_malloc(omp_data_map_t * map);

extern void omp_sync_stream(int num_devices, omp_stream_t dev_stream[], int destroy_stream);
extern void omp_sync_cleanup(int num_devices, int num_maps, omp_stream_t dev_stream[], omp_data_map_t data_map[]);

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
extern void xomp_beyond_block_reduction_float_stream_callback(cudaStream_t stream,  cudaError_t status, void* userData );
extern long omp_loop_map_range (omp_data_map_t * map, int dim, long start, long length, long * map_start, long * map_length);
extern void omp_deviceMalloc(omp_data_map_t * map);
/*
 * marshalled the array region of the source array, and copy data to to its new location (map_buffer)
 */
extern void omp_memcpyHostToDeviceAsync(omp_data_map_t * map);
extern void omp_memcpyDeviceToHostAsync(omp_data_map_t * map);
/* extern void omp_postACCKernel(int num_devices, int num_maps, cudaStream_t dev_stream[num_devices], omp_data_map_t data_map[num_devices][num_maps]);
*/
extern void omp_factor(int n, int factor[], int dims);
#ifdef __cplusplus
 }
#endif

#endif /* __HOMP_H__ */
