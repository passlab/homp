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
	int id;
	int sysid; /* the id from the system view */
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
} omp_map_type_t;

#define OMP_NUM_ARRAY_DIMENSIONS 3
typedef struct omp_data_map {
	void * source_ptr;
	int dim[OMP_NUM_ARRAY_DIMENSIONS]; /* dimensions for the original array */
    int map_dim[OMP_NUM_ARRAY_DIMENSIONS]; /* the dimensions for the mapped region */
	/* the offset of each dimension from the original array */
	int map_offset[OMP_NUM_ARRAY_DIMENSIONS];

	int sizeof_element;

	omp_map_type_t map_type; /* the map type, to, from, or tofrom */
	void * map_buffer; /* the mapped buffer on host. This pointer is either the
	offsetted pointer from the source_ptr, or the pointer to the marshalled array subregions */
	int marshalled_or_not;

	void * map_dev_ptr; /* the mapped buffer on device */
	cudaStream_t * stream; /* the stream operations of this data map are registered with */

	int mem_size; // = map_dim[0] * map_dim[1] * map_dim[2] * sizeof_element;
    int device_id;
} omp_data_map_t;

extern void omp_marshalArrayRegion (omp_data_map_t * dmap);
extern void omp_unmarshalArrayRegion(omp_data_map_t * dmap);
extern void omp_map_buffer(omp_data_map_t * map, int marshal);

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
extern void omp_loop_map_range (omp_data_map_t * map, int dim, int start, int length, int * map_start, int * map_length);
/*
 * marshalled the array region of the source array, and copy data to to its new location (map_buffer)
 */
extern void omp_deviceMalloc_memcpyHostToDeviceAsync(omp_data_map_t * map);
extern void omp_memcpyDeviceToHostAsync(omp_data_map_t * map);
/* extern void omp_postACCKernel(int num_devices, int num_maps, cudaStream_t dev_stream[num_devices], omp_data_map_t data_map[num_devices][num_maps]);
*/
extern void omp_postACCKernel(int num_devices, int num_maps, cudaStream_t dev_stream[], omp_data_map_t *data_map);
#ifdef __cplusplus
 }
#endif

#endif /* __HOMP_H__ */
