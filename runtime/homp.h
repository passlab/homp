/*
 * homp.h
 *
 *  Created on: Sep 16, 2013
 *      Author: yy8
 */

#ifndef __HOMP_H__
#define __HOMP_H__

#include <cuda.h>
#include <cuda_runtime.h>

#include <helper_cuda.h>

/* the OMP_DEFAULT_DEVICE env variable is also defined in 4.0, which set the default-device-var icv */
int default_device_var = -1; /* -1 means no device, the runtime should be initialized this to be 0 if there is at least one device */
/* the following APIs are already in 4.0 standard */
void omp_set_default_device(int device_num ) {
  default_device_var = device_num;
}
int omp_get_default_device(void) {
  return default_device_var;
}

/**
 * multiple device support from here
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

/**
 ********************** Compiler notes *********************************************
 * The recommended compiler flag name to output the
 * list of supported device is "-omp-device-types". The use of "-omp-device-types"
 * could be also as follows to restrict the compiler to only support
 * compilation for the listed device types: -omp-device-types="TYPE1,TYPE2,...,TYPE3"
 */

/* APIs to support multiple devices: */
char * omp_get_device_types( ); /* return a list of devices supported by the compiler in the format of TYPE1:TYPE2 */
omp_device_type_t omp_get_device_type(int devid);
char * omp_get_device_type_as_string(int devid);
int omp_get_num_devices_of_type(omp_device_type_t type); /* current omp has omp_get_num_devices(); */
void omp_get_devices(omp_device_type_t type, int *devnum_array, int *ndev); /* return a list of devices of the specified type */

/**
 ********************* Runtime notes ***********************************************
 * runtime may want to have internal array to supports the programming APIs for mulitple devices, e.g.
 */
char * omp_device_type_name[OMP_NUM_DEVICE_TYPES];

typedef struct omp_device {
	int id;
	int sysid; /* the id from the system view */
	omp_device_type_t type;
	int status;
	struct omp_device * next; /* the device list */
} omp_device_t;

omp_device_t * omp_get_device(int id);

typedef enum omp_map_type {
	OMP_MAP_TO,
	OMP_MAP_FROM,
	OMP_MAP_TOFROM,
} omp_map_type_t;

typedef struct omp_data_map {
	void * source_ptr;
	int dim_x;
	int dim_y;
	int dim_z;

	int sizeof_element;

	omp_map_type_t map_type; /* the map type, to, from, or tofrom */
	void * map_buffer; /* the mapped buffer on host. This pointer is either the
	offsetted pointer from the source_ptr, or the pointer to the marshalled array subregions */
	int marshalled_or_not;

	void * map_dev_ptr; /* the mapped buffer on device */
	cudaStream_t * stream; /* the stream operations of this data map are registered with */

	int mem_size; // = map_dim_x * map_dim_y * map_dim_z * sizeof_element;

    int map_offset_x; /* the offset from the 1st dimension of the original array */;
    int map_offset_y; /* the offset from the 2nd dimension of the original array */;
    int map_offset_z; /* the offset from the 3rd dimension of the original array */;

    int map_dim_x; /* the 1st dimension */;
    int map_dim_y; /* the 2nd dimension */;
    int map_dim_z; /* the 3rd dimension */;

    int device_id;
} omp_data_map_t;

void * omp_marshalArrayRegion (omp_data_map_t * dmap);
void * omp_unmarshalArrayRegion(omp_data_map_t * dmap);

void omp_map_buffer(omp_data_map_t * map, int marshal) {
	map->marshalled_or_not = marshal;
	(map)->mem_size = (map)->map_dim_x * (map)->map_dim_y * (map)->map_dim_z * (map)->sizeof_element;
	if (!marshal) map->map_buffer = map->source_ptr + map->map_offset_x; /* TODO: if it is 1-dimension, or two-dimension with contigunous memory, etc */
	else omp_marshalArrayRegion(map);
}

/*
 * marshalled the array region of the source array, and copy data to to its new location (map_buffer)
 */

void omp_deviceMalloc_memcpyHostToDeviceAsync(omp_data_map_t * map) {
	map->map_dev_ptr = xomp_deviceMalloc(map->mem_size);
	xomp_memcpyHostToDeviceAsync(((void *)map->map_dev_ptr),((const void *)map->map_buffer),map->mem_size, *map->stream);
}

void omp_memcpyDeviceToHostAsync(omp_data_map_t * map) {
    xomp_memcpyDeviceToHostAsync(((void *)map->map_buffer),((const void *)map->map_dev_ptr),map->mem_size, *map->stream);
}

void omp_postACCKernel(int num_devices, int num_maps, cudaStream_t dev_stream[num_devices], omp_data_map_t data_map[num_devices][num_maps]) {
	int i, j;

	for (i=0; i<num_devices; i++) {
		cudaSetDevice(i);
	    //Wait for all operations to finish
	    cudaStreamSynchronize(dev_stream[i]);
	    for (j=0; j<num_maps; j++) {
	    	omp_data_map_t * map = &data_map[i][j];
	    	omp_unmarshalArrayRegion(map);
	    	cudaFree(map->map_dev_ptr);
	    	if (map->marshalled_or_not) { /* if this is marshalled and need to free space since this is not useful anymore */
	    		free(map->map_buffer);
	    	}
	    }
	    cudaStreamDestroy(dev_stream[i]);
	}
}

extern omp_device_t * omp_devices;
extern int omp_num_device;

#endif /* __HOMP_H__ */
