/*
 * homp.c
 *
 *  Created on: Sep 16, 2013
 *      Author: yy8
 */
#include <stdio.h>
#include <string.h>
#include "homp.h"

/* OpenMP 4.0 support */
int default_device_var = -1;
void omp_set_default_device(int device_num ) {
  default_device_var = device_num;
}
int omp_get_default_device(void) {
  return default_device_var;
}

int omp_get_num_devices() {
	return omp_num_devices;
}

omp_device_t * omp_devices;
int omp_num_devices;
char * omp_device_type_name[OMP_NUM_DEVICE_TYPES];

/* APIs to support multiple devices: */
char * omp_supported_device_types() { /* return a list of devices supported by the compiler in the format of TYPE1:TYPE2 */
	/* FIXME */
	return "OMP_DEVICE_NVGPU";
}
omp_device_type_t omp_get_device_type(int devid) {
	return omp_devices[devid].type;
}

char * omp_get_device_type_as_string(int devid) {
	/* FIXME */
	return "OMP_DEVICE_NVGPU";
}

int omp_get_num_devices_of_type(omp_device_type_t type) { /* current omp has omp_get_num_devices(); */
	int num = 0;
	int i;
	for (i=0; i<omp_num_devices; i++)
		if (omp_devices[i].type == type) num++;
	return num;
}
/*
 * return the first ndev device IDs of the specified type, the function returns the actual number of devices
 * in the array (devnum_array)
 *
 * before calling this function, the caller should allocate the devnum_array[ndev]
 */
int omp_get_devices(omp_device_type_t type, int *devnum_array, int ndev) { /* return a list of devices of the specified type */
	int i;
	int num = 0;
	for (i=0; i<omp_num_devices; i++)
		if (omp_devices[i].type == type && num <= ndev) {
			devnum_array[num] = omp_devices[i].id;
			num ++;
		}
	return num;
}
omp_device_t * omp_get_device(int id) {
	return &omp_devices[id];
}

static void omp_query_device_count(int * count) {
	/* currently, only do the CUDA GPUs */
	cudaGetDeviceCount(count);
}

/* init the device objects, num_of_devices, default_device_var ICV etc */
void omp_init_devices() {
	omp_query_device_count(&omp_num_devices);
	omp_devices = malloc(sizeof(omp_device_t) * omp_num_devices);
	int i;
	for (i=0; i<omp_num_devices; i++)
	{
		omp_devices[i].id = i;
		omp_devices[i].type = OMP_DEVICE_NVGPU;
		omp_devices[i].status = 1;
		omp_devices[i].sysid = i;
		omp_devices[i].next = &omp_devices[i+1];
	}
	if (omp_num_devices) {
		default_device_var = 0;
		omp_devices[omp_num_devices-1].next = NULL;
	}
	printf("System has total %d GPU devices, and the number of active (enabled) devices can be controlled by setting OMP_NUM_ACTIVE_DEVICES variable\n", omp_num_devices);
}

int omp_get_num_active_devices() {
	 int num_dev;
	 char * ndev = getenv("OMP_NUM_ACTIVE_DEVICES");
	 if (ndev != NULL) {
		 num_dev = atoi(ndev);
	     if (num_dev == 0 || num_dev > omp_num_devices) num_dev = omp_num_devices;
	 } else {
		 num_dev = omp_num_devices;
	 }
	 return num_dev;
}

void omp_set_current_device(omp_device_t * d) {
	if (d->type == OMP_DEVICE_NVGPU) {
		cudaSetDevice(d->sysid);
	} else {
		fprintf(stderr, "device type (%d) is not yet supported!\n", d->type);
	}
}

void omp_init_stream(omp_device_t * d, omp_stream_t * stream) {
	stream->dev = d;
	if (d->type == OMP_DEVICE_NVGPU) {
		cudaStreamCreate(&stream->systream.cudaStream);
	} else {
		fprintf(stderr, "device type (%d) is not yet supported!\n", d->type);
	}
}

void omp_data_map_init_info(omp_data_map_info_t *info, omp_grid_topology_t * top, void * source_ptr, int sizeof_element,
		omp_map_type_t map_type, long dim0, long dim1, long dim2) {
	info->top = top;
	info->source_ptr = source_ptr;
	info->map_type = map_type;
	info->dim[0] = dim0;
	info->dim[1] = dim1;
	info->dim[2] = dim2;
	info->sizeof_element = sizeof_element;
	int i;
	for (i=0; i<OMP_NUM_ARRAY_DIMENSIONS; i++) {
		omp_data_map_halo_region_info_t * halo = &info->halo_region[i];
		halo->left = halo->right = halo->cyclic = 0;
		halo->top_dim = -1;
	}
	info->has_halo_region = 0;
}

void omp_map_add_halo_region(omp_data_map_info_t * info, int dim, int left, int right, int cyclic, int top_dim) {
	omp_grid_topology_t * top = info->top;
	/* calculate the coordinates of id in the top */
	int dimsize = top->dims[top_dim];
	if (dimsize == 1) { /* no need to do halo region at all */
		return;
	} else {
		info->halo_region[dim].left = left;
		info->halo_region[dim].right = right;
		info->halo_region[dim].cyclic = cyclic;
		info->halo_region[dim].top_dim = top_dim;
		info->has_halo_region = 1;
	}
}

/**
 * after initialization, by default, it will perform full map of the original array
 */
void omp_data_map_init_map(omp_data_map_t *map, omp_data_map_info_t * info, int devsid, omp_device_t * dev,	omp_stream_t * stream) {
	map->info = info;
	map->dev = dev;
	map->stream = stream;
	map->devsid = devsid;
	info->maps[devsid] = map; /* link this map to the info object */
	int i;
	for (i=0; i<OMP_NUM_ARRAY_DIMENSIONS; i++) {
		map->map_dim[i] = map->info->dim[i]; /* default, full mapping */
		map->mem_dim[i] = map->info->dim[i]; /* default, full mapping */
		map->map_offset[i] = 0;
		map->halo_mem[i].left_map = map->halo_mem[i].right_map = NULL;
	}
	map->marshalled_or_not = 0;
}

/**
 * given a sequence id, return the top coordinates
 * the function return the actual number of dimensions
 */
int omp_topology_get_coords(omp_grid_topology_t * top, int sid, int ndims, int coords[]) {
	if (top->ndims > ndims) {
		fprintf(stderr, "the given ndims and array are too small\n");
		return -1;
	}
	int i, nnodes;
	nnodes = top->nnodes;
    for ( i=0; i < top->ndims; i++ ) {
    	nnodes    = nnodes / top->dims[i];
    	coords[i] = sid / nnodes;
        sid  = sid % nnodes;
    }

    ndims = i;
    printf("sid %d coords: ", sid);
    for (i=0; i<ndims; i++)
	    printf("%d ", coords[i]);
    printf("\n");
    return i;
}

/* return the sequence id of the coord
 */
int omp_topology_get_devsid(omp_grid_topology_t * top, int coords[]) {
	int i;
	int devsid = 0;
	/* TODO: currently only for 2D */
	return coords[0]*top->dims[1] + coords[1];
}

void omp_topology_print(omp_grid_topology_t * top) {
	printf("top: %X (%d): ", top, top->nnodes);
	int i;
	for(i=0; i<top->ndims; i++)
		printf("%d ", top->dims[i]);
	printf("\n");
}

void omp_topology_get_neighbors(omp_grid_topology_t * top, int devsid, int topdim, int cyclic, int* left, int* right) {
	if (devsid < 0 || devsid > top->nnodes) {
		*left = -1;
		*right = -1;
		return;
	}
	int coords[top->ndims];
	omp_topology_get_coords(top, devsid, top->ndims, coords);

    int dimcoord = coords[topdim];
    int dimsize = top->dims[topdim];

    int leftdimcoord = dimcoord - 1;
    int rightdimcoord = dimcoord + 1;
    if (cyclic) {
    	if (leftdimcoord < 0)
    		leftdimcoord = dimsize - 1;
    	if (rightdimcoord == dimsize)
    		rightdimcoord = 0;
    	coords[topdim] = leftdimcoord;
    	*left = omp_topology_get_devsid(top, coords);
    	coords[topdim] = rightdimcoord;
    	*right = omp_topology_get_devsid(top, coords);
    	return;
    } else {
    	if (leftdimcoord < 0) {
    		*left = -1;
    		if (rightdimcoord == dimsize) {
    			*right = -1;
    			return;
    		} else {
    			coords[topdim] = rightdimcoord;
    			*right = omp_topology_get_devsid(top, coords);
    			return;
    		}
    	} else {
    		coords[topdim] = leftdimcoord;
    		*left = omp_topology_get_devsid(top, coords);
    		if (rightdimcoord == dimsize) {
    			*right = -1;
    			return;
    		} else {
    			coords[topdim] = rightdimcoord;
    			*right = omp_topology_get_devsid(top, coords);
    			return;
    		}
    	}
    }
}

/**
 * for a given device (with sequence id devsid in the target device list) who is part of the topology (top) of topdim dimension, apply
 * the even distribution of the dim of the source array in map
 *
 * if there is halo region in this dimension, this will also be taken into account
 */
void omp_data_map_do_even_map(omp_data_map_t *map, int dim, omp_grid_topology_t *top, int topdim, int devsid) {
	/* calculate the coordinates of id in the top */
	int coords[top->ndims];
	omp_topology_get_coords(top, devsid, top->ndims, coords);

    int dimcoord = coords[topdim];
    int dimsize = top->dims[topdim];

    omp_data_map_info_t *info = map->info;

    /* partition the array region into subregion and save it to the map */
    int n = info->dim[dim];
    int remaint = n % dimsize;
    int esize = n / dimsize;
    int map_dim, map_offset;
    if (dimcoord < remaint) { /* each of the first remaint dev has one more element */
    	map_dim = esize+1;
        map_offset = (esize+1)*dimcoord;
    } else {
    	map_dim = esize;
        map_offset = esize*dimcoord + remaint;
    }
   	map->map_dim[dim] = map_dim;
    map->map_offset[dim] = map_offset;
    omp_data_map_halo_region_info_t * halo = &info->halo_region[dim];
    if (halo->top_dim == topdim) {
    	map_dim += (halo->left + halo->right);
    	/* allocate the halo region in contiguous memory space*/
    }
    map->mem_dim[dim] = map_dim;
}

void omp_data_map_do_fix_map(omp_data_map_t * map, int dim, int start, int length, int devsid) {
	map->map_dim[dim] = length;
	map->mem_dim[dim] = length;
	map->map_offset[dim] = start;
}

void omp_data_map_unmarshal(omp_data_map_t * map) {
	if (!map->marshalled_or_not) return;
	omp_data_map_info_t * info = map->info;
	int sizeof_element = info->sizeof_element;
	int i;
	int region_line_size = map->map_dim[1]*sizeof_element;
	int full_line_size = info->dim[1]*sizeof_element;
	int region_off = 0;
	int full_off = 0;
	char * src_ptr = info->source_ptr + sizeof_element*info->dim[1]*map->map_offset[0] + sizeof_element*map->map_offset[1];
	for (i=0; i<map->map_dim[0]; i++) {
		memcpy(src_ptr+full_off, map->map_buffer+region_off, region_line_size);
		region_off += region_line_size;
		full_off += full_line_size;
	}
//	printf("total %ld bytes of data unmarshalled\n", region_off);
}

/**
 *  so far works for at most 2D
 */
void omp_data_map_marshal(omp_data_map_t * map) {
	omp_data_map_info_t * info = map->info;
	int sizeof_element = info->sizeof_element;
	int i;
	map->map_buffer = (void*) malloc(sizeof_element*map->map_dim[0]*map->map_dim[1]*map->map_dim[2]);
	int region_line_size = map->map_dim[1]*sizeof_element;
	int full_line_size = info->dim[1]*sizeof_element;
	int region_off = 0;
	int full_off = 0;
	char * src_ptr = info->source_ptr + sizeof_element*info->dim[1]*map->map_offset[0] + sizeof_element*map->map_offset[1];
	for (i=0; i<map->map_dim[0]; i++) {
		memcpy(map->map_buffer+region_off,src_ptr+full_off, region_line_size);
		region_off += region_line_size;
		full_off += full_line_size;
	}
//	printf("total %ld bytes of data marshalled\n", region_off);
}

void omp_print_data_map(omp_data_map_t * map) {
#ifdef DEBUG_MSG
	omp_data_map_info_t * info = map->info;
	printf("MAP: %X, source ptr: %X, dim[0]: %ld, dim[1]: %ld, dim[2]: %ld, map_dim[0]: %ld, map_dim[1]: %ld, map_dim[2]: %ld, "
				"map_offset[0]: %ld, map_offset[1]: %ld, map_offset[2]: %ld, sizeof_element: %d, map_buffer: %X, marshall_or_not: %d,"
				"map_dev_ptr: %X, stream: %X, map_size: %ld, device_id: %d\n\n", map, info->source_ptr, info->dim[0], info->dim[1], info->dim[2],
				map->map_dim[0], map->map_dim[1], map->map_dim[2], map->map_offset[0], map->map_offset[1], map->map_offset[2],
				info->sizeof_element, map->map_buffer, map->marshalled_or_not, map->map_dev_ptr, map->stream, map->map_size, map->devsid);
#endif
}

/* do a halo regin pull for data map of devid. If top is not NULL, devid will be translated to coordinate of the
 * virtual topology and the halo region pull will be based on this coordinate.
 * @param: int dim[ specify which dimension to do the halo region update.
 *      If dim < 0, do all the update of map dimensions that has halo region
 * @param: int from_left_right
 * 		0: from both left and right
 * 		1: from left only
 * 		2: from right only
 *
 */
void omp_halo_region_pull(omp_data_map_t * map, int dim, int from_left_right) {
	omp_data_map_info_t * info = map->info;
	/*FIXME: let us only handle 2-D array now */
	if (info->dim[2] != 1) {
		fprintf(stderr, "we only handle 2-d array so far!\n");
		return;
	}

	if (dim != 0) {
		fprintf(stderr, "we only handle 0-dim halo region so far \n");
		return;
	}

	omp_data_map_halo_region_mem_t * halo_mem = &map->halo_mem[0];
	omp_data_map_t * left_map = halo_mem->left_map;
	omp_data_map_t * right_map = halo_mem->left_map;
	if (left_map != NULL && (from_left_right == 0 || from_left_right == 1)) {
//		cudaMemcpyPeerAsync(halo_mem->left_in_ptr, map->dev->sysid, left_map->halo_mem[0].right_out_ptr, left_map->dev->sysid, halo_mem->left_in_size, map->stream.systream.cudaStream);
		cudaMemcpyPeer(halo_mem->left_in_ptr, map->dev->sysid, left_map->halo_mem[0].right_out_ptr, left_map->dev->sysid, halo_mem->left_in_size);
	}
	if (right_map != NULL && (from_left_right == 0 || from_left_right == 2)) {
//		cudaMemcpyPeerAsync(halo_mem->right_in_ptr, map->dev->sysid, right_map->halo_mem[0].left_out_ptr, right_map->dev->sysid, halo_mem->right_in_size, map->stream.systream.cudaStream);
		cudaMemcpyPeer(halo_mem->right_in_ptr, map->dev->sysid, right_map->halo_mem[0].left_out_ptr, right_map->dev->sysid, halo_mem->right_in_size);
	}
	return;
}

void omp_halo_region_pull_async(omp_data_map_t * map, int dim, int from_left_right) {
	omp_data_map_info_t * info = map->info;
	/*FIXME: let us only handle 2-D array now */
	if (info->dim[2] != 1) {
		fprintf(stderr, "we only handle 2-d array so far!\n");
		return;
	}

	if (dim != 0) {
		fprintf(stderr, "we only handle 0-dim halo region so far \n");
		return;
	}

	omp_data_map_halo_region_mem_t * halo_mem = &map->halo_mem[0];
	omp_data_map_t * left_map = halo_mem->left_map;
	omp_data_map_t * right_map = halo_mem->left_map;
	if (left_map != NULL && (from_left_right == 0 || from_left_right == 1)) {
		cudaMemcpyPeerAsync(halo_mem->left_in_ptr, map->dev->sysid, left_map->halo_mem[0].right_out_ptr, left_map->dev->sysid, halo_mem->left_in_size, map->stream->systream.cudaStream);
	}
	if (right_map != NULL && (from_left_right == 0 || from_left_right == 2)) {
		cudaMemcpyPeerAsync(halo_mem->right_in_ptr, map->dev->sysid, right_map->halo_mem[0].left_out_ptr, right_map->dev->sysid, halo_mem->right_in_size, map->stream->systream.cudaStream);
	}
}
/**
 * this function creates host buffer, if needed, and marshall data to the host buffer,
 *
 * it will also create device memory region (both the array region memory and halo region memory
 */
void omp_map_buffer_malloc(omp_data_map_t * map) {
	omp_data_map_info_t * info = map->info;
	int sizeof_element = info->sizeof_element;

	long map_size = sizeof_element;
	int i;
	for (i=0; i<OMP_NUM_ARRAY_DIMENSIONS; i++) {
		map_size *= map->mem_dim[i];
	}
	map->map_size = map_size;

	if (map->map_dim[1] == info->dim[1] && map->map_dim[2] == info->dim[2]) {
		map->map_buffer = info->source_ptr + map->map_offset[0]*map->map_dim[1]*map->map_dim[2]*sizeof_element;
//		map->map_buffer = (void*)((long)info->source_ptr + map->map_offset[0]*map->map_dim[1]*sizeof_element);
	} else {
		map->marshalled_or_not = 1;
		omp_data_map_marshal(map);
	}

	/* we need to allocate device memory, including both the array region and halo region */
	if (cudaErrorMemoryAllocation == cudaMalloc(&map->map_dev_ptr, map_size)) {
		fprintf(stderr, "cudaMalloc error to allocate mem on device for map %X\n", map);
	} else {
	}
	omp_data_map_halo_region_info_t * halo_info = info->halo_region;
	omp_data_map_halo_region_mem_t * halo_mem = map->halo_mem;

	for (i=0; i<OMP_NUM_ARRAY_DIMENSIONS; i++) {
		if (map->mem_dim[i] != map->map_dim[i]) { /* there is halo region */
			/* enable CUDA peer memcpy */
			halo_info = &halo_info[i];
			halo_mem = &halo_mem[i];
			int left, right;
			omp_topology_get_neighbors(info->top, map->devsid, halo_info->top_dim , halo_info->cyclic, &left, &right);
			printf("%d neighbors in dim %d: left: %d, right: %d\n", map->devsid, halo_info->top_dim, left, right);
			int can_access = 0;
			if (left >=0 ) {
				cudaDeviceCanAccessPeer(&can_access, map->devsid, left);
				if (!can_access) cudaDeviceEnablePeerAccess(left, 0);
				halo_mem->left_map = info->maps[left];
			}
			if (right >=0 ) {
				can_access = 0;
				cudaDeviceCanAccessPeer(&can_access, map->devsid, right);
				if (!can_access) cudaDeviceEnablePeerAccess(right, 0);
				halo_mem->right_map = info->maps[right];
			}
		}
	}
	/* TODO: so far only for two dimension array */
	if (map->mem_dim[0] != map->map_dim[0]) { /* there is halo region */
		halo_mem = &halo_mem[0];
		halo_mem->left_in_ptr = map->map_buffer;
		halo_mem->left_in_size = halo_info->left*map->mem_dim[1]*sizeof_element;
		halo_mem->left_out_ptr = map->map_buffer + halo_mem->left_in_size;
		halo_mem->right_in_ptr = map->map_buffer+(map->mem_dim[0]-halo_info->right)*map->mem_dim[1]*sizeof_element;
		halo_mem->right_in_size = halo_info->right*map->mem_dim[1]*sizeof_element;
		halo_mem->right_out_ptr = map->map_buffer+(map->mem_dim[0]-halo_info->right-halo_info->left)*map->mem_dim[1]*sizeof_element;
	}
	if (map->mem_dim[1] != map->map_dim[1]) { /* there is halo region */
		halo_info = &halo_info[1];
		int buffer_size = sizeof_element*map->mem_dim[0]*(halo_info->left+halo_info->right);
		cudaMalloc(&halo_mem->left_in_ptr, buffer_size);
		halo_mem->left_out_ptr = halo_mem->left_in_ptr + sizeof_element*halo_info->left*map->mem_dim[0];
		cudaMalloc(&halo_mem->right_out_ptr, buffer_size);
		halo_mem->right_in_ptr = halo_mem->right_out_ptr + sizeof_element*halo_info->left*map->mem_dim[0];
	}
}

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
void omp_loop_map_range (omp_data_map_t * map, int dim, long start, long length, long * map_start, long * map_length) {
	if (start <=0) {
		if (length < 0) {
			*map_start = 0;
			*map_length = map->map_dim[dim];
			return;
		} else if (length <= map->map_dim[dim]) {
			*map_start = 0;
			*map_length = length;
			return;
		} else {
			/* error */
		}
	} else { /* start > 0 */
		*map_start = start - map->map_offset[dim];
		*map_length = map->map_dim[dim] - *map_start; /* the max length */
		if (*map_start < 0) { /* out of the range */
			*map_length = -1;
			return;
		} else if (length <= *map_length) {
			*map_length = length;
			return;
		}
	}

	/* out of range */
	*map_start = -1;
	*map_length = -1;
	return;
}

/*
 * marshalled the array region of the source array, and copy data to to its new location (map_buffer)
 */
void omp_memcpyHostToDeviceAsync(omp_data_map_t * map) {
	cudaMemcpyAsync((void *)map->map_dev_ptr,(const void *)map->map_buffer,map->map_size, cudaMemcpyHostToDevice, map->stream->systream.cudaStream);
}

void omp_memcpyDeviceToHostAsync(omp_data_map_t * map) {
    cudaMemcpyAsync((void *)map->map_buffer,(const void *)map->map_dev_ptr,map->map_size, cudaMemcpyDeviceToHost, map->stream->systream.cudaStream);
}

void omp_memcpyDeviceToHost(omp_data_map_t * map) {
    cudaMemcpy((void *)map->map_buffer,(const void *)map->map_dev_ptr,map->map_size, cudaMemcpyDeviceToHost);
}

/**
 * sync device by syncing the stream so all the pending calls the stream are completed
 *
 * if destroy_stream != 0; the stream will be destroyed.
 */
void omp_sync_stream(int num_devices, omp_stream_t dev_stream[num_devices], int destroy_stream) {
	int i;
	omp_stream_t * st;

	if (destroy_stream){
		for (i=0; i<num_devices; i++) {
			st = &dev_stream[i];
			cudaSetDevice(st->dev->sysid);
			//Wait for all operations to finish
			cudaStreamSynchronize(st->systream.cudaStream);
			cudaStreamDestroy(st->systream.cudaStream);
		}
	} else {
		for (i=0; i<num_devices; i++) {
			st = &dev_stream[i];
			cudaSetDevice(st->dev->sysid);
			//Wait for all operations to finish
			cudaStreamSynchronize(st->systream.cudaStream);
		}
	}
}

void omp_sync_cleanup(int num_devices, int num_maps, omp_stream_t dev_stream[num_devices], omp_data_map_t data_map[]) {
	int i, j;
	omp_stream_t * st;

	for (i=0; i<num_devices; i++) {
		st = &dev_stream[i];
		cudaSetDevice(st->dev->sysid);
		cudaStreamSynchronize(st->systream.cudaStream);
		cudaStreamDestroy(st->systream.cudaStream);
	    for (j=0; j<num_maps; j++) {
	    	omp_data_map_t * map = &data_map[i*num_maps+j];
	    	cudaFree(map->map_dev_ptr);
	    	if (map->marshalled_or_not) { /* if this is marshalled and need to free space since this is not useful anymore */
	    		omp_data_map_unmarshal(map);
	    		free(map->map_buffer);
	    	}
	    }
	}
}
/*
 * When call this function, the stream should already synced
 */
void omp_map_device2host(int num_devices, int num_maps, omp_data_map_t *data_map) {
	int i, j;

	for (i=0; i<num_devices; i++) {
		cudaSetDevice(i);
	    //Wait for all operations to finish
	    for (j=0; j<num_maps; j++) {
	    	omp_data_map_t * map = &data_map[i*num_maps+j];
	    	cudaFree(map->map_dev_ptr);
	    	if (map->marshalled_or_not) { /* if this is marshalled and need to free space since this is not useful anymore */
	    		omp_data_map_unmarshal(map);
	    		free(map->map_buffer);
	    	}
	    }
	}
}


size_t xomp_get_maxThreadsPerBlock()
{
  // this often causes oversubscription to the cores supported by GPU SM processors
  //return xomp_getCudaDeviceProp()->maxThreadsPerBlock;
  return 128;
}

size_t xomp_get_max1DBlock(size_t s)
{
  size_t block_num = s/xomp_get_maxThreadsPerBlock();
  if (s % xomp_get_maxThreadsPerBlock()!= 0)
     block_num ++;
  return block_num;
}

void xomp_beyond_block_reduction_float_stream_callback(cudaStream_t stream,  cudaError_t status, void*  userData ) {
	omp_reduction_float_t * rdata = (omp_reduction_float_t*)userData;
	float result = 0.0;
	int i;
	for (i=0; i<rdata->num; i++)
		result += rdata->input[i];
}

/**
 * utilities
 */

/**
 * factor n into dims number of numbers whose multiplication equals to n
 */
void omp_factor(int n, int factor[], int dims) {
	switch (dims) {
	case 1:
	{
		factor[0] = n;
		return;
	}
	case 2:
	{
		switch (n) {
		case 1:
		case 2:
		case 3:
		case 5:
		case 7:
		case 11:
		case 13:
		{
			factor[0] = n;
			factor[1] = 1;
			return;
		}
		case 4:
		case 6:
		case 8:
		case 10:
		case 14:
		{
			factor[0] = n/2;
			factor[1] = 2;
			return;
		}
		case 9:
		case 15:
		{
			factor[0] = n/3;
			factor[1] = 3;
			return;
		}
		case 12:
		case 16:
		{
			factor[0] = n/4;
			factor[1] = 4;
			return;
		}
		}
		break;
	}
	case 3:
	{
		switch (n) {
		case 1:
		case 2:
		case 3:
		case 5:
		case 7:
		case 11:
		case 13:
		{
			factor[0] = n;
			factor[1] = 1;
			factor[2] = 1;
			return;
		}
		case 4:
		case 6:
		{
			factor[0] = n/2;
			factor[1] = 2;
			factor[2] = 1;
			return;
		}
		case 8:
		{
			factor[0] = 2;
			factor[1] = 2;
			factor[2] = 2;
			return;
		}
		default: break;
		}
		break;
	}
	default:
		fprintf(stderr, "more than 3 dimensions are not supported\n");
		break;
	}
}



