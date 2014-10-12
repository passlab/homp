/*
 * homp.c
 * device-independent implementation for the homp.h, 
 * see homp_<devname>.c file for each device-specific implementation 
 *
 *  Created on: Sep 16, 2013
 *      Author: yy8
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/timeb.h>

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

/**
 * notifying the helper threads to work on the offloading specified in off_info arg
 * It always start with copyto and may stops after copyto for target data
 * master is just the thread that will store
 */
void omp_offloading_start(omp_device_t ** targets, int num_targets, omp_offloading_info_t * off_info) {
	int i;
	for (i = 0; i < num_targets; i++) {
		targets[i]->offload_info = off_info;
	}
	pthread_barrier_wait(&off_info->barrier);
}

/* the second step for target data */
void omp_offloading_finish_copyfrom(omp_device_t ** targets, int num_targets, omp_offloading_info_t * off_info) {
	int i;
	off_info->stage = OMP_OFFLOADING_COPYFROM;
	for (i = 0; i < num_targets; i++) {
		targets[i]->offload_info = off_info;
	}
	pthread_barrier_wait(&off_info->barrier);
}

/**
 * TODO: extracting main body of helper_thread_main here or not ?????
 */
void omp_offloading_run(omp_offloading_info_t * off_info, int seqid) {

}

/* helper thread main */
void helper_thread_main(void * arg) {
	omp_device_t * dev = (omp_device_t*)arg;
	int devid = dev->id;
	/*************** wait *******************/
schedule: ;
//	printf("helper threading (devid: %d) waiting ....\n", devid);
	while (dev->offload_info == NULL);

	omp_offloading_info_t * off_info = dev->offload_info;
	omp_grid_topology_t * top = off_info->top;
	int seqid = omp_grid_topology_get_seqid(top, devid);
//	printf("devid: %d --> seqid: %d in top: %X\n", devid, seqid, top);

	omp_offloading_t * off = &off_info->offloadings[seqid];
	off->devseqid = seqid;
	off->dev = dev;
	off->off_info = off_info;
	omp_dev_stream_t * stream = &off->stream;

	if (off_info->stage == OMP_OFFLOADING_COPYFROM) goto offload_stage_copyfrom;
#if defined (OMP_BREAKDOWN_TIMING)
	/* the num_mapped_vars * 2 +4 is the rough number of events needed */
	int num_events = off_info->num_mapped_vars * 2 + 4;
	omp_event_t events[num_events];
	int event_index = 0;

	/* set up stream and event */
	omp_event_init(stream, &events[event_index], OMP_EVENT_HOST_RECORD);
	omp_event_record_start(&events[event_index]);
#endif
	omp_init_stream(dev, stream);

#if defined (OMP_BREAKDOWN_TIMING)
	for (i=1; i<num_events; i++) {
		omp_event_init(stream, &events[i], OMP_EVENT_HOST_DEV_RECORD);
	}
	omp_event_record_stop(&events[event_index++]);
#endif

	/* init data map and dev memory allocation */
	/***************** for each mapped variable has to and tofrom, if it has region mapped to this __ndev_i__ id, we need code here *******************************/
	int i = 0;
offload_stage_copyto: ;
	for (i=0; i<off_info->num_mapped_vars; i++) {
		omp_data_map_info_t * map_info = &off_info->data_map_info[i];
		omp_data_map_t * map = &map_info->maps[seqid];
		omp_data_map_init_map(map, map_info, dev, stream);
		omp_data_map_dist(map, seqid);
		omp_map_buffer_malloc(map);
#if DEBUG_MSG
		omp_print_data_map(map);
#endif

		if (map_info->map_type == OMP_DATA_MAP_TO || map_info->map_type == OMP_DATA_MAP_TOFROM) {
#if defined (OMP_BREAKDOWN_TIMING)
			omp_event_record_start(&events[event_index]);
#endif
			omp_map_memcpy_to_async(map); /* memcpy from host to device */
#if defined (OMP_BREAKDOWN_TIMING)
			omp_event_record_stop(&events[event_index++]);
#endif
		}
	}

offload_stage_kernexe: ;
	if (off_info->type == OMP_OFFLOADING_DATA_CODE || off_info->type == OMP_OFFLOADING_CODE) {
		/* launching the kernel */
		void * args = off_info->args;
		void (*kernel_launcher)(omp_offloading_t *, void *) = off_info->kernel_launcher;
		if (args == NULL) args = off->args;
		if (kernel_launcher == NULL) kernel_launcher = off->kernel_launcher;
#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_start(&events[event_index]);
#endif
		kernel_launcher(off, args);
#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_stop(&events[event_index++]);
#endif
	} else { /* only data offloading, i.e., OMP_OFFLOADING_DATA */
		/* put in the offloading stack */
		dev->offload_stack[dev->offload_stack_top++] = off_info;
		omp_stream_sync(&off->stream, 0);
		dev->offload_info = NULL;
		pthread_barrier_wait(&off_info->barrier);

		goto schedule;
	}

offload_stage_copyfrom: ;
	/* copy back results */
	for (i=0; i<off_info->num_mapped_vars; i++) {
		omp_data_map_info_t * map_info = &off_info->data_map_info[i];
		omp_data_map_t * map = &map_info->maps[seqid];
		if (map_info->map_type == OMP_DATA_MAP_FROM || map_info->map_type == OMP_DATA_MAP_TOFROM) {
#if defined (OMP_BREAKDOWN_TIMING)
			omp_event_record_start(&events[event_index]);
#endif
			omp_map_memcpy_from_async(map); /* memcpy from host to device */
#if defined (OMP_BREAKDOWN_TIMING)
			omp_event_record_stop(&events[event_index++]);
#endif
		}
	}

	/* sync stream to wait for completion */
	omp_sync_cleanup(off);
	pthread_barrier_wait(&off_info->barrier);
#if defined (OMP_BREAKDOWN_TIMING)

#endif

	dev->offload_info = NULL;
	goto schedule;
}

void omp_offloading_init_info(omp_offloading_info_t * info, omp_grid_topology_t * top, omp_device_t **targets, omp_offloading_type_t off_type,
		int num_mapped_vars, omp_data_map_info_t * data_map_info, void (*kernel_launcher)(omp_offloading_t *, void *), void * args) {
	info->top = top;
	info->targets = targets;
	info->type = off_type;
	info->num_mapped_vars = num_mapped_vars;
	info->data_map_info = data_map_info;
	info->kernel_launcher = kernel_launcher;
	info->args = args;

	pthread_barrier_init(&info->barrier, NULL, top->nnodes+1);
}

void omp_data_map_init_dist(omp_data_map_dist_t * dist, long start, long end, omp_data_map_dist_type_t dist_type,
		int halo_left, int halo_right, int halo_cyclic, int topdim) {
	dist->start = start;
	dist->end = end;
	dist->type = dist_type;
	dist->halo_left = halo_left;
	dist->halo_right = halo_right;
	dist->halo_cyclic = halo_cyclic;
	dist->topdim = topdim;
}

/* the dist is straight, i.e.
 * 1. the number of array dimensions is the same as or less than the number of the topology dimensions
 * 2. the full range in each dimension of the array is distributed to the corresponding dimension of the topology
 * 3. the distribution type is the same for all dimensions
 */
void omp_data_map_init_info_dist_straight(omp_data_map_info_t *info, omp_grid_topology_t * top, void * source_ptr, int num_dims, long* dims, int sizeof_element,
		omp_data_map_type_t map_type, omp_data_map_dist_t * dist, omp_data_map_dist_type_t dist_type) {
	int i;
	for (i=0; i<num_dims; i++) {
		dist[i].start = 0;
		dist[i].end = dims[i]-1;
		dist[i].type = dist_type;
		dist[i].topdim = i;
		dist[i].halo_left = dist[i].halo_right = 0;
	}

	omp_data_map_init_info(info, top, source_ptr, num_dims, dims, sizeof_element, map_type, dist);
}

/**
 * caller must meet the requirements of omp_data_map_init_info_dist_straight, plus:
 *
 * The halo region setup is the same in each dimension
 *
 */
void omp_data_map_init_info_dist_straight_with_halo(omp_data_map_info_t *info, omp_grid_topology_t * top, void * source_ptr, int num_dims, long* dims, int sizeof_element,
		omp_data_map_type_t map_type, omp_data_map_dist_t * dist, omp_data_map_dist_type_t dist_type, int halo_left, int halo_right, int halo_cyclic) {
	if (dist_type != OMP_DATA_MAP_DIST_EVEN) {
		fprintf(stderr, "%s: we currently only handle halo region for even distribution of arrays\n", __func__);
	}
	int i;
	for (i=0; i<num_dims; i++) {
		dist[i].start = 0;
		dist[i].end = dims[i]-1;
		dist[i].type = dist_type;
		dist[i].topdim = i;
		dist[i].halo_left = halo_left;
		dist[i].halo_right = halo_right;
		dist[i].halo_cyclic = halo_cyclic;
	}
	omp_data_map_init_info(info, top, source_ptr, num_dims, dims, sizeof_element, map_type, dist);
	info->has_halo_region = 1;
}

void omp_data_map_init_info(omp_data_map_info_t *info, omp_grid_topology_t * top, void * source_ptr, int num_dims, long* dims, int sizeof_element,
		omp_data_map_type_t map_type, omp_data_map_dist_t * dist) {
	if (num_dims > 3) {
		fprintf(stderr, "%d dimension array is not supported in this implementation!\n", num_dims);
		exit(1);
	}
	info->top = top;
	info->source_ptr = source_ptr;
	info->num_dims = num_dims;
	info->dims = dims;
	info->map_type = map_type;
	info->dist = dist;
	info->sizeof_element = sizeof_element;

	info->has_halo_region = 0;
}

/**
 * Given the host pointer (e.g. an array pointer), find the data map of the array onto a specific device,
 * which is provided as a off_loading_t object (the off_loading_t has devseq id as well as pointer to the
 * offloading_info object that a search may need to be performed. If a map_index is provided, the search will
 * be simpler and efficient, otherwise, it may be costly by comparing host_ptr with the source_ptr of each stored map
 * in the offloading stack (call chain)
 */
omp_data_map_t * omp_map_get_map(omp_offloading_t *off, void * host_ptr, int map_index) {
	omp_offloading_info_t * off_info = off->off_info;
	int devseqid = off->devseqid;
	if (map_index >= 0) {  /* the fast and common path */
		omp_data_map_t * map = &off_info->data_map_info[map_index].maps[devseqid];
		if (map->info->source_ptr == host_ptr) return map;
	}

	omp_device_t * dev = off->dev;
	int devid = dev->id;
	int off_stack_i = dev->offload_stack_top + 1;
	/* a search now for all maps */
	do {
		devseqid = omp_grid_topology_get_seqid(off_info->top, devid);
		if (devseqid >= 0) {
			int i; omp_data_map_info_t * dm_info;
			for (i=0; i<off_info->num_mapped_vars; i++) {
				dm_info = &off_info->data_map_info[i];
				if (dm_info->source_ptr == host_ptr) { /* we found */
					return &dm_info->maps[devseqid];
				}
			}
		}
		off_stack_i--;
		off_info = dev->offload_stack[off_stack_i]; /* we actually access the wrong memory location when off_stack_i = -1, but it is a safe place
		and since we are not changing that location */
	} while (off_stack_i >= 0);

	return NULL;
}

void omp_map_add_halo_region(omp_data_map_info_t * info, int dim, int left, int right, int cyclic) {
	info->dist[dim].halo_left = left;
	info->dist[dim].halo_right = right;
	info->dist[dim].halo_cyclic = cyclic;
	info->has_halo_region = 1;
}

/**
 * after initialization, by default, it will perform full map of the original array
 */
void omp_data_map_init_map(omp_data_map_t *map, omp_data_map_info_t * info, omp_device_t * dev, omp_dev_stream_t * stream) {
	map->info = info;
	map->dev = dev;
	map->stream = stream;
	map->marshalled_or_not = 0;
}

/**
 * Apply map to device seqid, seqid is the sequence id of the device in the grid topology
 *
 * do the distribution of array onto the grid topology of devices
 */
void omp_data_map_dist(omp_data_map_t *map, int seqid) {
	omp_data_map_info_t * info = map->info;
	omp_grid_topology_t * top = info->top;
	int coords[top->ndims];
	omp_topology_get_coords(top, seqid, top->ndims, coords);
	int i;
	for (i = 0; i < info->num_dims; i++) { /* process each dimension */
		omp_data_map_dist_t * dist = &info->dist[i];
		long n = dist->end - dist->start + 1;

		int topdim = dist->topdim;
		int topdimcoord = coords[topdim];
		int topdimsize = top->dims[topdim];
		if (dist->type == OMP_DATA_MAP_DIST_EVEN) { /* even distributions */
			/* partition the array region into subregion and save it to the map */
			long remaint = n % topdimsize;
			long esize = n / topdimsize;
		//	printf("n: %d, seqid: %d, map_dist: topdimsize: %d, remaint: %d, esize: %d\n", n, seqid, topdimsize, remaint, esize);
			long map_dim, map_offset;
			if (topdimcoord < remaint) { /* each of the first remaint dev has one more element */
				map_dim = esize + 1;
				map_offset = (esize + 1) * topdimcoord;
			} else {
				map_dim = esize;
				map_offset = esize * topdimcoord + remaint;
			}
			map->map_dim[i] = map_dim;
			map->map_offset[i] = dist->start + map_offset;
			if (info->has_halo_region) {
				map_dim += dist->halo_left + dist->halo_right;
			}
			map->mem_dim[i] = map_dim;
		} else if (dist->type == OMP_DATA_MAP_DIST_FULL) { /* full rang dist */
			map->map_dim[i] = n;
			map->mem_dim[i] = n;
			map->map_offset[i] = dist->start;

		} else {
			fprintf(stderr, "other dist type %d is not yet supported\n",
					dist->type);
			exit(1);
		}
	}

	/* allocate buffer on both host and device, on host, it is the buffer for
	 * marshalled data (move data from non-contiguous memory regions to a contiguous memory region
	 */

}

void omp_map_unmarshal(omp_data_map_t * map) {
	if (!map->marshalled_or_not) return;
	omp_data_map_info_t * info = map->info;
	int sizeof_element = info->sizeof_element;
	int i;
	switch (info->num_dims) {
	case 1: {
		fprintf(stderr,
				"data unmarshall can only do 2-d or 3-d array, currently is 1-d\n");
		break;
	}
	case 2: {
		long region_line_size = map->map_dim[1]*sizeof_element;
		long full_line_size = info->dims[1]*sizeof_element;
		long region_off = 0;
		long full_off = 0;
		char * src_ptr = info->source_ptr + sizeof_element*info->dims[1]*map->map_offset[0] + sizeof_element*map->map_offset[1];
		for (i=0; i<map->map_dim[0]; i++) {
			memcpy(src_ptr+full_off, map->map_buffer+region_off, region_line_size);
			region_off += region_line_size;
			full_off += full_line_size;
		}
		break;
	}
	case 3: {
		break;
	}
	default: {
		fprintf(stderr, "data unmarshall can only do 2-d or 3-d array\n");
		break;
	}
	}
//	printf("total %ld bytes of data unmarshalled\n", region_off);
}

/**
 *  so far works for at most 2D
 */
void omp_map_marshal(omp_data_map_t * map) {
	omp_data_map_info_t * info = map->info;
	int sizeof_element = info->sizeof_element;
	int i;
	map->map_buffer = (void*) malloc(map->map_size);
	switch (info->num_dims) {
	case 1: {
		fprintf(stderr,
				"data marshall can only do 2-d or 3-d array, currently is 1-d\n");
		break;
	}
	case 2: {
		long region_line_size = map->map_dim[1] * sizeof_element;
		long full_line_size = info->dims[1] * sizeof_element;
		long region_off = 0;
		long full_off = 0;
		char * src_ptr = info->source_ptr
				+ sizeof_element * info->dims[1] * map->map_offset[0]
				+ sizeof_element * map->map_offset[1];
		for (i = 0; i < map->map_dim[0]; i++) {
			memcpy(map->map_buffer+region_off, src_ptr+full_off, region_line_size);
			region_off += region_line_size;
			full_off += full_line_size;
		}
		break;
	}
	case 3: {
		break;
	}
	default: {
		fprintf(stderr, "data marshall can only do 2-d or 3-d array\n");
		break;
	}
	}

//	printf("total %ld bytes of data marshalled\n", region_off);
}

/**
 * for a ndims-dimensional array, the dimensions are stored in dims array.
 * Given an element with index stored in idx array, this function return the offset
 * of that element in row-major. E.g. for an array [3][4][5], element [2][2][3] has offset 53
 */
int omp_top_offset(int ndims, int * dims, int * idx) {
	int i;
	int off = 0;
	int mt = 1;
	for (i=ndims-1; i>=0; i--) {
		off += mt * idx[i];
		mt *= dims[i];
	}
	return off;
}

long omp_array_offset(int ndims, long * dims, long * idx) {
	int i;
	long off = 0;
	long mt = 1;
	for (i=ndims-1; i>=0; i--) {
		off += mt * idx[i];
		mt *= dims[i];
	}
	return off;
}

/**
 * this function creates host buffer, if needed, and marshall data to the host buffer,
 *
 * it will also create device memory region (both the array region memory and halo region memory
 */
void omp_map_buffer_malloc(omp_data_map_t * map) {
    int error;
    int i;
	omp_data_map_info_t * info = map->info;
	int sizeof_element = info->sizeof_element;
	long buffer_offset = 0;

	long map_size = map->map_dim[0] * sizeof_element;
	/* TODO: we have not yet handle halo region yet */
	for (i=1; i<info->num_dims; i++) {
		if (map->map_dim[i] != info->dims[i]) {
			/* check the dimension from 1 to the highest, if any one is not the full range of the dimension in the original array,
			 * we have non-contiguous memory space and we need to marshall data
			 */
			map->marshalled_or_not = 1;
		}
		map_size *= map->map_dim[i];

	}
	map->map_size = map_size;
	if (!map->marshalled_or_not) {
		map->map_buffer = info->source_ptr + sizeof_element * omp_array_offset(info->num_dims, map->map_dim, map->map_offset);
	} else {
		omp_map_marshal(map);
	}

	/* we need to allocate device memory, including both the array region TODO: to deal with halo region */
	omp_map_malloc_dev(map);
	if (!info->has_halo_region) return;

#if 0
	omp_data_map_halo_region_info_t * halo_info = info->halo_region;
	omp_data_map_halo_region_mem_t * halo_mem = map->halo_mem;

	for (i=0; i<OMP_NUM_ARRAY_DIMENSIONS; i++) {
		if (info->has_halo_region[i]) { /* there is halo region */
			/* enable CUDA peer memcpy */
			halo_mem = &halo_mem[i];
			int left, right;
			omp_topology_get_neighbors(info->top, map->devsid, (&halo_info[i])->top_dim , (&halo_info[i])->cyclic, &left, &right);
#if DEBUG_MSG
			printf("%d neighbors in dim %d: left: %d, right: %d\n", map->devsid, (&halo_info[i])->top_dim, left, right);
#endif
			int can_access = 0;
			if (left >=0 ) {
				result = cudaDeviceCanAccessPeer(&can_access, map->devsid, left);
                                gpuErrchk(result);
				if (can_access)
                                { 
                                  result = cudaDeviceEnablePeerAccess(left, 0);
                                  if(result != cudaErrorPeerAccessAlreadyEnabled)
                                    gpuErrchk(result);
                                } 
                                else
                                {
                                  printf("Cannot do P2P access from %d to %d, use CPUsync.\n",map->devsid, left);
                                }
				halo_mem->left_map = info->maps[left];
			}
			if (right >=0 ) {
				can_access = 0;
				result = cudaDeviceCanAccessPeer(&can_access, map->devsid, right);
                                gpuErrchk(result);
				if (can_access)
                                {
                                  result = cudaDeviceEnablePeerAccess(right, 0);
                                  if(result != cudaErrorPeerAccessAlreadyEnabled)
                                    gpuErrchk(result);
                                }
                                else
                                {
                                  printf("Cannot do P2P access from %d to %d, use CPUsync\n",map->devsid, right);
                                }
				halo_mem->right_map = info->maps[right];
			}
		}
	}
	halo_mem = map->halo_mem;
	/* TODO: so far only for two dimension array */
	if (info->has_halo_region[0]) { /* there is halo region */
		halo_mem = &halo_mem[0];
		halo_mem->left_in_size = (&halo_info[0])->left*map->map_dim[1]*sizeof_element;
printf("tid %d allocating dim %d halo:%d * %d\n",omp_get_thread_num(),0, (&halo_info[0])->left, map->map_dim[1]);
                if (cudaErrorMemoryAllocation == cudaMalloc(&halo_mem->left_in_ptr, halo_mem->left_in_size)) {
                        gpuErrchk(cudaErrorMemoryAllocation);
                	fprintf(stderr, "cudaMalloc error to allocate mem on device for map %X\n", map);
                }
                if (cudaErrorMemoryAllocation == cudaMalloc(&halo_mem->left_out_ptr, halo_mem->left_in_size)) {
                        gpuErrchk(cudaErrorMemoryAllocation);
                	fprintf(stderr, "cudaMalloc error to allocate mem on device for map %X\n", map);
                }
		/* we calculate from the end of the address */
		halo_mem->right_in_size = (&halo_info[0])->right*map->map_dim[1]*sizeof_element;
                if (cudaErrorMemoryAllocation == cudaMalloc(&halo_mem->right_in_ptr, halo_mem->right_in_size)) {
                        gpuErrchk(cudaErrorMemoryAllocation);
                	fprintf(stderr, "cudaMalloc error to allocate mem on device for map %X\n", map);
                }
                if (cudaErrorMemoryAllocation == cudaMalloc(&halo_mem->right_out_ptr, halo_mem->right_in_size)) {
                        gpuErrchk(cudaErrorMemoryAllocation);
                	fprintf(stderr, "cudaMalloc error to allocate mem on device for map %X\n", map);
                }

//		map->map_dev_ptr = map->mem_dev_ptr;
//		if (halo_mem->left_map)
//			map->map_dev_ptr = map->map_dev_ptr + (halo_info->left*map->mem_dim[1]*sizeof_element);
	}
	halo_mem = map->halo_mem;
	if (info->has_halo_region[1]) { /* there is halo region */
		halo_mem = &halo_mem[1];
printf("allocating dim %d halo:%d * %d\n",1, (map->map_dim[0]+(&halo_info[1])->right+(&halo_info[1])->left), (&halo_info[1])->left);
		halo_mem->left_in_size = (&halo_info[1])->left*(map->map_dim[0]+(&halo_info[1])->right+(&halo_info[1])->left)*sizeof_element;
                if (cudaErrorMemoryAllocation == cudaMalloc(&halo_mem->left_in_ptr, halo_mem->left_in_size)) {
                        gpuErrchk(cudaErrorMemoryAllocation);
                	fprintf(stderr, "cudaMalloc error to allocate mem on device for map %X\n", map);
                }
                if (cudaErrorMemoryAllocation == cudaMalloc(&halo_mem->left_out_ptr, halo_mem->left_in_size)) {
                        gpuErrchk(cudaErrorMemoryAllocation);
                	fprintf(stderr, "cudaMalloc error to allocate mem on device for map %X\n", map);
                }

		halo_mem->right_in_size = (&halo_info[1])->right*(map->map_dim[0]+(&halo_info[1])->right+(&halo_info[1])->left)*sizeof_element;
                if (cudaErrorMemoryAllocation == cudaMalloc(&halo_mem->right_in_ptr, halo_mem->right_in_size)) {
                        gpuErrchk(cudaErrorMemoryAllocation);
                	fprintf(stderr, "cudaMalloc error to allocate mem on device for map %X\n", map);
                }
                if (cudaErrorMemoryAllocation == cudaMalloc(&halo_mem->right_out_ptr, halo_mem->right_in_size)) {
                        gpuErrchk(cudaErrorMemoryAllocation);
                	fprintf(stderr, "cudaMalloc error to allocate mem on device for map %X\n", map);
                }
//		map->map_dev_ptr = map->mem_dev_ptr;
//		if (halo_mem->left_map)
//		  map->map_dev_ptr =  map->map_dev_ptr + (halo_info->left*sizeof_element);
	}
//	if (map->mem_dim[0] != map->map_dim[0]) { /* there is halo region */
//		halo_mem = &halo_mem[0];
//		halo_mem->left_in_ptr = map->mem_dev_ptr;
//		halo_mem->left_in_size = halo_info->left*map->map_dim[1]*sizeof_element;
//		halo_mem->left_out_ptr = map->mem_dev_ptr + halo_mem->left_in_size;
//		/* we calculate from the end of the address */
//		halo_mem->right_in_ptr = map->mem_dev_ptr + map->map_size - halo_info->right*map->map_dim[1]*sizeof_element;
//		halo_mem->right_in_size = halo_info->right*map->map_dim[1]*sizeof_element;
//		halo_mem->right_out_ptr = halo_mem->right_in_ptr - halo_info->left*map->map_dim[1]*sizeof_element;
//
//		if (halo_mem->left_map)
//			map->map_dev_ptr = halo_mem->left_out_ptr;
//		else map->map_dev_ptr = map->mem_dev_ptr;
//	}
//	if (map->mem_dim[1] != map->map_dim[1]) { /* there is halo region */
//		halo_info = &halo_info[1];
//		int buffer_size = sizeof_element*map->map_dim[0]*(halo_info->left+halo_info->right);
//		result = cudaMalloc(&halo_mem->left_in_ptr, buffer_size);
//                gpuErrchk(result);
//		halo_mem->left_out_ptr = halo_mem->left_in_ptr + sizeof_element*halo_info->left*map->map_dim[0];
//		result = cudaMalloc(&halo_mem->right_out_ptr, buffer_size);
//                gpuErrchk(result);
//		halo_mem->right_in_ptr = halo_mem->right_out_ptr + sizeof_element*halo_info->left*map->map_dim[0];
//	}
#endif
}

void omp_print_data_map(omp_data_map_t * map) {
	omp_data_map_info_t * info = map->info;
	printf("devid: %d, MAP: %X, source ptr: %X, dim[0]: %ld, dim[1]: %ld, dim[2]: %ld, map_dim[0]: %ld, map_dim[1]: %ld, map_dim[2]: %ld, "
				"map_offset[0]: %ld, map_offset[1]: %ld, map_offset[2]: %ld, sizeof_element: %d, map_buffer: %X, marshall_or_not: %d,"
				"map_dev_ptr: %X, stream: %X, map_size: %ld\n\n", map->dev->id, map, info->source_ptr, info->dims[0], info->dims[1], info->dims[2],
				map->map_dim[0], map->map_dim[1], map->map_dim[2], map->map_offset[0], map->map_offset[1], map->map_offset[2],
				info->sizeof_element, map->map_buffer, map->marshalled_or_not, map->map_dev_ptr, map->stream, map->map_size);
}

#if 0
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
        cudaError_t result;
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
	omp_data_map_t * right_map = halo_mem->right_map;
	if (left_map != NULL && (from_left_right == 0 || from_left_right == 1)) {
//		cudaMemcpyPeerAsync(halo_mem->left_in_ptr, map->dev->sysid, left_map->halo_mem[0].right_out_ptr, left_map->dev->sysid, halo_mem->left_in_size, map->stream.systream.cudaStream);
		result = cudaMemcpyPeer(halo_mem->left_in_ptr, map->dev->sysid, left_map->halo_mem[0].right_out_ptr, left_map->dev->sysid, halo_mem->left_in_size);
                gpuErrchk(result); 
	}
	if (right_map != NULL && (from_left_right == 0 || from_left_right == 2)) {
//		cudaMemcpyPeerAsync(halo_mem->right_in_ptr, map->dev->sysid, right_map->halo_mem[0].left_out_ptr, right_map->dev->sysid, halo_mem->right_in_size, map->stream.systream.cudaStream);
		result = cudaMemcpyPeer(halo_mem->right_in_ptr, map->dev->sysid, right_map->halo_mem[0].left_out_ptr, right_map->dev->sysid, halo_mem->right_in_size);
                gpuErrchk(result); 
	}
	return;
}

void omp_halo_region_pull_async(omp_data_map_t * map, int dim, int from_left_right) {
        cudaError_t result;
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
	omp_data_map_t * right_map = halo_mem->right_map;
	if (left_map != NULL && (from_left_right == 0 || from_left_right == 1)) {
		int can_access = 0;
	        result = cudaDeviceCanAccessPeer(&can_access, map->devsid,  left_map->devsid);
                gpuErrchk(result);
                if(can_access)
                {
#ifdef DEBUG_MSG
printf("P2P from %d to %d\n",map->devsid,  left_map->devsid);
#endif
		  result = cudaMemcpyPeerAsync(halo_mem->left_in_ptr, map->dev->sysid, left_map->halo_mem[0].right_out_ptr, left_map->dev->sysid, halo_mem->left_in_size, map->stream->systream.cudaStream);
                  gpuErrchk(result); 
                }else
                {
#ifdef DEBUG_MSG
printf("CPUSync from %d to %d\n",map->dev->sysid,  left_map->dev->sysid);
#endif
                  char* CPUbuffer = (char*)malloc(halo_mem->left_in_size);
                  result = cudaMemcpy(CPUbuffer,left_map->halo_mem[0].right_out_ptr,halo_mem->left_in_size,cudaMemcpyDeviceToHost); 
                  gpuErrchk(result); 
                  result = cudaMemcpy(halo_mem->left_in_ptr,CPUbuffer,halo_mem->left_in_size,cudaMemcpyHostToDevice); 
                  gpuErrchk(result);
                  free(CPUbuffer); 
                }
	}
	if (right_map != NULL && (from_left_right == 0 || from_left_right == 2)) {
		int can_access = 0;
	        result = cudaDeviceCanAccessPeer(&can_access, map->devsid, right_map->devsid);
                gpuErrchk(result);
                if(can_access)
                {
#ifdef DEBUG_MSG
printf("P2P from %d to %d\n",map->devsid,  right_map->devsid);
#endif
		  result = cudaMemcpyPeerAsync(halo_mem->right_in_ptr, map->dev->sysid, right_map->halo_mem[0].left_out_ptr, right_map->dev->sysid, halo_mem->right_in_size, map->stream->systream.cudaStream);
                  gpuErrchk(result); 
                }else
                {
#ifdef DEBUG_MSG
printf("CPUSync from %d to %d\n",map->devsid,  right_map->devsid);
#endif
                  char* CPUbuffer = (char*)malloc(halo_mem->right_in_size);
                  result = cudaMemcpy(CPUbuffer,right_map->halo_mem[0].left_out_ptr,halo_mem->right_in_size,cudaMemcpyDeviceToHost); 
                  gpuErrchk(result); 
                  result = cudaMemcpy(halo_mem->right_in_ptr,CPUbuffer,halo_mem->right_in_size,cudaMemcpyHostToDevice); 
                  gpuErrchk(result); 
                  free(CPUbuffer); 
                }
	}
}

#endif /* if 0 */

/**
 * return the mapped range index from the iteration range of the original array
 * e.g. A[128], when being mapped to a device for A[64:64] (from 64 to 128), then, the range 100 to 128 in the original A will be
 * mapped to 36 to 64 in the mapped region of A
 *
 * @param: omp_data_map_t * map: the mapped variable, we should use the original pointer and let the runtime retrieve the map
 * @param: int dim: which dimension to retrieve the range
 * @param: int start: the start index from the original array, if start is -1, use the map_offset_<dim>, which will simply set
 * 		map_start = 0 for obvious reasons
 * @param: int length: the length of the range, if -1, use the mapped dim from the start
 * @param: int * map_start: the mapped start index in the mapped range, if return <0 value, wrong input
 * @param: int * map_length: normally just the length, if lenght == -1, use the map_dim[dim]
 *
 * @return: return the actual offset for map_start from the original iteration range
 *
 * NOTE: the mapped range must be a subset of the range of the specified map in the specified dim
 *
 */
long omp_loop_map_range (omp_data_map_t * map, int dim, long start, long length, long * map_start, long * map_length) {
	if (start <=0) {
		if (length < 0) {
			*map_start = 0;
			*map_length = map->map_dim[dim];
			return map->map_offset[dim];
		} else if (length <= map->map_dim[dim]) {
			*map_start = 0;
			*map_length = length;
			return map->map_offset[dim];
		} else {
			/* error */
		}
	} else { /* start > 0 */
		*map_start = start - map->map_offset[dim];
		*map_length = map->map_dim[dim] - *map_start; /* the max length */
		if (*map_start < 0) { /* out of the range */
			*map_length = -1;
			return -1;
		} else if (length <= *map_length) {
			return start;
		}
	}

	/* out of range */
	*map_start = -1;
	*map_length = -1;
	return -1;
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
    return i;
}

/* return the sequence id of the coord
 */
int omp_grid_topology_get_seqid_coords(omp_grid_topology_t * top, int coords[]) {
/*
	// TODO: currently only for 2D
	if (top->ndims == 1) return top->idmap[coords[0]].devid;
	else if (top->ndims == 2) return top->idmap[coords[0]*top->dims[1] + coords[1]].devid;
	else return -1;
	*/
	return omp_top_offset(top->ndims, top->dims, coords);
}

int omp_grid_topology_get_seqid(omp_grid_topology_t * top, int devid) {
	int i;
	for (i=0; i<top->nnodes; i++)
		if (top->idmap[i] == devid) return i;

	return -1;
}

/**
 * simple and common topology setup, i.e. devid is from 0 to n-1
 *
 */
void omp_grid_topology_init_simple (omp_grid_topology_t * top, omp_device_t **devs, int nnodes, int ndims, int *dims, int *periodic, int * idmap) {
	int i;
	omp_factor(nnodes, dims, ndims);
	for (i=0; i<ndims; i++) {
		periodic[i] = 0;
	}

	for (i=0; i<nnodes; i++) {
		idmap[i] = devs[i]->id;
	}
	top->nnodes = nnodes;
	top->ndims = ndims;
	top->dims = dims;
	top->periodic = periodic;
	top->idmap = idmap;
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
    	*left = omp_grid_topology_get_seqid_coords(top, coords);
    	coords[topdim] = rightdimcoord;
    	*right = omp_grid_topology_get_seqid_coords(top, coords);
    	return;
    } else {
    	if (leftdimcoord < 0) {
    		*left = -1;
    		if (rightdimcoord == dimsize) {
    			*right = -1;
    			return;
    		} else {
    			coords[topdim] = rightdimcoord;
    			*right = omp_grid_topology_get_seqid_coords(top, coords);
    			return;
    		}
    	} else {
    		coords[topdim] = leftdimcoord;
    		*left = omp_grid_topology_get_seqid_coords(top, coords);
    		if (rightdimcoord == dimsize) {
    			*right = -1;
    			return;
    		} else {
    			coords[topdim] = rightdimcoord;
    			*right = omp_grid_topology_get_seqid_coords(top, coords);
    			return;
    		}
    	}
    }
}

/* read timer in second */
double read_timer()
{
	struct timeb tm;
	ftime(&tm);
	return (double)tm.time + (double)tm.millitm/1000.0;
}

/* read timer in ms */
double read_timer_ms()
{
	struct timeb tm;
	ftime(&tm);
	return (double)tm.time * 1000.0 + (double)tm.millitm;
}
