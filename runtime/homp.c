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
omp_device_t * omp_host_dev;
volatile int omp_device_complete = 0;

int omp_num_devices;
volatile omp_printf_turn = 0; /* a simple mechanism to allow multiple dev shepherd threads to print in turn so the output do not scrambled together */
omp_device_type_info_t omp_device_types[OMP_NUM_DEVICE_TYPES] = {
	{OMP_DEVICE_HOST, "OMP_DEVICE_HOST", 1},
	{OMP_DEVICE_NVGPU, "OMP_DEVICE_NVGPU", 0},
	{OMP_DEVICE_ITLMIC, "OMP_DEVICE_ITLMIC", 0},
	{OMP_DEVICE_TIDSP, "OMP_DEVICE_TIDSP", 0},
	{OMP_DEVICE_AMDAPU, "OMP_DEVICE_AMDAPU", 0},
	{OMP_DEVICE_THSIM, "OMP_DEVICE_THSIM", 0},
	{OMP_DEVICE_REMOTE, "OMP_DEVICE_REMOTE", 0},
	{OMP_DEVICE_LOCALPS, "OMP_DEVICE_LOCALPS", 0}
};

/* APIs to support multiple devices: */
char * omp_supported_device_types() { /* return a list of devices supported by the compiler in the format of TYPE1:TYPE2 */
	/* FIXME */
	return "OMP_DEVICE_HOST";
}
omp_device_type_t omp_get_device_type(int devid) {
	return omp_devices[devid].type;
}

char * omp_get_device_type_as_string(int devid) {
	return omp_device_types[omp_devices[devid].type].name;
}

int omp_get_num_devices_of_type(omp_device_type_t type) { /* current omp has omp_get_num_devices(); */
	return omp_device_types[type].num_devs;
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
 * seqid is the sequence id of the device in the top, it is also used as index to access maps
 *
 */
void omp_cleanup(omp_offloading_t * off) {
	int i;
	omp_offloading_info_t * off_info = off->off_info;
	omp_stream_sync(off->stream);
	omp_stream_destroy(off->stream);

	for (i = 0; i < off_info->num_mapped_vars; i++) {
		omp_data_map_t * map = &off_info->data_map_info[i].maps[off->devseqid];
		if (map->map_type == OMP_DATA_MAP_COPY) {
			if (map->mem_noncontiguous) {
				omp_map_unmarshal(map);
				free(map->map_buffer);
			}

			if (omp_device_mem_discrete(map->dev->mem_type)) {
				omp_map_free_dev(map->dev, map->map_dev_ptr);
			}
		}
	}
}

void omp_offloading_init_info(const char * name, omp_offloading_info_t * info, omp_grid_topology_t * top, omp_device_t **targets, int recurring, omp_offloading_type_t off_type,
		int num_mapped_vars, omp_data_map_info_t * data_map_info, void (*kernel_launcher)(omp_offloading_t *, void *), void * args) {
	info->name = name;
	info->top = top;
	info->targets = targets;
	info->recurring = recurring == 0? 0 : 1;
	info->type = off_type;
	if (off_type == OMP_OFFLOADING_DATA) { /* we handle offloading data as two steps, thus a recurruing offloading */
		info->recurring = 1;
	}
	info->num_mapped_vars = num_mapped_vars;
	info->data_map_info = data_map_info;
	info->kernel_launcher = kernel_launcher;
	info->args = args;
	info->halo_x_info = NULL;

	pthread_barrier_init(&info->barrier, NULL, top->nnodes+1);
}

void omp_offloading_append_data_exchange_info (omp_offloading_info_t * info, omp_data_map_halo_exchange_info_t * halo_x_info, int num_maps_halo_x) {
	info->halo_x_info = halo_x_info;
	info->num_maps_halo_x = num_maps_halo_x;
}

void omp_offloading_standalone_data_exchange_init_info(const char * name, omp_offloading_info_t * info,
		omp_grid_topology_t * top, omp_device_t **targets, int recurring, int num_mapped_vars, omp_data_map_info_t * data_map_info, omp_data_map_halo_exchange_info_t * halo_x_info, int num_maps_halo_x ) {
	info->name = name;
	info->top = top;
	info->targets = targets;
	info->recurring = recurring == 0? 0 : 1;
	info->type = OMP_OFFLOADING_STANDALONE_DATA_EXCHANGE;
	info->num_mapped_vars = num_mapped_vars;
	info->data_map_info = data_map_info;
	info->halo_x_info = halo_x_info;
	info->num_maps_halo_x = num_maps_halo_x;

	pthread_barrier_init(&info->barrier, NULL, top->nnodes+1);
}

char * omp_get_device_typename(omp_device_t * dev) {
	int i;
	for (i=0; i<OMP_NUM_DEVICE_TYPES; i++) {
		if (omp_device_types[i].type == dev->type) return omp_device_types[i].name;
	}
	return NULL;
}

void omp_offloading_clear_report_info(omp_offloading_info_t * info) {
	pthread_barrier_destroy(&info->barrier);
#if defined (OMP_BREAKDOWN_TIMING)
	int i;
	for (i=0; i<info->top->nnodes; i++) {
		omp_offloading_t * off = &info->offloadings[i];
		int devid = off->dev->id;
		int devsysid = off->dev->sysid;
		char * type = omp_get_device_typename(off->dev);
		int j;
		printf("\n----------------------- Profiling Report (ms) for Offloading kernel(%s) on %s dev %d (sysid: %d) ---------------------------------------------\n", info->name,  type, devid, devsysid);
		omp_event_print_profile_header();
		for (j=0; j<off->num_events; j++) {
			omp_event_t * ev = &off->events[j];
			if (ev->event_name != NULL) omp_event_print_elapsed(ev);
		}
		printf("----------------------- End Profiling Report for Offloading kernel(%s) on dev: %d -----------------------------------------------\n", info->name, devid);
		free(off->events);
	}
#endif
}

void omp_data_map_init_info(const char * symbol, omp_data_map_info_t *info, omp_grid_topology_t * top, void * source_ptr, int num_dims, long* dims, int sizeof_element,
		omp_data_map_t * maps, omp_data_map_direction_t map_direction, omp_data_map_type_t map_type, omp_data_map_dist_t * dist) {
	if (num_dims > 3) {
		fprintf(stderr, "%d dimension array is not supported in this implementation!\n", num_dims);
		exit(1);
	}
	info->symbol = symbol;
	info->top = top;
	info->source_ptr = source_ptr;
	info->num_dims = num_dims;
	info->dims = dims;
	info->maps = maps; memset(maps, 0, sizeof(omp_data_map_t) * top->nnodes);
	info->map_direction = map_direction;
	info->map_type = map_type;
	info->dist = dist;
	info->halo_info = NULL;
	info->sizeof_element = sizeof_element;
}

void omp_data_map_init_info_with_halo(const char * symbol, omp_data_map_info_t *info, omp_grid_topology_t * top, void * source_ptr, int num_dims, long* dims, int sizeof_element,
		omp_data_map_t * maps, omp_data_map_direction_t map_direction, omp_data_map_type_t map_type, omp_data_map_dist_t * dist, omp_data_map_halo_region_info_t * halo_info) {
	if (num_dims > 3) {
		fprintf(stderr, "%d dimension array is not supported in this implementation!\n", num_dims);
		exit(1);
	}
	int i;
	for (i=0; i<num_dims; i++) {
		halo_info[i].left = 0;
		halo_info[i].right = 0;
		halo_info[i].cyclic = 0;
		halo_info[i].topdim = -1;
	}
	info->symbol = symbol;
	info->top = top;
	info->source_ptr = source_ptr;
	info->num_dims = num_dims;
	info->dims = dims;
	info->maps = maps; memset(maps, 0, sizeof(omp_data_map_t) * top->nnodes);
	info->map_direction = map_direction;
	info->map_type = map_type;
	info->dist = dist;
	info->sizeof_element = sizeof_element;
	info->halo_info = halo_info;
}

void omp_data_map_init_dist(omp_data_map_dist_t * dist, long start, long length, omp_data_map_dist_type_t dist_type, int topdim) {
	dist->start = start;
	dist->length = length;
	dist->type = dist_type;
	dist->topdim = topdim;
}

/* the dist is straight, i.e.
 * 1. the number of array dimensions is the same as or less than the number of the topology dimensions
 * 2. the full range in each dimension of the array is distributed to the corresponding dimension of the topology
 * 3. the distribution type is the same for all dimensions
 */
void omp_data_map_init_info_straight_dist(const char * symbol, omp_data_map_info_t *info, omp_grid_topology_t * top, void * source_ptr, int num_dims, long* dims, int sizeof_element,
		omp_data_map_t * maps, omp_data_map_direction_t map_direction, omp_data_map_type_t map_type, omp_data_map_dist_t * dist, omp_data_map_dist_type_t dist_type) {
	int i;
	for (i=0; i<num_dims; i++) {
		dist[i].start = 0;
		dist[i].length = dims[i];
		dist[i].type = dist_type;
		dist[i].topdim = i;
	}

	omp_data_map_init_info(symbol, info, top, source_ptr, num_dims, dims, sizeof_element, maps, map_direction, map_type, dist);
}

/**
 * caller must meet the requirements of omp_data_map_init_info_dist_straight, plus:
 *
 * The halo region setup is the same in each dimension
 *
 */
void omp_data_map_init_info_straight_dist_and_halo(const char * symbol, omp_data_map_info_t *info, omp_grid_topology_t * top, void * source_ptr, int num_dims, long* dims, int sizeof_element,
		omp_data_map_t * maps, omp_data_map_direction_t map_direction, omp_data_map_type_t map_type, omp_data_map_dist_t * dist, omp_data_map_dist_type_t dist_type, omp_data_map_halo_region_info_t * halo_info, int halo_left, int halo_right, int halo_cyclic) {
	if (dist_type != OMP_DATA_MAP_DIST_EVEN) {
		fprintf(stderr, "%s: we currently only handle halo region for even distribution of arrays\n", __func__);
	}
	int i;
	for (i=0; i<num_dims; i++) {
		dist[i].start = 0;
		dist[i].length = dims[i];
		dist[i].type = dist_type;
		dist[i].topdim = i;
		halo_info[i].left = halo_left;
		halo_info[i].right = halo_right;
		halo_info[i].cyclic = halo_cyclic;
		halo_info[i].topdim = i;
	}
	info->symbol = symbol;
	info->top = top;
	info->source_ptr = source_ptr;
	info->num_dims = num_dims;
	info->dims = dims;
	info->maps = maps; memset(maps, 0, sizeof(omp_data_map_t) * top->nnodes);
	info->map_direction = map_direction;
	info->map_type = map_type;
	info->dist = dist;
	info->sizeof_element = sizeof_element;
	info->halo_info = halo_info;
}

int omp_data_map_has_halo(omp_data_map_info_t * info, int dim) {
	if (info->halo_info == NULL) return 0;
	if (info->halo_info[dim].left == 0 && info->halo_info[dim].right == 0) return 0;
	return 1;
}

int omp_data_map_get_halo_left_devseqid(omp_data_map_t * map, int dim) {
	if (omp_data_map_has_halo(map->info, dim) == 0) return -1;
	return (map->halo_mem[dim].left_dev_seqid);
}

int omp_data_map_get_halo_right_devseqid(omp_data_map_t * map, int dim) {
	if (omp_data_map_has_halo(map->info, dim) == 0) return -1;
	return (map->halo_mem[dim].right_dev_seqid);
}

static omp_data_map_t * omp_map_get_map_from_cache (omp_offloading_t *off, void * host_ptr) {
	int i;
	for (i=0; i<off->num_maps; i++) {
		omp_data_map_t * map = off->map_cache[i].map;
		if (map->info->source_ptr == host_ptr) return map;
	}

	return NULL;
}

void omp_offload_append_map_to_cache (omp_offloading_t *off, omp_data_map_t *map, int inherited) {
	if (off->num_maps >= OFF_MAP_CACHE_SIZE) { /* error, report */
		fprintf(stderr, "map cache is full for off (%X), cannot add map %X\n", off, map);
		exit(1);
	}
	off->map_cache[off->num_maps].map = map;
	off->map_cache[off->num_maps].inherited = inherited;

	off->num_maps++;
}


int omp_map_is_map_inherited(omp_offloading_t *off, omp_data_map_t *map) {
	int i;
	for (i=0; i<off->num_maps; i++) {
		if (off->map_cache[i].map == map) return off->map_cache[i].inherited;
	}

	fprintf(stderr, "map %X is not in the map cache of off %X, wrong info and misleading return value of this function\n", map, off);
	return 0;

}


/* get map from inheritance (off stack) */
omp_data_map_t * omp_map_get_map_inheritance (omp_device_t * dev, void * host_ptr) {
	int i;
	for (i=dev->offload_stack_top; i>=0; i--) {
		omp_offloading_t * ancestor_off = dev->offload_stack[i];
		omp_data_map_t * map = omp_map_get_map_from_cache(ancestor_off, host_ptr);
		if (map != NULL) {
			return map;
		}
	}
	return NULL;
}

/**
 * Given the host pointer (e.g. an array pointer), find the data map of the array onto a specific device,
 * which is provided as a off_loading_t object (the off_loading_t has devseq id as well as pointer to the
 * offloading_info object that a search may need to be performed. If a map_index is provided, the search will
 * be simpler and efficient, otherwise, it may be costly by comparing host_ptr with the source_ptr of each stored map
 * in the offloading stack (call chain)
 *
 * The call also put a map into off->map_cache if it is not in the cache
 */
omp_data_map_t * omp_map_get_map(omp_offloading_t *off, void * host_ptr, int map_index) {
	/* STEP 1: search from the cache first */
	omp_data_map_t * map = omp_map_get_map_from_cache(off, host_ptr);
	if (map != NULL) {
		return map;
	}

	/* STEP 2: if not in cache, do quick search if given by a map_index, and then do a thorough search. put in cache if finds it */
	omp_offloading_info_t * off_info = off->off_info;
	int devseqid = off->devseqid;
	if (map_index >= 0 && map_index <= off_info->num_mapped_vars) {  /* the fast and common path */
		map = &off_info->data_map_info[map_index].maps[devseqid];
		if (host_ptr == NULL || map->info->source_ptr == host_ptr) {
			/* append to the off->map_cache */
			omp_offload_append_map_to_cache(off, map, 0);
			return map;
		}
	} else { /* thorough search for all the mapped variables */
		int i; omp_data_map_info_t * dm_info;
		for (i=0; i<off_info->num_mapped_vars; i++) {
			dm_info = &off_info->data_map_info[i];
			if (dm_info->source_ptr == host_ptr) { /* we found */
				map = &dm_info->maps[devseqid];
				//printf("find a match: %X\n", host_ptr);
				omp_offload_append_map_to_cache(off, map, 0);
				//omp_print_data_map(map);
				return map;
			}
		}
	}

	/* STEP 3: seach the offloading stack if this inherits data map from previous data offloading */
//	printf("omp_map_get_map: off: %X, off_info: %X, host_ptr: %X\n", off, off_info, host_ptr);
	map = omp_map_get_map_inheritance (off->dev, host_ptr);
	if (map != NULL) omp_offload_append_map_to_cache(off, map, 1);

	return map;
}

void omp_map_add_halo_region(omp_data_map_info_t * info, int dim, int left, int right, int cyclic) {
	info->halo_info[dim].left = left;
	info->halo_info[dim].right = right;
	info->halo_info[dim].cyclic = cyclic;
	info->halo_info[dim].topdim = info->dist[dim].topdim;
}

/**
 * after initialization
 */
void omp_data_map_init_map(omp_data_map_t *map, omp_data_map_info_t * info, omp_device_t * dev, omp_dev_stream_t * stream, omp_offloading_t * off) {
	map->info = info;
	map->dev = dev;
	map->stream = stream;
	map->mem_noncontiguous = 0;
	map->map_type = info->map_type;
	map->access_level = OMP_DATA_MAP_ACCESS_LEVEL_1;

	if (map->map_type == OMP_DATA_MAP_AUTO) {
		if (omp_device_mem_discrete(dev->mem_type)) {
			map->map_type = OMP_DATA_MAP_COPY;
			//printf("COPY data map: %X\n", map);
		} else { /* we can make it shared and we will do it */
			map->map_type = OMP_DATA_MAP_SHARED;
			//printf("SHARED data map: %X\n", map);
		}
	} else if (map->map_type == OMP_DATA_MAP_SHARED && omp_device_mem_discrete(dev->mem_type)) {
		fprintf(stderr, "direct sharing data between host and the dev: %d is not possible with discrete mem space, we use COPY approach now\n", dev->id);
		map->map_type = OMP_DATA_MAP_COPY;
	}
}

/* forward declaration to suppress compiler warning */
void omp_topology_get_neighbors(omp_grid_topology_t * top, int seqid, int topdim, int cyclic, int* left, int* right);
/**
 * Apply map to device seqid, seqid is the sequence id of the device in the grid topology
 *
 * do the distribution of array onto the grid topology of devices
 */
void omp_data_map_dist(omp_data_map_t *map, int seqid, omp_offloading_t * off) {
	omp_data_map_info_t * info = map->info;
	omp_grid_topology_t * top = info->top;
	int coords[top->ndims];
	omp_topology_get_coords(top, seqid, top->ndims, coords);
	int i;
	for (i = 0; i < info->num_dims; i++) { /* process each dimension */
		omp_data_map_dist_t * dist = &info->dist[i];
		long n = dist->length;

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
			if (info->halo_info != NULL) { /** here we also need to deal with boundary, the first one that has no left halo and the last one that has no right halo for non-cyclic case */
				omp_data_map_halo_region_info_t * halo = &info->halo_info[i];

				if (halo->left != 0 || halo->right != 0) { /* we have halo in this dimension */
					omp_data_map_halo_region_mem_t * halo_mem = &map->halo_mem[i];
					int *left = &halo_mem->left_dev_seqid;
					int *right = &halo_mem->right_dev_seqid;
					omp_topology_get_neighbors(info->top, seqid, halo->topdim, halo->cyclic, left, right);
					//printf("dev: %d, map_info: %X, %d neighbors in dim %d: left: %d, right: %d\n", map->dev->id, map->info, seqid, topdim, left, right);
					if (*left >=0 ) {
						map_offset = map_offset - halo->left;
						map_dim += halo->left;
					}
					if (*right >=0 ) {
						map_dim += halo->right;
					}
				}
			}
			map->map_offset[i] = dist->start + map_offset;
			map->map_dim[i] = map_dim;
		} else if (dist->type == OMP_DATA_MAP_DIST_FULL) { /* full rang dist */
			map->map_dim[i] = n;
			map->map_offset[i] = dist->start;
		} else if (dist->type == OMP_DATA_MAP_DIST_BALANCE) {
			/* performance model based data distribution */

		} else {
			fprintf(stderr, "other dist type %d is not yet supported\n",
					dist->type);
			exit(1);
		}
	}

	map->access_level = OMP_DATA_MAP_ACCESS_LEVEL_2;

	/* allocate buffer on both host and device, on host, it is the buffer for
	 * marshalled data (move data from non-contiguous memory regions to a contiguous memory region
	 */

}

void omp_map_unmarshal(omp_data_map_t * map) {
	if (!map->mem_noncontiguous) return;
	omp_data_map_info_t * info = map->info;
	int sizeof_element = info->sizeof_element;
	int i;
	switch (info->num_dims) {
	case 1: {
		fprintf(stderr, "data unmarshall can only do 2-d or 3-d array, currently is 1-d\n");
		break;
	}
	case 2: {
		long region_line_size = map->map_dim[1]*sizeof_element;
		long full_line_size = info->dims[1]*sizeof_element;
		long region_off = 0;
		long full_off = 0;
		char * src_ptr = &info->source_ptr[sizeof_element*info->dims[1]*map->map_offset[0] + sizeof_element*map->map_offset[1]];
		for (i=0; i<map->map_dim[0]; i++) {
			memcpy((void*)&src_ptr[full_off], (void*)&map->map_buffer[region_off], region_line_size);
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
		char * src_ptr = &info->source_ptr[sizeof_element * info->dims[1] * map->map_offset[0]
				+ sizeof_element * map->map_offset[1]];
		for (i = 0; i < map->map_dim[0]; i++) {
			memcpy((void*)&map->map_buffer[region_off], (void*)&src_ptr[full_off], region_line_size);
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
void omp_map_buffer(omp_data_map_t * map, omp_offloading_t * off) {
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
			map->mem_noncontiguous = 1;
		}
		map_size *= map->map_dim[i];

	}
	map->map_size = map_size;
	map->map_buffer = &info->source_ptr[sizeof_element * omp_array_offset(info->num_dims, info->dims, map->map_offset)];

	/* so far, for noncontiguous mem space, we will do copy */
	if (map->map_type == OMP_DATA_MAP_SHARED) {
		if (map->mem_noncontiguous) {
			map->map_type = OMP_DATA_MAP_COPY;
		}
	}

	if (map->map_type == OMP_DATA_MAP_COPY) {
		if (map->mem_noncontiguous) {
			omp_map_marshal(map);
			if (omp_device_mem_discrete(map->dev->mem_type)) {
				map->map_dev_ptr = omp_map_malloc_dev(map->dev, map->map_size);
			} else map->map_dev_ptr = map->map_buffer;
		} else {
			map->map_dev_ptr = omp_map_malloc_dev(map->dev, map->map_size);
		}
	} else map->map_dev_ptr = map->map_buffer;

	map->access_level = OMP_DATA_MAP_ACCESS_LEVEL_2;

	if (info->halo_info == NULL) return;

	/** memory management for halo region */

	/** TODO:  this is only for 2-d array
	 * For 3-D array, it becomes more complicated
	 * For 1-D array, we will not need allocate buffer for the halo region. but we still need to update the left/right in/out ptr to the
	 * correct location.
	 *
	 * There may be a uniformed way of dealing with this (for at least 1/2/3-d array
	 *
	 * The halo memory management is use a attached approach, i.e. the halo region is part of the main computation subregion, and those
	 * left/right in/out are buffers for gathering and scattering halo region elements to its correct location.
	 *
	 * We may use a detached approach, i.e. just use the halo buffer for computation, however, that will involve more complicated
	 * array index calculation.
	 */

	/*********************************************************************************************************************************************************/
	/************************* Barrier will be needed among all participating devs since we are now using neighborhood devs **********************************/
	/*********************************************************************************************************************************************************/

	/** FIXME: so far, we only have 2-d row distribution working, i.e. no need to allocate halo region buffer */
//        BEGIN_SERIALIZED_PRINTF(off->devseqid);
	for (i=0; i<info->num_dims; i++) {
		omp_data_map_halo_region_info_t * halo_info = &info->halo_info[i];
		if (halo_info->left != 0 || halo_info->right != 0) { /* there is halo region in this dimension */
			omp_data_map_halo_region_mem_t * halo_mem = &map->halo_mem[i];
			if (halo_mem->left_dev_seqid >= 0) {
				if (i==0 && !map->mem_noncontiguous && info->num_dims == 2) { /* this is only for row-dist, 2-d array, i.e. no marshalling */
					halo_mem->left_in_size = halo_info->left*map->map_dim[1]*sizeof_element;
					halo_mem->left_in_ptr = map->map_dev_ptr;
					halo_mem->left_out_size = halo_info->right*map->map_dim[1]*sizeof_element;
					halo_mem->left_out_ptr = &((char*)map->map_dev_ptr)[halo_mem->left_in_size];
#if CORRECTNESS_CHECK

					printf("dev: %d, halo left in size: %d, left in ptr: %X, left out size: %d, left out ptr: %X\n", off->devseqid,
							halo_mem->left_in_size,halo_mem->left_in_ptr,halo_mem->left_out_size,halo_mem->left_out_ptr);
#endif
				} else {
					fprintf(stderr, "current dist/map setting does not support halo region\n");
				}

				omp_device_t * leftdev = off->off_info->targets[halo_mem->left_dev_seqid];
				if (!omp_map_enable_memcpy_DeviceToDevice(leftdev, map->dev)) { /* no peer2peer access available, use host relay */
					halo_mem->left_in_host_relay_ptr = (char*)malloc(halo_mem->left_in_size); /** FIXME, mem leak here and we have not thought where to free */
					halo_mem->left_in_data_in_relay_pushed = 0;
					halo_mem->left_in_data_in_relay_pulled = 0;

					//printf("dev: %d, map: %X, left: %d, left host relay buffer allocated\n", off->devseqid, map, halo_mem->left_dev_seqid);
				} else {
					//printf("dev: %d, map: %X, left: %d, left dev: p2p enabled\n", off->devseqid, map, halo_mem->left_dev_seqid);
					halo_mem->left_in_host_relay_ptr = NULL;
				}
			}
			if (halo_mem->right_dev_seqid >= 0) {
				if (i==0 && !map->mem_noncontiguous && info->num_dims == 2) { /* this is only for row-dist, 2-d array, i.e. no marshalling */
					halo_mem->right_in_size = halo_info->right*map->map_dim[1]*sizeof_element;
					halo_mem->right_in_ptr = &((char *)map->map_dev_ptr)[map->map_size - halo_mem->right_in_size];
					halo_mem->right_out_size = halo_info->left*map->map_dim[1]*sizeof_element;
					halo_mem->right_out_ptr = &((char *)halo_mem->right_in_ptr)[0 - halo_mem->right_out_size];
#if CORRECTNESS_CHECK
					printf("dev: %d, halo right in size: %d, right in ptr: %X, right out size: %d, right out ptr: %X\n", off->devseqid,
												halo_mem->right_in_size,halo_mem->right_in_ptr,halo_mem->right_out_size,halo_mem->right_out_ptr);
#endif
				} else {
					fprintf(stderr, "current dist/map setting does not support halo region\n");
				}
				omp_device_t * rightdev = off->off_info->targets[halo_mem->right_dev_seqid];
				if (!omp_map_enable_memcpy_DeviceToDevice(rightdev, map->dev)) { /* no peer2peer access available, use host relay */
					halo_mem->right_in_host_relay_ptr = (char*)malloc(halo_mem->right_in_size); /** FIXME, mem leak here and we have not thought where to free */
					halo_mem->right_in_data_in_relay_pushed = 0;
					halo_mem->right_in_data_in_relay_pulled = 0;

					//printf("dev: %d, map: %X, right: %d, right host relay buffer allocated\n", off->devseqid, map, halo_mem->right_dev_seqid);
				} else {
					//printf("dev: %d, map: %X, right: %d, right host p2p enabled\n", off->devseqid, map, halo_mem->right_dev_seqid);
					halo_mem->right_in_host_relay_ptr = NULL;
				}
			}
		}
	}
//	END_SERIALIZED_PRINTF();

	map->access_level = OMP_DATA_MAP_ACCESS_LEVEL_4;
#if 0
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
	}
#endif
}

void omp_print_data_map(omp_data_map_t * map) {
	omp_data_map_info_t * info = map->info;
	printf("devid: %d, MAP: %X, source ptr: %X, dim[0]: %ld, dim[1]: %ld, dim[2]: %ld, map_dim[0]: %ld, map_dim[1]: %ld, map_dim[2]: %ld, "
				"map_offset[0]: %ld, map_offset[1]: %ld, map_offset[2]: %ld, sizeof_element: %d, map_buffer: %X, marshall_or_not: %d,"
				"map_dev_ptr: %X, stream: %X, map_size: %ld\n\n", map->dev->id, map, info->source_ptr, info->dims[0], info->dims[1], info->dims[2],
				map->map_dim[0], map->map_dim[1], map->map_dim[2], map->map_offset[0], map->map_offset[1], map->map_offset[2],
				info->sizeof_element, map->map_buffer, map->mem_noncontiguous, map->map_dev_ptr, map->stream, map->map_size);
}

void omp_print_off_maps(omp_offloading_t * off) {
	int i;
	printf("off %X maps: ", off);
	for (i=0; i<off->num_maps; i++) printf("%X, ", off->map_cache[i]);
	printf("\n");
}

/* do a halo regin pull for data map of devid. If top is not NULL, devid will be translated to coordinate of the
 * virtual topology and the halo region pull will be based on this coordinate.
 * @param: int dim[ specify which dimension to do the halo region update.
 *      If dim < 0, do all the update of map dimensions that has halo region
 * @param: from_left_right, to do in which direction

 *
 */
void omp_halo_region_pull(omp_data_map_t * map, int dim, omp_data_map_exchange_direction_t from_left_right) {
	omp_data_map_info_t * info = map->info;
	/*FIXME: let us only handle 2-D array now */
	if (info->num_dims != 2 || dim != 0 || map->mem_noncontiguous) {
		fprintf(stderr, "we only handle 2-d array, dist/halo at 0-d and non-marshalling so far!\n");
		omp_print_data_map(map);
		return;
	}

	omp_data_map_halo_region_mem_t * halo_mem = &map->halo_mem[dim];
#if CORRECTNESS_CHECK
    BEGIN_SERIALIZED_PRINTF(map->dev->id);
	printf("dev: %d, map: %X, left: %d, right: %d\n", map->dev->id, map, halo_mem->left_dev_seqid, halo_mem->right_dev_seqid);
#endif

	if (halo_mem->left_dev_seqid >= 0 && (from_left_right == OMP_DATA_MAP_EXCHANGE_FROM_LEFT_ONLY || from_left_right == OMP_DATA_MAP_EXCHANGE_FROM_LEFT_RIGHT)) { /* pull from left */
		omp_data_map_t * left_map = &info->maps[halo_mem->left_dev_seqid];
		omp_data_map_halo_region_mem_t * left_halo_mem = &left_map->halo_mem[dim];

		/* if I need to push right_out data to the host relay buffer for the left_map, I should do it first */
		if (left_halo_mem->right_in_host_relay_ptr != NULL) {
			/* wait make sure the data in the right_in_host_relay buffer is already pulled */
			while (left_halo_mem->right_in_data_in_relay_pushed > left_halo_mem->right_in_data_in_relay_pulled);
			omp_map_memcpy_from((void*)left_halo_mem->right_in_host_relay_ptr, (void*)halo_mem->left_out_ptr, map->dev, halo_mem->left_out_size);
			left_halo_mem->right_in_data_in_relay_pushed ++;
		} else {
			/* do nothing here because the left_map helper thread will do a direct device-to-device pull */
		}

		if (halo_mem->left_in_host_relay_ptr == NULL) { /* no need host relay */
			omp_map_memcpy_DeviceToDevice((void*)halo_mem->left_in_ptr, map->dev, (void*)left_halo_mem->right_out_ptr, left_map->dev, halo_mem->left_in_size);
#if CORRECTNESS_CHECK
			printf("dev: %d, dev2dev memcpy from left: %X <----- %X\n", map->dev->id, halo_mem->left_in_ptr, left_halo_mem->right_out_ptr);
#endif
		} else { /* need host relay */
			/*
			omp_set_current_device_dev(left_map->dev);
			omp_map_memcpy_from(halo_mem->left_in_host_relay_ptr, left_map->halo_mem[dim].right_out_ptr, left_map->dev, halo_mem->left_in_size);
			omp_set_current_device_dev(map->dev);
			*/
			while (halo_mem->left_in_data_in_relay_pushed <= halo_mem->left_in_data_in_relay_pulled); /* wait for the data to be ready in the relay buffer on host */
			/* wait for the data in the relay buffer is ready */
			omp_map_memcpy_to((void*)halo_mem->left_in_ptr, map->dev, (void*)halo_mem->left_in_host_relay_ptr, halo_mem->left_in_size);
			halo_mem->left_in_data_in_relay_pulled++;
		}
	}
	if (halo_mem->right_dev_seqid >= 0 && (from_left_right == OMP_DATA_MAP_EXCHANGE_FROM_RIGHT_ONLY || from_left_right == OMP_DATA_MAP_EXCHANGE_FROM_LEFT_RIGHT)) {
		omp_data_map_t * right_map = &info->maps[halo_mem->right_dev_seqid];
		omp_data_map_halo_region_mem_t * right_halo_mem = &right_map->halo_mem[dim];

		/* if I need to push left_out data to the host relay buffer for the right_map, I should do it first */
		if (right_halo_mem->left_in_host_relay_ptr != NULL) {
			while (right_halo_mem->left_in_data_in_relay_pushed > right_halo_mem->left_in_data_in_relay_pulled);
			omp_map_memcpy_from((void*)right_halo_mem->left_in_host_relay_ptr, (void*)halo_mem->right_out_ptr, map->dev, halo_mem->right_out_size);
			right_halo_mem->left_in_data_in_relay_pushed ++;
		} else {
			/* do nothing here because the left_map helper thread will do a direct device-to-device pull */
		}

		if (halo_mem->right_in_host_relay_ptr == NULL) {
			omp_map_memcpy_DeviceToDevice((void*)halo_mem->right_in_ptr, map->dev, (void*)right_halo_mem->left_out_ptr, right_map->dev, halo_mem->right_in_size);
#if CORRECTNESS_CHECK
			printf("dev: %d, dev2dev memcpy from right: %X <----- %X\n", map->dev->id, halo_mem->right_in_ptr, right_halo_mem->left_out_ptr);
#endif

		} else {
			/*
			omp_set_current_device_dev(right_map->dev);
			omp_map_memcpy_from(halo_mem->right_in_host_relay_ptr, right_map->halo_mem[dim].left_out_ptr, right_map->dev, halo_mem->right_in_size);
			omp_set_current_device_dev(map->dev);
			*/
			while (halo_mem->right_in_data_in_relay_pushed <= halo_mem->right_in_data_in_relay_pulled); /* wait for the data to be ready in the relay buffer on host */
			omp_map_memcpy_to((void*)halo_mem->right_in_ptr, map->dev, (void*)halo_mem->right_in_host_relay_ptr, halo_mem->right_in_size);
			halo_mem->right_in_data_in_relay_pulled++;
		}
	}
#if CORRECTNESS_CHECK
	END_SERIALIZED_PRINTF();
#endif
	return;
}

#if 0
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

#endif

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

void omp_topology_get_neighbors(omp_grid_topology_t * top, int seqid, int topdim, int cyclic, int* left, int* right) {
	if (seqid < 0 || seqid > top->nnodes) {
		*left = -1;
		*right = -1;
		return;
	}
	int coords[top->ndims];
	omp_topology_get_coords(top, seqid, top->ndims, coords);

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
