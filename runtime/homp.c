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
#include <time.h>
#include "homp.h"

#define num_allowed_dist_policies 5
#define default_dist_policy_index 0
omp_dist_policy_argument_t omp_dist_policy_args[num_allowed_dist_policies] = {
		{OMP_DIST_POLICY_BLOCK,         		-1,     "OMP_DIST_POLICY_BLOCK", 				"BLOCK"},
		{OMP_DIST_POLICY_SCHED_DYNAMIC, 		100,    "OMP_DIST_POLICY_SCHED_DYNAMIC",   		"SCHED_DYNAMIC"},
		{OMP_DIST_POLICY_SCHED_GUIDED, 			100,	"OMP_DIST_POLICY_SCHED_GUIDED",  		"SCHED_GUIDED"},
		{OMP_DIST_POLICY_SCHED_PROFILE_AUTO,	100,	"OMP_DIST_POLICY_SCHED_PROFILE_AUTO",	"SCHED_PROFILE_AUTO"},
		{OMP_DIST_POLICY_SCHED_AUTO,        	-1, 	"OMP_DIST_POLICY_SCHED_AUTO",   		"SCHED_AUTO"},
};

void omp_print_dist_policy_options() {
	int i;
	printf("==========================================================================================================================\n");
	printf("Usage: Loop distribution policy can be specified by setting LOOP_DIST_POLICY environment variable to one of the following:\n");
	printf("  The first number is the policy id, and the second number is the chunk size. If it is -1, the chunk size is not used.\n");
	for (i=0; i < num_allowed_dist_policies; i++) {
		printf("\t%d,<n>:\t%s, full name: %s, default n: %d\n", omp_dist_policy_args[i].type, omp_dist_policy_args[i].shortname, omp_dist_policy_args[i].name, omp_dist_policy_args[i].chunk);
	}
	i= default_dist_policy_index;
	printf("\t--------------------------------------------------------------------------------------------------------------------\n");
	printf("\tDefault: %d,%d:\t%s, full name: %s\n", omp_dist_policy_args[i].type, omp_dist_policy_args[i].chunk, omp_dist_policy_args[i].shortname, omp_dist_policy_args[i].name);
	printf("==========================================================================================================================\n");

}

omp_dist_policy_t omp_read_dist_policy_options(int * chunk_size) {
	omp_dist_policy_t p;
	char *dist_policy_str = getenv("LOOP_DIST_POLICY");
	int i = num_allowed_dist_policies;
	if (dist_policy_str != NULL) {
		sscanf(dist_policy_str, "%d,%d", &p, chunk_size);
		for (i = 0; i < num_allowed_dist_policies; i++) {
			if (omp_dist_policy_args[i].type == p) break;
		}
	}
	if (i == num_allowed_dist_policies) {
		i = default_dist_policy_index;
		p = omp_dist_policy_args[i].type;
		*chunk_size = omp_dist_policy_args[i].chunk;
	}
//	printf("Dist policy: %d,%d: %s, full name: %s\n", omp_dist_policy_args[i].type, *chunk_size, omp_dist_policy_args[i].shortname, omp_dist_policy_args[i].name);
	printf("Dist policy and chunk size are set and stored in the following two global variables that can be used in the program.\n");
	printf("\tLOOP_DIST_POLICY     = %d (%s)\n", omp_dist_policy_args[i].type, omp_dist_policy_args[i].name);
	printf("\tLOOP_DIST_CHUNK_SIZE = %d\n", *chunk_size);
	printf("==========================================================================================================================\n\n");

	return p;
}

/* we use as minimum malloc as posibble, so the mem layout is as follows:
 * -----------------------------------------
 * offloading_info_t[1]
 * omp_data_map_info[num_maps]
 * omp_offloading_t[num_targets]
 * omp_data_map_t[num_maps][num_targets]
 * -----------------------------------------
 */
omp_offloading_info_t * omp_offloading_init_info(const char *name, omp_grid_topology_t *top, int recurring,
												 omp_offloading_type_t off_type, int num_maps,
												 void (*kernel_launcher)(omp_offloading_t *, void *), void *args,
												 int loop_nest_depth) {
	int total_size = sizeof(omp_offloading_info_t) + num_maps*sizeof(omp_data_map_info_t) +
					 top->nnodes * sizeof(omp_offloading_t) + sizeof(struct omp_data_map)*num_maps*top->nnodes;
	char * buffer = calloc(total_size, sizeof(char));
	omp_offloading_info_t * info = (omp_offloading_info_t*) buffer;
	info->data_map_info = (omp_data_map_info_t*)&buffer[sizeof(omp_offloading_info_t)];
	info->offloadings = (omp_offloading_t*) &buffer[sizeof(omp_offloading_info_t) +
													num_maps*sizeof(omp_data_map_info_t)];
	omp_data_map_t * maps = (omp_data_map_t*) &buffer[sizeof(omp_offloading_info_t) +
													  num_maps*sizeof(omp_data_map_info_t) + top->nnodes  * sizeof(omp_offloading_t)];

	int i;
	int offsetofmaps = 0;
	for (i=0; i<num_maps; i++) {
		info->data_map_info[i].maps = &maps[offsetofmaps];
		offsetofmaps += top->nnodes;
	}
	info->name = name;
	info->top = top;
	info->count = recurring == 0? 0 : 1;
	info->type = off_type;
	if (off_type == OMP_OFFLOADING_DATA) { /* we handle offloading data as two steps, thus a recurruing offloading */
		info->count = 1;
	}
	info->num_mapped_vars = num_maps;
	info->kernel_launcher = kernel_launcher;
	info->args = args;
	info->halo_x_info = NULL;
	info->start_time = 0;
	info->loop_depth = loop_nest_depth;
	for (i=0; i<loop_nest_depth; i++) {
		info->loop_dist_info[i].target_type = OMP_DIST_TARGET_LOOP_ITERATION;
		info->loop_dist_info[i].target = info;
		info->loop_dist_info[i].target_dim = i;
	}

	//memset(info->offloadings, NULL, sizeof(omp_offloading_t)*top->nnodes);
	/* we move the off initialization from each device thread here, i.e. we serialized this */
	int loop_dist_done = 0;
	if (info->type == OMP_OFFLOADING_DATA || info->type == OMP_OFFLOADING_STANDALONE_DATA_EXCHANGE)
		loop_dist_done = 1;
	for (i=0; i<top->nnodes; i++) {
		omp_offloading_t * off = &info->offloadings[i];
		omp_device_t * dev = &omp_devices[top->idmap[i]];
		off->devseqid = i;
		off->dev = dev;
		off->off_info = info;
		off->num_maps = 0;
		off->stage = OMP_OFFLOADING_INIT;
		off->events == NULL;
		off->num_events = 0;
		off->loop_dist_done = loop_dist_done;
	}

	pthread_barrier_init(&info->barrier, NULL, top->nnodes+1);
	pthread_barrier_init(&info->inter_dev_barrier, NULL, top->nnodes);
	return info;
}

void omp_offloading_append_data_exchange_info (omp_offloading_info_t * info, omp_data_map_halo_exchange_info_t * halo_x_info, int num_maps_halo_x) {
	info->halo_x_info = halo_x_info;
	info->num_maps_halo_x = num_maps_halo_x;
}

omp_offloading_info_t * omp_offloading_standalone_data_exchange_init_info(const char *name, omp_grid_topology_t *top, int recurring,
																		  omp_data_map_halo_exchange_info_t *halo_x_info,
																		  int num_maps_halo_x) {
	int total_size = sizeof(omp_offloading_info_t) + top->nnodes * sizeof(omp_offloading_t);
	char *buffer = calloc(total_size, sizeof(char));
	omp_offloading_info_t *info = (omp_offloading_info_t *) buffer;
	info->offloadings = (omp_offloading_t *) &buffer[sizeof(omp_offloading_info_t)];

	info->name = name;
	info->top = top;
	info->count = recurring == 0? 0 : 1;
	info->type = OMP_OFFLOADING_STANDALONE_DATA_EXCHANGE;
	info->num_mapped_vars = 0;
	info->halo_x_info = halo_x_info;
	info->num_maps_halo_x = num_maps_halo_x;

	int i;

	//memset(info->offloadings, NULL, sizeof(omp_offloading_t)*top->nnodes);
	/* we move the off initialization from each device thread here, i.e. we serialized this */
	for (i=0; i<top->nnodes; i++) {
		omp_offloading_t * off = &info->offloadings[i];
		omp_device_t * dev = &omp_devices[top->idmap[i]];
		off->devseqid = i;
		off->dev = dev;
		off->off_info = info;
		off->num_maps = 0;
		off->stage = OMP_OFFLOADING_INIT;
	}

	pthread_barrier_init(&info->barrier, NULL, top->nnodes+1);
	pthread_barrier_init(&info->inter_dev_barrier, NULL, top->nnodes);
	return info;
}


void omp_offloading_append_profile_per_iteration(omp_offloading_info_t *info, long num_fp_operations, long num_loads,
												 long num_stores) {
	info->per_iteration_profile.num_fp_operations = num_fp_operations;
	info->per_iteration_profile.num_load = num_loads;
	info->per_iteration_profile.num_store = num_stores;
}

void omp_offloading_fini_info(omp_offloading_info_t * info) {
	pthread_barrier_destroy(&info->barrier);
	pthread_barrier_destroy(&info->inter_dev_barrier);
#if defined (OMP_BREAKDOWN_TIMING)
	int i;
	for (i=0; i<info->top->nnodes; i++) {
		omp_offloading_t * off = &info->offloadings[i];
		free(off->events);
	}
#endif
	free(info);
}

// Initialize data_map_info
void omp_data_map_init_info(const char *symbol, omp_data_map_info_t *info, omp_offloading_info_t *off_info,
                            void *source_ptr, int num_dims, int sizeof_element, omp_data_map_direction_t map_direction,
                            omp_data_map_type_t map_type) {
	if (num_dims > 3) {
		fprintf(stderr, "%d dimension array is not supported in this implementation!\n", num_dims);
		exit(1);
	}
	info->symbol = symbol;
	info->off_info = off_info;
	info->source_ptr = source_ptr;
	info->num_dims = num_dims;
	//memset(info->maps, 0, sizeof(omp_data_map_t) * off_info->top->nnodes);
	info->map_direction = map_direction;
	info->map_type = map_type;
	info->num_halo_dims = 0;
	info->sizeof_element = sizeof_element;
	info->remap_needed = 0;
	int i;
	for (i=0; i<num_dims; i++) {
		info->dist_info[i].target_type = OMP_DIST_TARGET_DATA_MAP;
		info->dist_info[i].target = info;
		info->dist_info[i].target_dim = i; /* we donot know which dim this dist is applied to, also dist is an array, making default here */
	}
}

void omp_data_map_info_set_dims_1d(omp_data_map_info_t * info, long dim0) {
	info->dims[0] = dim0;
}

void omp_data_map_info_set_dims_2d(omp_data_map_info_t * info, long dim0, long dim1) {
	info->dims[0] = dim0;
	info->dims[1] = dim1;
}
void omp_data_map_info_set_dims_3d(omp_data_map_info_t * info, long dim0, long dim1, long dim2) {
	info->dims[0] = dim0;
	info->dims[1] = dim1;
	info->dims[2] = dim2;
}

void omp_init_dist_info(omp_dist_info_t *dist_info, omp_dist_policy_t dist_policy, long start, long length,
						int chunk_size, int topdim) {
	dist_info->start = start;
	dist_info->length = length;
	dist_info->end = start + length;
	dist_info->policy = dist_policy;
	dist_info->dim_index = topdim;
	dist_info->redist_needed = 0;
	dist_info->chunk_size = chunk_size;
}

void omp_data_map_dist_init_info(omp_data_map_info_t *map_info, int dim, omp_dist_policy_t dist_policy, long start,
								 long length, int chunk_size, int topdim) {
	omp_dist_info_t * dist_info = &map_info->dist_info[dim];
	omp_init_dist_info(dist_info, dist_policy, start, length, chunk_size, topdim);
}

void omp_loop_dist_init_info(omp_offloading_info_t *off_info, int level, omp_dist_policy_t dist_policy, long start,
							 long length, int chunk_size, int topdim) {
	omp_dist_info_t * dist_info = &off_info->loop_dist_info[level];
	omp_init_dist_info(dist_info, dist_policy, start, length, chunk_size, topdim);
	if (dist_policy == OMP_DIST_POLICY_SCHED_DYNAMIC ||
		dist_policy == OMP_DIST_POLICY_SCHED_GUIDED ||
		dist_policy == OMP_DIST_POLICY_SCHED_PROFILE_AUTO ) {
		dist_info->redist_needed = 1; /* multiple distributions are needed after the initial one */
	}
}

void omp_loop_dist_static_ratio(omp_offloading_info_t *off_info, int level, long start, long length, float * ratios, int topdim) {
	omp_dist_info_t * dist_info = &off_info->loop_dist_info[level];
	omp_init_dist_info(dist_info, OMP_DIST_POLICY_SCHED_STATIC_RATIO, start, length, 0, topdim);
	int i;
	for (i=0; i<off_info->top->dims[topdim]; i++) {
		/* TODO: */
	}
}

/* return whether realignment is needed or not for this align in the future */
static int omp_set_align_dist_policy(omp_dist_info_t * dist_info, omp_dist_target_type_t alignee_type, void * alignee, int alignee_dim, long start) {
	omp_dist_info_t *alignee_dist_info = NULL;
	if (alignee_type == OMP_DIST_TARGET_DATA_MAP) {
		omp_data_map_info_t *alignee_map_info = (omp_data_map_info_t *) alignee;
		alignee_dist_info = &alignee_map_info->dist_info[alignee_dim];
	} else { /* OMP_DIST_TARGET_LOOP_ITERATION */
		alignee_dist_info = &((omp_offloading_info_t*)alignee)->loop_dist_info[alignee_dim];
	}

	start = (start == OMP_ALIGNEE_START ? alignee_dist_info->start : start);
	if (alignee_dist_info->policy == OMP_DIST_POLICY_ALIGN) {/* if there is a chain of alignment, make it point to the root */
		return omp_set_align_dist_policy(dist_info, alignee_dist_info->alignee_type, alignee_dist_info->alignee.data_map_info, alignee_dist_info->dim_index, start);
	} else {
		omp_init_dist_info(dist_info, OMP_DIST_POLICY_ALIGN, start, 0, 0, alignee_dim);
		dist_info->alignee_type = alignee_type;
		dist_info->alignee.data_map_info = (omp_data_map_info_t *) alignee;
		dist_info->redist_needed = alignee_dist_info->redist_needed;
	}

	return dist_info->redist_needed;
}

/* to align one data map with another data map, if dim>=0, align a specific dim, if dim<0, align all the dims */
void omp_data_map_dist_align_with_data_map(omp_data_map_info_t *map_info, int dim, long start,
                                           omp_data_map_info_t *alignee, int alignee_dim) {
	int redist_needed = 0;
	if (dim >= 0 && alignee_dim >=0) {
		redist_needed = omp_set_align_dist_policy(&map_info->dist_info[dim], OMP_DIST_TARGET_DATA_MAP, alignee, alignee_dim, start);
	} else if (dim == OMP_ALL_DIMENSIONS && alignee_dim == OMP_ALL_DIMENSIONS){ /* for all the dimensions that will be aligned */
		int i;
		for (i=0; i<map_info->num_dims;i++) {
			if (omp_set_align_dist_policy(&map_info->dist_info[i], OMP_DIST_TARGET_DATA_MAP, alignee, i, start))
				redist_needed = 1;
		}
	} else if (dim == OMP_ALL_DIMENSIONS && alignee_dim >=0) {
		int i;
		for (i=0; i<map_info->num_dims;i++) {
			if (omp_set_align_dist_policy(&map_info->dist_info[i], OMP_DIST_TARGET_DATA_MAP, alignee, alignee_dim, start))
				redist_needed = 1;
		}
	} else {
		abort();
	}
	if (redist_needed) map_info->remap_needed = 1;
}

/* to align one data map with another data map, if dim>=0, align a specific dim, if dim<0, align all the dims */
void omp_data_map_dist_align_with_data_map_with_halo(omp_data_map_info_t *map_info, int dim, long start,
                                                     omp_data_map_info_t *alignee, int alignee_dim) {
	int redist_needed = 0;
	if (dim >= 0 && alignee_dim >=0) {
		redist_needed = omp_set_align_dist_policy(&map_info->dist_info[dim], OMP_DIST_TARGET_DATA_MAP, alignee, alignee_dim, start);
		if (alignee->num_halo_dims) {
			omp_data_map_halo_region_info_t * halo_info = &alignee->halo_info[alignee_dim];
			omp_map_add_halo_region(map_info, dim, halo_info->left, halo_info->right, halo_info->edging);
		}
	} else if (dim == OMP_ALL_DIMENSIONS && alignee_dim == OMP_ALL_DIMENSIONS){ /* for all the dimensions that will be aligned */
		int i;
		for (i=0; i<map_info->num_dims;i++) {
			if (omp_set_align_dist_policy(&map_info->dist_info[i], OMP_DIST_TARGET_DATA_MAP, alignee, i, start))
				redist_needed = 1;
			if (alignee->num_halo_dims) {
				omp_data_map_halo_region_info_t * halo_info = &alignee->halo_info[i];
				omp_map_add_halo_region(map_info, i, halo_info->left, halo_info->right, halo_info->edging);
			}
		}
	} else if (dim == OMP_ALL_DIMENSIONS && alignee_dim >=0) {
		int i;
		for (i=0; i<map_info->num_dims;i++) {
			if (omp_set_align_dist_policy(&map_info->dist_info[i], OMP_DIST_TARGET_DATA_MAP, alignee, alignee_dim, start))
				redist_needed = 1;
			if (alignee->num_halo_dims) {
				omp_data_map_halo_region_info_t * halo_info = &alignee->halo_info[alignee_dim];
				omp_map_add_halo_region(map_info, i, halo_info->left, halo_info->right, halo_info->edging);
			}
		}
	} else {
		abort();
	}
	if (redist_needed) map_info->remap_needed = 1;
}

void omp_data_map_dist_align_with_loop(omp_data_map_info_t *map_info, int dim, long start,
                                       omp_offloading_info_t *alignee, int alignee_level) {
	int redist_needed = 0;
	if (dim >= 0 && alignee_level >=0) {
		redist_needed = omp_set_align_dist_policy(&map_info->dist_info[dim], OMP_DIST_TARGET_LOOP_ITERATION, alignee, alignee_level, start);
	} else if (dim == OMP_ALL_DIMENSIONS && alignee_level == OMP_ALL_DIMENSIONS){ /* for all the dimensions that will be aligned */
		int i;
		for (i=0; i<map_info->num_dims;i++) {
			if (omp_set_align_dist_policy(&map_info->dist_info[i], OMP_DIST_TARGET_LOOP_ITERATION, alignee, i, start))
				redist_needed = 1;
		}
	} else if (dim == OMP_ALL_DIMENSIONS && alignee_level >=0) {
		int i;
		for (i=0; i<map_info->num_dims;i++) {
			if (omp_set_align_dist_policy(&map_info->dist_info[i], OMP_DIST_TARGET_LOOP_ITERATION, alignee, alignee_level, start))
				redist_needed = 1;
		}
	} else {
		abort();
	}
	if (redist_needed) map_info->remap_needed = 1;
}

void omp_loop_dist_align_with_data_map(omp_offloading_info_t *loop_off_info, int level, long start,
                                       omp_data_map_info_t *alignee, int alignee_dim) {
	if (level >= 0 && alignee_dim >=0) {
		omp_set_align_dist_policy(&loop_off_info->loop_dist_info[level], OMP_DIST_TARGET_DATA_MAP, alignee, alignee_dim, start);
	} else if (level == OMP_ALL_DIMENSIONS && alignee_dim == OMP_ALL_DIMENSIONS){ /* for all the dimensions that will be aligned */
		int i;
		for (i=0; i<loop_off_info->loop_depth;i++) {
			omp_set_align_dist_policy(&loop_off_info->loop_dist_info[i], OMP_DIST_TARGET_DATA_MAP, alignee, i, start);
		}
	} else if (level == OMP_ALL_DIMENSIONS && alignee_dim >=0) {
		int i;
		for (i=0; i<loop_off_info->loop_depth;i++) {
			omp_set_align_dist_policy(&loop_off_info->loop_dist_info[level], OMP_DIST_TARGET_DATA_MAP, alignee, alignee_dim, start);
		}
	} else {
		abort();
	}
	loop_off_info->loop_redist_needed = alignee->remap_needed;
}

void omp_loop_dist_align_with_loop(omp_offloading_info_t *loop_off_info, int level, long start,
                                   omp_offloading_info_t *alignee, int alignee_level) {
	if (level >= 0 && alignee_level >=0) {
		omp_set_align_dist_policy(&loop_off_info->loop_dist_info[level], OMP_DIST_TARGET_LOOP_ITERATION, alignee, alignee_level, start);
	} else if (level == OMP_ALL_DIMENSIONS && alignee_level == OMP_ALL_DIMENSIONS){ /* for all the dimensions that will be aligned */
		int i;
		for (i=0; i<loop_off_info->loop_depth;i++) {
			omp_set_align_dist_policy(&loop_off_info->loop_dist_info[i], OMP_DIST_TARGET_LOOP_ITERATION, alignee, i, start);
		}
	} else if (level == OMP_ALL_DIMENSIONS && alignee_level >=0) {
		int i;
		for (i=0; i<loop_off_info->loop_depth;i++) {
			omp_set_align_dist_policy(&loop_off_info->loop_dist_info[level], OMP_DIST_TARGET_LOOP_ITERATION, alignee, alignee_level, start);
		}
	} else {
		abort();
	}
	loop_off_info->loop_redist_needed = alignee->loop_redist_needed;
}

omp_data_map_t *omp_map_offcache_iterator(omp_offloading_t *off, int index, int * inherited) {
	if (index >= off->num_maps) return NULL;
	*inherited = off->map_cache[index].inherited;
	return off->map_cache[index].map;
}

static omp_data_map_t *omp_map_get_map_from_offcache(omp_offloading_t *off, void *host_ptr) {
	int i;
	for (i=0; i<off->num_maps; i++) {
		omp_data_map_t * map = off->map_cache[i].map;
		if (map->info->source_ptr == host_ptr) return map;
	}

	return NULL;
}

void omp_map_append_map_to_offcache(omp_offloading_t *off, omp_data_map_t *map, int inherited) {
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
		omp_data_map_t * map = omp_map_get_map_from_offcache(ancestor_off, host_ptr);
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
	omp_data_map_t * map = omp_map_get_map_from_offcache(off, host_ptr);
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
			omp_map_append_map_to_offcache(off, map, 0);
			return map;
		}
	} else { /* thorough search for all the mapped variables */
		int i; omp_data_map_info_t * dm_info;
		for (i=0; i<off_info->num_mapped_vars; i++) {
			dm_info = &off_info->data_map_info[i];
			if (dm_info->source_ptr == host_ptr) { /* we found */
				map = &dm_info->maps[devseqid];
				//printf("find a match: %X\n", host_ptr);
				omp_map_append_map_to_offcache(off, map, 0);
				//omp_print_data_map(map);
				return map;
			}
		}
	}

	/* STEP 3: seach the offloading stack if this inherits data map from previous data offloading */
//	printf("omp_map_get_map: off: %X, off_info: %X, host_ptr: %X\n", off, off_info, host_ptr);
	map = omp_map_get_map_inheritance (off->dev, host_ptr);
	if (map != NULL) omp_map_append_map_to_offcache(off, map, 1);

	return map;
}

void omp_map_add_halo_region(omp_data_map_info_t *info, int dim, int left, int right, omp_dist_halo_edging_type_t edging) {
	info->num_halo_dims++;
	info->halo_info[dim].left = left;
	info->halo_info[dim].right = right;
	info->halo_info[dim].edging = edging;
	info->halo_info[dim].topdim = info->dist_info[dim].dim_index;
}

/**
 * after initialization
 */
void omp_data_map_init_map(omp_data_map_t *map, omp_data_map_info_t *info, omp_device_t *dev) {
//	if (map->access_level >= OMP_DATA_MAP_ACCESS_LEVEL_INIT) return;
	map->info = info;
	map->dev = dev; /* mainly use as cache so we save one pointer deference */
	map->mem_noncontiguous = 0;
	map->map_type = info->map_type;

	if (map->map_type == OMP_DATA_MAP_AUTO) {
		if (omp_device_mem_discrete(map->dev->mem_type)) {
			map->map_type = OMP_DATA_MAP_COPY;
			//printf("COPY data map: %X, dev: %d\n", map, dev->id);
		} else { /* we can make it shared and we will do it */
			map->map_type = OMP_DATA_MAP_SHARED;
			//printf("SHARED data map: %X, dev: %d\n", map, dev->id);
		}
	} else if (map->map_type == OMP_DATA_MAP_SHARED && omp_device_mem_discrete(dev->mem_type)) {
		fprintf(stderr, "direct sharing data between host and the dev: %d is not possible with discrete mem space, we use COPY approach now\n", dev->id);
		map->map_type = OMP_DATA_MAP_COPY;
	}
	int i;
	for (i=0; i<info->num_dims; i++) {
		map->map_dist[i].counter = 0;
		map->map_dist[i].next = NULL;
	}
	map->access_level = OMP_DATA_MAP_ACCESS_LEVEL_INIT;
}

/* forward declaration to suppress compiler warning */
void omp_topology_get_neighbors(omp_grid_topology_t * top, int seqid, int topdim, int cyclic, int* left, int* right);


void omp_dist_block(long start, long full_length, long position, int dim, long * offstart, long *length) {
	/* partition the array region into subregion and save it to the map */
	long remaint = full_length % dim;
	long esize = full_length / dim;
	//	printf("n: %d, seqid: %d, map_dist: topdimsize: %d, remaint: %d, esize: %d\n", n, seqid, topdimsize, remaint, esize);
	long len, offset;
	if (position < remaint) { /* each of the first remaint dev has one more element */
		len = esize + 1;
		offset = (esize + 1) * position;
	} else {
		len = esize;
		offset = esize * position + remaint;
	}
	*offstart = start + offset;
	*length = len;
}

static __inline__ int __homp_cas(volatile int *ptr, int ag, int x) {
	return __sync_bool_compare_and_swap(ptr, ag, x);
}

#if !defined(LINEAR_MODEL_1) && !defined(LINEAR_MODEL_2)
#define LINEAR_MODEL_2 1
#endif

#if defined(LINEAR_MODEL_1) && defined(LINEAR_MODEL_2)
#undef LINEAR_MODEL_1
#define LINEAR_MODEL_2 1
#endif

#if defined (LINEAR_MODEL_1)
#warning "LINEAR_MODEL_1 is used"
#endif

#if defined (LINEAR_MODEL_2)
#warning "LINEAR_MODEL_2 is used"
#endif
/**
 * The general dist algorithm that applies to both data distribution and iteration distribution
 */
void omp_dist(omp_dist_info_t *dist_info, omp_dist_t *dist, omp_grid_topology_t *top, int *coords, int seqid, int dim) {
	long full_length = dist_info->end - dist_info->start;
	long offset = 0;
	long length = 0;
	int i;
	omp_device_t * dev = &omp_devices[top->idmap[seqid]];

	int dim_index = dist_info->dim_index;
	//	printf("dim_index: %d\n", dim_index);
	if (dist_info->policy == OMP_DIST_POLICY_BLOCK) { /* even distributions */
		int topdimcoord = coords[dim_index]; /* dim_indx is top dim the dist is applied onto */
		int topdimsize = top->dims[dim_index];
		/* partition the array region into subregion and save it to the map */
		long remaint = full_length % topdimsize;
		long esize = full_length / topdimsize;
		//	printf("n: %d, seqid: %d, map_dist: topdimsize: %d, remaint: %d, esize: %d\n", n, seqid, topdimsize, remaint, esize);
		if (topdimcoord < remaint) { /* each of the first remaint dev has one more element */
			length = esize + 1;
			offset = (esize + 1) * topdimcoord;
		} else {
			length = esize;
			offset = esize * topdimcoord + remaint;
		}

		dist->offset = dist_info->start + offset;
		dist->length = length;
	} else if (dist_info->policy == OMP_DIST_POLICY_FULL) { /* full rang dist_info */
		dist->length = full_length;
		dist->offset = dist_info->start;
	} else if (dist_info->policy == OMP_DIST_POLICY_ALIGN) {
		omp_dist_info_t *alignee_dist_info = NULL;
		omp_dist_t *alignee_dist = NULL;
		if (dist_info->alignee_type == OMP_DIST_TARGET_DATA_MAP) {
			omp_data_map_info_t * alignee_data_map_info = dist_info->alignee.data_map_info;
			int anseqid = omp_grid_topology_get_seqid(alignee_data_map_info->off_info->top, dev->id);

			/* get the actual alignee map */
			omp_data_map_t * alignee_data_map = &alignee_data_map_info->maps[anseqid];
			/* do the alignee distribution first */
			if (alignee_data_map->access_level < OMP_DATA_MAP_ACCESS_LEVEL_INIT)
				omp_data_map_init_map(alignee_data_map, alignee_data_map_info, dev);
			if (alignee_data_map->access_level < OMP_DATA_MAP_ACCESS_LEVEL_DIST) {
				omp_data_map_dist(alignee_data_map, anseqid);;
			}
			/* copy the alignee length and offset */
			alignee_dist_info = &alignee_data_map_info->dist_info[dim_index];
			alignee_dist = &alignee_data_map->map_dist[dim_index];
		} else if (dist_info->alignee_type == OMP_DIST_TARGET_LOOP_ITERATION) {
			omp_offloading_info_t *alignee_off_info = dist_info->alignee.loop_iteration;
			//alignee_dist_info = &alignee_loop->loop_dist_info[dim_index];
			int anseqid = omp_grid_topology_get_seqid(alignee_off_info->top, dev->id);
			omp_offloading_t *alignee_off = &alignee_off_info->offloadings[anseqid];
			if (!alignee_off->loop_dist_done) { /* TODO not yet executed, so loop iteration has not yet distributed */
				//printf("dist the alignee loop iteration\n");
				omp_loop_iteration_dist(alignee_off);
			}
			alignee_dist_info = &alignee_off_info->loop_dist_info[dim_index];
			alignee_dist = &alignee_off->loop_dist[dim_index];
		} else {
			alignee_dist_info = NULL;
			alignee_dist = NULL;
		}
		dist->length = alignee_dist->length;
		dist->offset = alignee_dist->offset - alignee_dist_info->start + dist_info->start;
//		printf("aligned dist on dev %d: offset: %d, length: %d\n", seqid, alignee_dist->offset, alignee_dist->length);
	} else if (dist_info->policy == OMP_DIST_POLICY_SCHED_AUTO) {
		/* in LINEAR_MODEL_1, only computation is considered */
		if (dist_info->target_type == OMP_DIST_TARGET_LOOP_ITERATION ) {
		} else {
			abort();
			/* error, so far AUTO is only applied to loop iteration */
		}
		omp_offloading_info_t * off_info = (omp_offloading_info_t*)dist_info->target;
		omp_offloading_t * off = &off_info->offloadings[seqid];
		omp_event_t *events = off->events;
#if defined (OMP_BREAKDOWN_TIMING)
		/* For aligned AUTO, the alignee may not yet initialized so here we have limited access to
		 * some of objects of the alignees, events are one of them.
		 * So here we only charge this as modeling cost if off->events are all initialized in the offloading.
		 * Otherwise, this modeling cost is charged to the aligner as the cost of distribution, not modeling
		 */
		if (events != NULL)
			omp_event_record_start(&events[runtime_dist_modeling_index]);
#endif
#ifdef LINEAR_MODEL_1
		/* compute the total capability */
		double total_flops = 0.0;
		for (i =0; i <off_info->top->nnodes; i++) {
			double flops = off_info->offloadings[i].dev->total_real_flopss;
			total_flops += flops;
		}

		for (i =0; i <off_info->top->nnodes; i++) {
			double flops = off_info->offloadings[i].dev->total_real_flopss;
			length = (flops/total_flops) * full_length + 0.5; /* +0.5 is for rounding, so 3.4->4, and 3.5->4 */
			if (i == off_info->top->nnodes-1 && offset + length != full_length) { /* fix rounding error */
				length = full_length - offset;
			}
			if (off->devseqid == i) {
				dist->length = length;
				dist->offset = offset;
				break;
			}
			//printf("LINEAR_MODEL_1: Dev %d (%f GFlops/s): offset: %d, length: %d of total length: %d from start: %d\n", i, flops, offset, length, dist_info->length, dist_info->start);
			offset += length;
		}
#endif
/* in LINEAR_MODEL_2, computation and data movement cost are considered */
#ifdef LINEAR_MODEL_2
		/* this is more restricted, since now all aligned data map and non-aligned data map in this offloading
	     * has to be took into account for the auto policy. The data movement cost for non-aligned data map will be part
	     * of the constant of a linear equation, and data-movement cost for aligned
	     * data-map will be calculated as part of the coefficient of the model.
	     *
	     * For data movement, the model will be the classical a+b model, i.e. T = a + n*B, where a is latency, n is number of bytes to be transfered
	     * and B is bandwidth.
	     */

		/* compute constant A and B for T = n*A+B, where n is number of loop iterations. The full model is as follows
		 * T = n * (Sum(S_Ai))/Bandwidth + Sum(S_NAi)/Bandwidth + #arrays * Latency + n*FLOPs/Fp, where S_Ai is the sizes of an array that has
		 * alignment distribtuion with loop auto policy, S_NAi, is the sizes of an array that does not have alignment with loop. # array is equal
		 * to the number of data movment. FLOPs is FLOPs per loop iterations, here we assume same number of iterations of all loop iterations. Fp is
		 * FLOPS/s of the device
		 *
		 * So A = (Sum(S_Ai))/Bandwidth + FLOPs/Fp, B = Sum(S_NAi)/Bandwidth + #arrays * Latency
		 * When we have T = n*A+B, we have n = T/A - B/A for each device, thus we form a linear systems for multiple devices.
		 */

		/* this serves as fast cache for later alignment */
		offset = 0;
		struct align_map {
			omp_data_map_t * map;
			omp_dist_t * align_dist;
		} align_maps[off_info->num_mapped_vars];
		int num_aligned_maps = 0;
		double A = 0.0;
		double B = 0.0;
		int num_transfer = 0;
		for (i=0; i<off_info->num_mapped_vars; i++) {
			omp_data_map_info_t * map_info = &off_info->data_map_info[i];
			if (map_info->map_type == OMP_DATA_MAP_FROM || map_info->map_type == OMP_DATA_MAP_TO) {
				num_transfer++;
			} else if (map_info->map_type == OMP_DATA_MAP_TOFROM) {
				num_transfer += 2;
			} else continue;

			omp_data_map_t * map = &map_info->maps[seqid];
			long map_size = map_info->sizeof_element;
			align_maps[i].align_dist = NULL;
			int num_aligned_dims = 0;
			int j;
			for (j = 0; j < map_info->num_dims; j++) { /* process each dimension */
				omp_dist_info_t *andist_info = &map_info->dist_info[j];
				omp_dist_t * map_dist = &map->map_dist[j];
				if (andist_info->policy != OMP_DIST_POLICY_ALIGN) { /* all non-auto distribution dimension, we will just do it */
					omp_dist(andist_info, map_dist, top, coords, seqid, j);
					map_size *= map_dist->length;
				} else {/* ALIGN policy, and so far, we only handle one-dimension ALIGN with loop*/
					align_maps[i].map = map;
					align_maps[i].align_dist = map_dist;
					num_aligned_maps++;
					num_aligned_dims++;
					if (num_aligned_dims == 2) {
					//	printf("we only handle one-dimension alignment of array with loops iterations\n");
					}
				}
			}
			map->map_size = map_size;
			if (map_info->map_type == OMP_DATA_MAP_TOFROM) {
				map_size = map_size*2;
			}
			if (align_maps[i].align_dist != NULL) { /* we have an alignment */
				A += map_size;
			} else {
				B += map_size;
			}
		}

		A = A/(dev->bandwidth*10.0e6); /* bandwidth in MB/s */
		B = B/(dev->bandwidth*10.0e6);
		A += off_info->per_iteration_profile.num_fp_operations/(dev->total_real_flopss * 10.0e9); /* FLOPs is in GFLOPs/s */
		B += num_transfer * (dev->latency * 10.0e-6); /* latency in us */
		/* here T = n*A+B --> n = T/A - B/A. We have then Ar = 1/A, and Br = -B/A*/

		double Ar = 1.0/A;
		double Br = 0.0 - B/A;
		/* broadcast this to other device */
		off->Ar = Ar;
		off->Br = Br;
		/* sync so to make sure all received this info */
		pthread_barrier_wait(&off_info->inter_dev_barrier);
		/* solve the linear system */
		double allArs = 0.0;
		double allBrs = 0.0;
		for (i =0; i <off_info->top->nnodes; i++) {
			omp_offloading_t * anoff = &off_info->offloadings[i];
			allArs += anoff->Ar;
			allBrs += anoff->Br;
		}
		//printf("allArs: %f, allBrs: %f\n", allArs, allBrs);
		double T0 = (full_length - allBrs)/allArs; /* the predicted execution time by all the devices of the loop */
		/* now compute the offset and length for AUTO dist policy */
		for (i =0; i <off_info->top->nnodes; i++) {
			omp_offloading_t * anoff = &off_info->offloadings[i];
			length = (long)(T0*anoff->Ar + anoff->Br + 0.5);
			if (length <= 0) length = 0;
			if (length >= full_length) length = full_length;
			if (i == off_info->top->nnodes-1 && offset + length != full_length) { /* fix rounding error */
				length = full_length - offset;
			}
			if (seqid == i) {
				dist->offset = offset;
				dist->length = length;
				break;
			}
//			printf("LINEAR_MODEL_2: Dev %d: offset: %d, length: %d of total length: %d from start: %d, predicted exe time: %f\n",
//				   i, offset, length, dist_info->length, dist_info->start, T0);
			offset += length;
		}

		/* do the alignment map for arrays */
		for (i=0; i<off_info->num_mapped_vars; i++) {
			omp_dist_t * align_dist = align_maps[i].align_dist;
			if (align_dist != NULL) {
				align_dist->length = dist->length;
				align_dist->offset = dist->offset;
			}
		}
#endif
#if defined (OMP_BREAKDOWN_TIMING)
		if (events != NULL)
			omp_event_record_stop(&events[runtime_dist_modeling_index]);
#endif
	} else if (dist_info->policy == OMP_DIST_POLICY_SCHED_STATIC_RATIO) { /* simply distribute according to a user specified ratio */
		omp_offloading_info_t * off_info = (omp_offloading_info_t*)dist_info->target;
		for (i =0; i <off_info->top->nnodes; i++) {
			omp_offloading_t * anoff = &off_info->offloadings[i];
			length = full_length * anoff->loop_dist[dim].ratio;
			if (length <= 0) length = 0;
			if (length >= full_length) length = full_length;
			if (i == off_info->top->nnodes-1 && offset + length != full_length) { /* fix rounding error */
				length = full_length - offset;
			}
			if (seqid == i) {
				dist->offset = offset;
				dist->length = length;
				break;
			}
//			printf("STATIC_RATIO: Dev %d: offset: %d, length: %d of total length: %d from start: %d, predicted exe time: %f\n",
//				   i, offset, length, dist_info->length, dist_info->start, T0);
			offset += length;
		}
	} else if (dist_info->policy == OMP_DIST_POLICY_SCHED_DYNAMIC) { /* only for loop */
		/* get next chunk */
		length = dist_info->chunk_size;
		do {
			offset = dist_info->start;
			int new_offset = offset + length;
			if (new_offset > dist_info->end) {
				length = dist_info->end - offset;
				if (length <= 0) {
					length = 0;
					break;
				}
				new_offset = offset + length;
			}
			if (length > 0) {
				if (__homp_cas(&dist_info->start, offset, new_offset)) break;
			}
		} while (1);
		dist->offset = offset;
		dist->length = length;
	} else if (dist_info->policy == OMP_DIST_POLICY_SCHED_GUIDED) { /* only for loop */
		length = dist->length / 2; /* cut in half each time to pick a chunk */
		if (length < 10) length = 10; /* minimum 10 iterations per chunk */
		do {
			offset = dist_info->start;
			int new_offset = offset + length;
			if (new_offset > dist_info->end) {
				length = dist_info->end - offset;
				if (length <= 0) {
					length = 0;
					break;
				}
				new_offset = offset + length;
			}
			if (length > 0) {
				if (__homp_cas(&dist_info->start, offset, new_offset)) break;
			}
		} while (1);
		dist->offset = offset;
		dist->length = length;
	} else if (dist_info->policy == OMP_DIST_POLICY_SCHED_PROFILE_AUTO) { /* only for loop */
#ifndef OMP_BREAKDOWN_TIMING
		printf("OMP_BREAKDOWN_TIMING must be set to use OMP_DIST_POLICY_SCHED_PROFILE_AUTO policy\n");
#endif
        /* two dists are needed for this policy and we use counter for that */
		if (dist->counter == 0) { /* SCHEDULE_STATIC policy */
			length = dist_info->chunk_size;
			do {
				offset = dist_info->start;
				int new_offset = offset + length;
				if (new_offset > dist_info->end) {
					length = dist_info->end - offset;
					if (length <= 0) {
						length = 0;
						break;
					}
					new_offset = offset + length;
				}
				if (length > 0) {
					if (__homp_cas(&dist_info->start, offset, new_offset)) break;
				}
			} while (1);
			dist->offset = offset;
			dist->length = length;
//			printf("PROFILE_AUTO, PROFILE: Dev %d: offset: %d, length: %d of total length: %d from start: %d, predicted exe time: %f\n",
//				   i, offset, length, dist_info->length, dist_info->start, T0);
		} else { /* similar to ratio policy */
			omp_offloading_info_t * off_info = (omp_offloading_info_t*)dist_info->target;
			full_length = dist_info->end - dist_info->start;
			offset = 0;
			for (i =0; i <off_info->top->nnodes; i++) {
				omp_offloading_t * anoff = &off_info->offloadings[i];
				omp_event_t * ev = &anoff->events[total_event_accumulated_index];
				int evcount = ev->count;
				if (evcount != 1) {
					printf("Assertation failure for total_event_accumulated_index events, it should be called only once, but it is %d\n", evcount);
				}
				double Ti = omp_event_get_elapsed(ev);
				double ratio = 0.0;
				int j;
				for (j=0; j<off_info->top->nnodes; j++) {
					ratio += Ti/omp_event_get_elapsed(&off_info->offloadings[j].events[total_event_accumulated_index]);
				}
				length = full_length * ratio;

				if (length <= 0) length = 0;
				if (length >= full_length) length = full_length;
				if (i == off_info->top->nnodes-1 && offset + length != full_length) { /* fix rounding error */
					length = full_length - offset;
				}
				if (seqid == i) {
					dist->offset = offset;
					dist->length = length;
					break;
				}

				offset += length;
//			printf("PROFILE_AUTO, AUTO: Dev %d: offset: %d, length: %d of total length: %d from start: %d, predicted exe time: %f\n",
//				   i, offset, length, dist_info->length, dist_info->start, T0);
			}
		}
	} else {
		fprintf(stderr, "other dist_info type %d is not yet supported\n",
				dist_info->policy);
		abort();
		exit(1);
	}
	dist->counter++;
}

/* return the total amount of distribution */
long omp_loop_iteration_dist(omp_offloading_t *off) {
	//if (off->loop_dist_done) return;
	omp_offloading_info_t *off_info = off->off_info;
	omp_grid_topology_t * top = off_info->top;

	int coords[top->ndims];
	omp_topology_get_coords(top, off->devseqid, top->ndims, coords);
	int i;

	//printf("omp_dist_call: %d\n", __LINE__);
	long total = 1;
	for (i = 0; i < off_info->loop_depth; i++) { /* process each dimension */
		omp_dist_info_t *dist_info = &off_info->loop_dist_info[i];
		off->loop_dist[i].info = dist_info;

		omp_dist(dist_info, &off->loop_dist[i], top, coords, off->devseqid, i);
		total *= off->loop_dist[i].length;
	}

	off->loop_dist_done = 1;
	return total;
}

/**
 * Apply map to device seqid, seqid is the sequence id of the device in the grid topology
 *
 * do the distribution of array onto the grid topology of devices
 *
 * return the total amount of dist
 */
long omp_data_map_dist(omp_data_map_t *map, int seqid) {
//	if (map->access_level >= OMP_DATA_MAP_ACCESS_LEVEL_DIST) return; /* a simple way of mutual exclusion of multiple entry by one thread*/
	omp_data_map_info_t *map_info = map->info;
	omp_offloading_info_t * off_info = map_info->off_info;
	omp_grid_topology_t * top = off_info->top;

	int coords[top->ndims];
	omp_topology_get_coords(top, seqid, top->ndims, coords);
	int i;

	//printf("omp_dist_call: %d\n", __LINE__);
	int sizeof_element = map_info->sizeof_element;
	long map_size = sizeof_element;
	long map_wextra_size = sizeof_element;

	long offset_from0 = 0;
	long mt_from0 = 1;
	long offset_wextra_from0 = 0;
	long mt_wextra_from0 = 1;

	long total = 1;
	for (i = map_info->num_dims-1; i>=0; i--) { /* process each dimension */
		omp_dist_info_t *dist_info = &map_info->dist_info[i];

#if 0
		/* TODO: we may not need this optimization, which makes code harder to read
		 * The idea of this optimization is to only re-dist those dimensions that are needed.
		 * Since we most time handle max 4 dimension, this is not a big deal of overehead.
		 */
		/* if this is the first time, we need to dist */
		if (map->map_dist[i].access_level < OMP_DATA_MAP_ACCESS_LEVEL_DIST) {
			map->map_dist[i].info = dist_info;
			omp_dist(dist_info, &map->map_dist[i], top, coords, seqid);
		} else if (dist_info->redist_needed) /* if this is align and we are re-enter,  we need to dist regardless of whatever */
#endif
		omp_dist(dist_info, &map->map_dist[i], top, coords, seqid, i);

		long length = map->map_dist[i].length;
		long offset = map->map_dist[i].offset;

		map_size *= length;
		offset_from0 += mt_from0 * offset;
		mt_from0 *= map_info->dims[i];
		if (i>0 && (length != map_info->dims[i])) {
			/* check the dimension from 1 to the highest, if any one is not the full range of the dimension in the original array,
			 * we have non-contiguous memory space and we need to marshall data
			 */
			map->mem_noncontiguous = 1;
		}

		if (map_info->num_halo_dims) {
			omp_data_map_halo_region_info_t *halo = &map_info->halo_info[i];

			/* handle halo region  */
			/** here we also need to deal with boundary, the first one that has no left halo and the last one that has no right halo for non-cyclic case */
			if (halo->left != 0 || halo->right != 0) { /* we have halo in this dimension */
				omp_data_map_halo_region_mem_t *halo_mem = &map->halo_mem[i];
				int *left = &halo_mem->left_dev_seqid;
				int *right = &halo_mem->right_dev_seqid;
				omp_topology_get_neighbors(top, seqid, halo->topdim, halo->edging == OMP_DIST_HALO_EDGING_PERIODIC, left, right);
//				if (*left >= 0) {
					length += halo->left;
					offset = offset - halo->left;
//				}
//				if (*right >= 0) {
					length += halo->right;
//				}
			}
		}
		map_wextra_size *= length;
		offset_wextra_from0 += mt_wextra_from0 * offset;
		mt_wextra_from0 *= map_info->dims[i];
		total *= length;
	}
	map->map_size = map_size;
	map->map_wextra_size = map_wextra_size;
	map->map_source_ptr = map_info->source_ptr + sizeof_element * offset_from0;
	map->map_source_wextra_ptr = map_info->source_ptr + sizeof_element * offset_wextra_from0;

	map->access_level = OMP_DATA_MAP_ACCESS_LEVEL_DIST;
	return total;
}

long omp_loop_get_range(omp_offloading_t *off, int loop_level, long *start, long *length) {
	if (off->loop_dist[loop_level].info == NULL) {
		omp_loop_iteration_dist(off);
	}
	*start = 0;
	*length = off->loop_dist[loop_level].length;
	return off->loop_dist[loop_level].offset;
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

/**
 * Calculate row-major array element offset from the beginning of an array given the multi-dimensional index of an element,
 */
long omp_array_element_offset(int ndims, long * dims, long * idx) {
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
 * Calculate row-major array element offset from the beginning of an array given the multi-dimensional index of an element,
 */
long omp_map_element_offset(omp_data_map_t * map) {
	int ndims = map->info->num_dims;
	long * dims = map->info->dims;
	int i;
	long off = 0;
	long mt = 1;
	for (i=ndims-1; i>=0; i--) {
		off += mt * map->map_dist[i].offset;
		mt *= dims[i];
	}
	return off;
}

/**
 * this function creates host buffer, if needed, and marshall data to the host buffer,
 *
 * it will also create device memory region (both the array region memory and halo region memory
 */
void omp_map_malloc(omp_data_map_t *map, omp_offloading_t *off) {
    int i;
	omp_data_map_info_t *map_info = map->info;
	omp_offloading_info_t * off_info = map_info->off_info;
	omp_grid_topology_t * top = off_info->top;
	int sizeof_element = map_info->sizeof_element;

	if (map->map_type == OMP_DATA_MAP_SHARED) {
		map->map_dev_ptr = map->map_source_ptr;
		map->map_dev_wextra_ptr = map->map_source_wextra_ptr;
	} else if (map->map_type == OMP_DATA_MAP_COPY) {
		map->map_dev_wextra_ptr = omp_map_malloc_dev(map->dev, map->map_source_wextra_ptr, map->map_wextra_size);
		map->map_dev_ptr = map->map_dev_wextra_ptr; /* this will be updated if there is halo region */
	} else {
		printf("unknown map type at this stage: map: %X, %d\n", map, map->map_type);
		abort();
	}

	/*
	* The halo memory management use an attached approach, i.e. the halo region is part of the main computation subregion, and those
	* left/right in/out are buffers for gathering and scattering halo region elements to its correct location.
	*
	* We may use a detached approach, i.e. just use the halo buffer for computation, however, that will involve more complicated
	* array index calculation.
	*/

	/*********************************************************************************************************************************************************/
	/************************* Barrier may be needed among all participating devs since we are now using neighborhood devs **********************************/
	/*********************************************************************************************************************************************************/

	/* TODO: so far, we only handle row-wise halo region in the first dimension of an array, i.e. contiginous memory space */
	if (map_info->num_halo_dims) {
		/* TODO: assert map->map_size != map->map_wextra_size; */
		if (map->mem_noncontiguous) {
			printf("So far, we only handle row-wise halo region in the first dimension of an array\n");
			abort();
		}
		//		BEGIN_SERIALIZED_PRINTF(off->devseqid);
		/* mem size of a row */
		long row_size = sizeof_element;
		for (i = 1; i < map_info->num_dims; i++) {
			row_size *= map_info->dims[i];
		}
		omp_data_map_halo_region_info_t *halo_info = &map_info->halo_info[0];
		omp_data_map_halo_region_mem_t *halo_mem = &map->halo_mem[0];
		halo_mem->left_in_size = halo_info->left * row_size;
		map->map_dev_ptr = map->map_dev_wextra_ptr + halo_mem->left_in_size;
		halo_mem->left_in_ptr = map->map_dev_wextra_ptr;
		halo_mem->left_out_size = halo_info->right * row_size;
		halo_mem->left_out_ptr = map->map_dev_ptr;
		//printf("dev: %d, halo left in size: %d, left in ptr: %X, left out size: %d, left out ptr: %X\n", off->devseqid,
		//	   halo_mem->left_in_size,halo_mem->left_in_ptr,halo_mem->left_out_size,halo_mem->left_out_ptr);
		if (halo_mem->left_dev_seqid >= 0) {
			omp_device_t *leftdev = &omp_devices[top->idmap[halo_mem->left_dev_seqid]];
			if (!omp_map_enable_memcpy_DeviceToDevice(leftdev, map->dev)) { /* no peer2peer access available, use host relay */
#define USE_HOSTMEM_AS_RELAY 1
#ifndef USE_HOSTMEM_AS_RELAY
				/** FIXME, mem leak here and we have not thought where to free */
				halo_mem->left_in_host_relay_ptr = (char *) malloc(halo_mem->left_in_size);
#else
				/* we can use the host memory for the halo region as the relay buffer */
				halo_mem->left_in_host_relay_ptr = map->map_source_wextra_ptr;
#endif
				halo_mem->left_in_data_in_relay_pushed = 0;
				halo_mem->left_in_data_in_relay_pulled = 0;

			//	printf("dev: %d, map: %X, left: %d, left host relay buffer allocated\n", off->devseqid, map, halo_mem->left_dev_seqid);
			} else {
			//	printf("dev: %d, map: %X, left: %d, left dev: p2p enabled\n", off->devseqid, map, halo_mem->left_dev_seqid);
				halo_mem->left_in_host_relay_ptr = NULL;
			}
		} //else map->map_dev_ptr = map->map_dev_wextra_ptr;

		halo_mem->right_in_size = halo_info->right * row_size;
		halo_mem->right_in_ptr = &((char *) map->map_dev_ptr)[map->map_size];
		halo_mem->right_out_size = halo_info->left * row_size;
		halo_mem->right_out_ptr = &((char *) halo_mem->right_in_ptr)[0 - halo_mem->right_out_size];
		//printf("dev: %d, halo right in size: %d, right in ptr: %X, right out size: %d, right out ptr: %X\n", off->devseqid,
		//	   halo_mem->right_in_size,halo_mem->right_in_ptr,halo_mem->right_out_size,halo_mem->right_out_ptr);
		if (halo_mem->right_dev_seqid >= 0) {
			omp_device_t *rightdev = &omp_devices[top->idmap[halo_mem->right_dev_seqid]];
			if (!omp_map_enable_memcpy_DeviceToDevice(rightdev, map->dev)) { /* no peer2peer access available, use host relay */
#ifndef USE_HOSTMEM_AS_RELAY
				/** FIXME, mem leak here and we have not thought where to free */
				halo_mem->right_in_host_relay_ptr = (char *) malloc(halo_mem->right_in_size);
#else
				/* we can use the host memory for the halo region as the relay buffer */
				halo_mem->right_in_host_relay_ptr = &((char *) map->map_source_ptr)[map->map_size];
#endif
				halo_mem->right_in_data_in_relay_pushed = 0;
				halo_mem->right_in_data_in_relay_pulled = 0;

			//	printf("dev: %d, map: %X, right: %d, right host relay buffer allocated\n", off->devseqid, map, halo_mem->right_dev_seqid);
			} else {
			//	printf("dev: %d, map: %X, right: %d, right host p2p enabled\n", off->devseqid, map, halo_mem->right_dev_seqid);
				halo_mem->right_in_host_relay_ptr = NULL;
			}
		}
		//		END_SERIALIZED_PRINTF();
	}

	map->access_level = OMP_DATA_MAP_ACCESS_LEVEL_MALLOC;
}

/**
 * seqid is the sequence id of the device in the top, it is also used as index to access maps
 */
void omp_map_free(omp_data_map_t *map, omp_offloading_t *off) {
	if (map->map_type == OMP_DATA_MAP_COPY)
		omp_map_free_dev(map->dev, map->map_dev_wextra_ptr, 0);
}

void omp_print_map_info(omp_data_map_info_t * info) {
	int i;
	printf("MAPS for %s", info->symbol);
	for (i=0; i<info->num_dims;i++) printf("[%d]", info->dims[i]);
	printf(", ");
	if (info->num_halo_dims) {
		printf("halo");
		omp_data_map_halo_region_info_t * halo_info = info->halo_info;
		for (i=0; i<info->num_dims;i++) printf("[%d|%d]", halo_info[i].left, halo_info[i].right);
		printf(", ");
	}
	printf("direction: %d, map_type: %d, ptr: %X\n", info->map_direction, info->map_type, info->source_ptr);
	for (i=0; i<info->off_info->top->nnodes; i++) {
		printf("\t%d, ", i);
		omp_print_data_map(&info->maps[i]);
	}
}

void omp_print_data_map(omp_data_map_t * map) {
	omp_data_map_info_t * info = map->info;
	printf("dev %d(%s), ", map->dev->id, omp_get_device_typename(map->dev));
	int soe = info->sizeof_element;
	int i;
	//for (i=0; i<info->num_dims;i++) printf("[%d:%d]", map->map_dist[i].offset, map->map_dist[i].offset+map->map_dist[i].length-1);
	for (i=0; i<info->num_dims;i++) printf("[%d:%d]", map->map_dist[i].offset, map->map_dist[i].length);

	char * mem = "COPY";
	if (map->map_type == OMP_DATA_MAP_SHARED) {
		mem = "SHARED";
	}
	printf(", size: %d, size wextra: %d, mem: %s\n",map->map_size, map->map_wextra_size, mem);
	printf("\t\tsrc ptr: %X, src wextra ptr: %X, dev ptr: %X, dev wextra ptr: %X\n", map->map_source_ptr, map->map_source_wextra_ptr, map->map_dev_ptr, map->map_dev_wextra_ptr);
	if (info->num_halo_dims) {
		//printf("\t\thalo memory:\n");
		omp_data_map_halo_region_mem_t * all_halo_mems = map->halo_mem;
		for (i=0; i<info->num_dims; i++) {
			omp_data_map_halo_region_mem_t * halo_mem = &all_halo_mems[i];
			printf("\t\t%d-d halo, L_IN: %X[%d], L_OUT: %X[%d], R_OUT: %X[%d], R_IN: %X[%d]", i,
				   halo_mem->left_in_ptr, halo_mem->left_in_size/soe, halo_mem->left_out_ptr, halo_mem->left_out_size/soe,
				   halo_mem->right_out_ptr, halo_mem->right_out_size/soe, halo_mem->right_in_ptr, halo_mem->right_in_size/soe);
			//printf(", L_IN_relay: %X, R_IN_relay: %X", halo_mem->left_in_host_relay_ptr, halo_mem->right_in_host_relay_ptr);
			printf("\n");
		}
	}
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
#define CORRECTNESS_CHECK 1
#undef CORRECTNESS_CHECK
void omp_halo_region_pull(omp_data_map_t * map, int dim, omp_data_map_exchange_direction_t from_left_right) {
	omp_data_map_info_t * info = map->info;
	/*FIXME: let us only handle 2-D array now */
	if (dim != 0 || map->mem_noncontiguous) {
		//fprintf(stderr, "we only handle noncontiguous distribution and halo at dimension 0 so far!\n");
		omp_print_map_info(map->info);
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
			//if (map->map_type == OMP_DATA_MAP_COPY || left_map->map_type == OMP_DATA_MAP_COPY)
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

#undef CORRECTNESS_CHECK

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
			long region_line_size = map->map_dist[1].length*sizeof_element;
			long full_line_size = info->dims[1]*sizeof_element;
			long region_off = 0;
			long full_off = 0;
			char * src_ptr = &info->source_ptr[sizeof_element*info->dims[1]*map->map_dist[0].offset + sizeof_element*map->map_dist[1].offset];
			for (i=0; i<map->map_dist[0].length; i++) {
				memcpy((void*)&src_ptr[full_off], (void*)&map->map_source_ptr[region_off], region_line_size);
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
void * omp_map_marshal(omp_data_map_t *map) {
	omp_data_map_info_t * info = map->info;
	int sizeof_element = info->sizeof_element;
	int i;
	switch (info->num_dims) {
		case 1: {
			fprintf(stderr,
					"data marshall can only do 2-d or 3-d array, currently is 1-d\n");
			break;
		}
		case 2: {
			void *map_buffer = (void*) malloc(map->map_size);
			long region_line_size = map->map_dist[1].length * sizeof_element;
			long full_line_size = info->dims[1] * sizeof_element;
			long region_off = 0;
			long full_off = 0;
			char * src_ptr = &info->source_ptr[sizeof_element * info->dims[1] * map->map_dist[0].offset
											   + sizeof_element * map->map_dist[1].offset];
			for (i = 0; i < map->map_dist[0].length; i++) {
				memcpy((void*)&map_buffer[region_off], (void*)&src_ptr[full_off], region_line_size);
				region_off += region_line_size;
				full_off += full_line_size;
			}
			return map_buffer;
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

	return NULL;
//	printf("total %ld bytes of data marshalled\n", region_off);
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
 * simple and common topology setup, i.e. devid is from 0 to nnodes-1
 */
omp_grid_topology_t * omp_grid_topology_init_simple(int ndevs, int ndims) {
	/* idmaps array is right after the object in mem */
	omp_grid_topology_t * top = (omp_grid_topology_t*) malloc(sizeof(omp_grid_topology_t)+ sizeof(int)* ndevs);
	top->nnodes = ndevs;
	top->ndims = ndims;
	top->idmap = (int*)&top[1];
	omp_factor(ndevs, top->dims, ndims);
	int i;
	for (i=0; i<ndims; i++) {
		top->periodic[i] = 0;
	}

	for (i=0; i< ndevs; i++) {
		top->idmap[i] = omp_devices[i].id;
	}
	return top;
}

/**
 * Init a topology of devices froma a list of devices identified in the array @devs of @ndevs devices
 */
omp_grid_topology_t * omp_grid_topology_init(int ndevs, int *devs, int ndims) {
	omp_grid_topology_t * top = (omp_grid_topology_t*) malloc(sizeof(omp_grid_topology_t)+ sizeof(int)*ndevs);
	top->nnodes = ndevs;
	top->ndims = ndims;
	top->idmap = (int*)&top[1];
	omp_factor(ndevs, top->dims, ndims);
	int i;
	for (i=0; i<ndims; i++) {
		top->periodic[i] = 0;
	}

	for (i=0; i<ndevs; i++) {
		top->idmap[i] = devs[i];
	}
	return top;
}

void omp_grid_topology_fini(omp_grid_topology_t * top) {
	free(top);
}

void omp_topology_print(omp_grid_topology_t * top) {
	printf("top: %X (%d): ", top, top->nnodes);
	int i;
	for(i=0; i<top->ndims; i++)
		printf("[%d]", top->dims[i]);
	printf("\n");
	for (i=0; i<top->nnodes; i++) {
		printf("%d->%d\n", i, top->idmap[i]);
	}
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


#if defined (OMP_BREAKDOWN_TIMING)
/**
 * sum up all the profiling info of the infos to a info at location 0, all the infos should have the same target and topology.
 * Will also align the start_time with the provided info if it has, otherwise,
 * the call make assumptions that the first info is the one start first and finish last.
 */
void omp_offloading_info_sum_profile(omp_offloading_info_t ** infos, int count, double start_time, double compl_time) {
	int i, j, k;
	omp_offloading_info_t *suminfo = infos[0];
	if (start_time == 0) {
		printf("suminfo start: %f\n", suminfo->start_time);
	} else {
		suminfo->start_time = start_time;
		suminfo->compl_time = compl_time;
	}
	//sprintf(suminfo->name, "Accumulated Profiles %d Types of Offloading", count);
	suminfo->name = "AccumulatedMultipleTypesofOffloading";
	for (i = 0; i < suminfo->top->nnodes; i++) {
		suminfo->offloadings[i].num_events = misc_event_index_start;
	}
	//printf("count: %d, #events: %d, #dev: %d\n", count, misc_event_index_start, suminfo->top->nnodes);

	for (i = 0; i < suminfo->top->nnodes; i++) {
		for (k = 0; k < misc_event_index_start; k++) {
			omp_event_t *sumev = &(suminfo->offloadings[i].events[k]);
			for (j = 1; j < count; j++) {
				omp_offloading_info_t *info = infos[j];
				omp_offloading_t *off = &info->offloadings[i];
				omp_event_t *ev = &off->events[k];
				sumev->elapsed_host += ev->elapsed_host;
				sumev->elapsed_dev += ev->elapsed_dev;
				//printf("%d %d %d\n", k, i, j);

				if (sumev->event_name == NULL) sumev->event_name = ev->event_name;
				//if (strlen(sumev->event_description) == 0) memcpy(sumev->event_description, ev->event_description, strlen(ev->event_description));
			}
			if (k == total_event_index) { /* 0, the first one */
				sumev->start_time_host = suminfo->start_time;
				sumev->start_time_dev = suminfo->start_time;
			} else {
				omp_event_t *lastev = &(suminfo->offloadings[i].events[k-1]);
				sumev->start_time_host = lastev->start_time_host + lastev->elapsed_host;
				sumev->start_time_dev = lastev->start_time_dev + lastev->elapsed_dev;
			}
		}
	}
}

void omp_offloading_info_report_filename(omp_offloading_info_t * info, char * filename) {
	char *original_name = info->name;

	int name_len = 4;
	int guid_len = 8;
	int recu_len = 6;
	char name[name_len];
	int i;
	for (i=0; i<strlen(original_name) && i<name_len; i++) {
		name[i] = original_name[i];
	}
	for (;i<name_len; i++) {
		name[i] = '_';
	}

	int filename_length = name_len + 1 + guid_len + 1 + recu_len + 5 + 1;
	sprintf(filename, "%*s", name_len, name);
	sprintf(filename +name_len, "%s", "_");
	sprintf(filename +name_len+1, "%0*X", guid_len, info);
	sprintf(filename +name_len+1+guid_len, "%s", "_");
	sprintf(filename +name_len+1+guid_len+1, "%0*d", recu_len, info->count);
	sprintf(filename +name_len+1+guid_len+1+recu_len, "%s", ".plot");
	filename[filename_length] = '\0';
}


double omp_event_get_elapsed(omp_event_t *ev) {
	omp_event_record_method_t record_method = ev->record_method;
	if (record_method == OMP_EVENT_HOST_RECORD) {
		return ev->elapsed_host;
	} else if (record_method == OMP_EVENT_DEV_RECORD) {
		return ev->elapsed_dev;
	}
	return 0.0;
}

#if defined(PROFILE_PLOT)
char *colors[] = {
		"#FFFFFF", /* white */
		"#FF0000", /* red */
		"#00FF00", /* Lime */
		"#0000FF", /* Blue */
		"#FFFF00", /* Yellow */
		"#00FFFF", /* Cyan/Aqua */
		"#FF00FF", /* Megenta/Fuchsia */
		"#808080", /* Gray */
		"#800000", /* Maroon */
		"#808000", /* Olive */
		"#008000", /* Green */
		"#800080", /* Purple */
		"#008080", /* Teal */
		"#000080", /* Navy */
};
#endif
void omp_offloading_info_report_profile(omp_offloading_info_t * info) {
	int i, j;
#if defined(PROFILE_PLOT)
	char plotscript_filename[128];
	omp_offloading_info_report_filename(info, plotscript_filename);
	FILE * plotscript_file = fopen(plotscript_filename, "w+");
	fprintf(plotscript_file, "set title \"Offloading (%s) Profile on %d Devices\"\n", info->name, info->top->nnodes);
	int yoffset_per_entry = 10;
	double xsize = (info->compl_time - info->start_time)*1.1;
	double yrange = info->top->nnodes*yoffset_per_entry+12;
	double xrange = yrange * 2.2;
	double xratio = xrange/xsize; /* so new mapped len will be len*xratio */
	fprintf(plotscript_file, "set yrange [0:%f]\n", yrange);
	fprintf(plotscript_file, "set xlabel \"execution time in ms\"\n");
	fprintf(plotscript_file, "set xrange [0:%f]\n", xrange);
	fprintf(plotscript_file, "set style fill pattern 2 bo 1\n");
	fprintf(plotscript_file, "set style rect fs solid 1 noborder\n");
//	fprintf(plotscript_file, "set style line 1 lt 1 lw 1 lc rgb \"#000000\"\n");
	fprintf(plotscript_file, "set border 15 lw 0.2\n");
//	fprintf(plotscript_file, "set style line 2 lt 1 lw 1 lc rgb \"#9944CC\"\n");
	fprintf(plotscript_file, "set xtics out nomirror\n");
	fprintf(plotscript_file, "unset key\n");
	fprintf(plotscript_file, "set ytics out nomirror (");
	int yposoffset = 5;
	omp_offloading_t *off;
	for (i=0; i<info->top->nnodes; i++) {
		off = &info->offloadings[i];
		int devid = off->dev->id;
		int devsysid = off->dev->sysid;
		char *type = omp_get_device_typename(off->dev);
		fprintf(plotscript_file, "\"dev %d(sysid:%d,type:%s)\" %d", devid, devsysid, type, yposoffset);
		yposoffset += yoffset_per_entry;
		if (i != info->top->nnodes - 1) fprintf(plotscript_file, ",");
	}
	fprintf(plotscript_file, ")\n");
	int recobj_count = 1;
	int legend_bottom = info->top->nnodes*yoffset_per_entry+5;
	int legend_width = xrange/12 > 10? xrange/12: 10;
	int legend_offset = legend_width/3;
	for (j=1; j<misc_event_index_start; j++) {
		omp_event_t * ev = &off->events[j];
		if (ev->event_name != NULL) {
			fprintf(plotscript_file, "set object %d rect from %d, %d to %d, %d fc rgb \"%s\"\n",
				recobj_count++, legend_offset, legend_bottom,  legend_offset+legend_width, legend_bottom+3, colors[j]);
			fprintf(plotscript_file, "set label \"%s\" at %d,%d font \"Helvetica,8\'\"\n\n", ev->event_name, legend_offset, legend_bottom-2);
			legend_offset += legend_width + legend_width/3;
		}
	}
/*
set yrange [0:25.5]
set xlabel "execution time in ms"
set xrange [0:25]
set style fill pattern 2 bo 1
set style rect fs solid 1 noborder
set style line 1 lt 1 lw 1 lc rgb "#000000"
set border 15 lw 0.2
set style line 2 lt 1 lw 1 lc rgb "#9944CC"
set xtics out nomirror
unset key
# for each device, we have an entry like the following
set ytics out nomirror ("device 0" 3, "device 1" 6, "device 2" 9, "device 3" 12, "device 4" 15)
*/
#endif

	for (i=0; i<info->top->nnodes; i++) {
		omp_offloading_t * off = &info->offloadings[i];
		int devid = off->dev->id;
		int devsysid = off->dev->sysid;
		char * type = omp_get_device_typename(off->dev);
		printf("\n-------------- Profiles (ms) for Offloading(%s) on %s dev %d (sysid: %d) ---------------\n", info->name,  type, devid, devsysid);
		printf("-------------- Last TOTAL: %.2f, Last start: %.2f ---------------------\n", info->compl_time - info->start_time, info->start_time);
		omp_event_print_profile_header();
		for (j=0; j<off->num_events; j++) {
			omp_event_t * ev = &off->events[j];
			if (j == misc_event_index_start) printf("--------------------- Misc Report ------------------------------------------------------------\n");
			if (ev->count) {
//                printf("%d   ", j);
				double start_time, elapsed;
				omp_event_print_elapsed(ev, info->start_time, &start_time, &elapsed);
#if defined(PROFILE_PLOT)
				if (j>0 && j < misc_event_index_start) { /* only plot the major event */
					fprintf(plotscript_file, "set object %d rect from %f, %d to %f, %d fc rgb \"%s\"\n",
							recobj_count++, start_time, i * yoffset_per_entry,  (start_time + elapsed)*xratio, (i + 1) * yoffset_per_entry, colors[j]);
				}
#endif
			}
		}
		printf("---------------- End Profiling Report for Offloading(%s) on dev: %d ----------------------------\n", info->name, devid);

#if defined(PROFILE_PLOT)
		fprintf(plotscript_file, "set arrow from  0,%d to %f,%d nohead\n", i * yoffset_per_entry, xrange, i * yoffset_per_entry);
#endif
	}

	long length = info->loop_dist_info[0].length; // - info->loop_dist_info[0].start;

	printf("\n============ Accumulated total time (ms) (MAPTO, KERN, MAPFROM, and EXCHANGE) shown as average ==============\n");
	printf("\tDEVICE\t\t\t\tTOTAL\t\tMAPTO(#)\tKERN(#)\t\tMAPFROM(#)\tEXCHANGE(#)\tDIST(%d)\tDIST(\%)\tMODELING OVERHEAD\n", length);
	printf("-------------------------------------------------------------------------------------------------------------\n");
	for (i=0; i<info->top->nnodes; i++) {
		omp_offloading_t * off = &info->offloadings[i];
		int devid = off->dev->id;
		int devsysid = off->dev->sysid;
		char * type = omp_get_device_typename(off->dev);

		/* mapto */
		omp_event_t * ev = &off->events[acc_mapto_event_index];
		int mapto_count = ev->count;
		double mapto_time_ms = (mapto_count != 0) ? omp_event_get_elapsed(ev)/mapto_count: 0;

		/* kern */
		ev = &off->events[acc_kernel_exe_event_index];
		int kernel_count = ev->count;
		double kern_time_ms = (kernel_count != 0) ? omp_event_get_elapsed(ev)/kernel_count: 0;

		/* mapfrom */
		ev = &off->events[acc_mapfrom_event_index];
		int mapfrom_count = ev->count;
		double mapfrom_time_ms = (mapfrom_count != 0) ? omp_event_get_elapsed(ev)/mapfrom_count: 0;

		/* data exchange */
		ev = &off->events[acc_ex_event_index];
		int ex_count = ev->count;
		double ex_time_ms = (ex_count != 0) ? omp_event_get_elapsed(ev)/ex_count: 0;

		double accu_total_ms = mapto_time_ms + kern_time_ms + mapfrom_time_ms + ex_time_ms;

		//	events = &events[total_event_accumulated_index];
		//	printf("%s dev %d (sysid: %d): %.4f\n", type, devid, devsysid, events->elapsed_host/events->count);
		printf("%s dev %d (sysid: %d, %.1f GFLOPS):\t%.2f\t\t%.2f(%d)\t%.2f(%d)\t%.2f(%d)\t%.2f(%d)", type, devid, devsysid, off->dev->total_real_flopss, accu_total_ms, mapto_time_ms, mapto_count, kern_time_ms, kernel_count, mapfrom_time_ms, mapfrom_count, ex_time_ms, ex_count);
		float percentage = 100*((float)off->loop_dist[0].length/(float)length);
		printf("\t%d\t\t%.1f%%", off->loop_dist[0].length, percentage);

		ev = &off->events[runtime_dist_modeling_index];
		if (ev->event_name != NULL) {
			int modeling_count = ev->count;
			double modeling_time_ms = (ex_count != 0) ? omp_event_get_elapsed(ev) / modeling_count : 0;
			printf("\t%.1f(%d) %.1f%%\n", modeling_time_ms, modeling_count, (modeling_time_ms/accu_total_ms)*100.0);
		}
		printf("\n");
	}
	printf("--------------------------- End Accumulated total time report -----------------------------------------------\n\n");

	/* write the report to a CSV file */
	char report_cvs_filename[128];
	sprintf(report_cvs_filename, "%s.csv\0", info->name);
	FILE * report_cvs_file = fopen(report_cvs_filename, "a+");
	char time_buff[100];
	time_t now = time (0);
	strftime (time_buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
	fprintf(report_cvs_file, "%s on %d devices: size: %d, %s\n", info->name, info->top->nnodes, length, time_buff);
	int events_to_print[misc_event_index_start];
	int num_events_to_print = 7;
	events_to_print[0] = total_event_accumulated_index;
	events_to_print[1] = acc_mapto_event_index;
	events_to_print[2] = acc_kernel_exe_event_index;
	events_to_print[3] = acc_mapfrom_event_index;
	events_to_print[4] = acc_ex_event_index;
	events_to_print[5] = total_event_index;
	events_to_print[6] = runtime_dist_modeling_index;

	for (i=0; i<omp_num_devices; i++) {
		omp_device_t * dev = &omp_devices[i];
		int devid = dev->id;
		int devsysid = dev->sysid;
		char * type = omp_get_device_typename(dev);
		fprintf(report_cvs_file, ",%s:%d(sysid:%d)", type, devid, devsysid);
	}
	fprintf(report_cvs_file, "\n");
	int lastdevid = 0;
	for (i=0; i<num_events_to_print; i++) {
		int j;
		lastdevid = 0;
		for (j=0; j<info->top->nnodes; j++) {
			omp_offloading_t *off = &info->offloadings[j];
			int devid = off->dev->id;
			omp_event_t * ev = &off->events[events_to_print[i]];
			if (ev->event_name == NULL) continue;
			int count = ev->count;
			double time_ms = (count != 0) ? omp_event_get_elapsed(ev)/count: 0;
			if (j == 0) fprintf(report_cvs_file, "%s", ev->event_name);
			while(lastdevid <= devid) {
				fprintf(report_cvs_file, ",\t");
				lastdevid++;
			}
			fprintf(report_cvs_file, "%.4f", time_ms);
		}
		fprintf(report_cvs_file, "\n");
	}

	lastdevid = 0;
	for (j=0; j<info->top->nnodes; j++) {
		omp_offloading_t *off = &info->offloadings[j];

		int devid = off->dev->id;
		if (j == 0) fprintf(report_cvs_file, "DIST(%d)", length);
		while(lastdevid <= devid) {
			fprintf(report_cvs_file, ",\t");
			lastdevid++;
		}
		fprintf(report_cvs_file, "%d", off->loop_dist[0].length);
	}
	fprintf(report_cvs_file, "\n");
	lastdevid = 0;
	for (j=0; j<info->top->nnodes; j++) {
		omp_offloading_t *off = &info->offloadings[j];

		int devid = off->dev->id;
		if (j == 0) fprintf(report_cvs_file, "DIST(\%)");
		while(lastdevid <= devid) {
			fprintf(report_cvs_file, ",\t");
			lastdevid++;
		}
//		printf("%d:%d,", off->loop_dist[0].length, length);
		float percentage = 100*((float)off->loop_dist[0].length/(float)length);
		fprintf(report_cvs_file, "%.1f\%", percentage);
	}
	fprintf(report_cvs_file, "\n");
	lastdevid = 0;
	for (j=0; j<info->top->nnodes; j++) {
		omp_offloading_t *off = &info->offloadings[j];
		int devid = off->dev->id;
		if (j == 0) fprintf(report_cvs_file, "DEV GFLOPS");
		while(lastdevid <= devid) {
			fprintf(report_cvs_file, ",\t");
			lastdevid++;
		}
		fprintf(report_cvs_file, "%.1f", off->dev->total_real_flopss);
	}
	fprintf(report_cvs_file, "\n");

	fprintf(report_cvs_file, "-------------------------------------------------------------------------------------------------\n\n");
	fclose(report_cvs_file);

#if defined(PROFILE_PLOT)
	fprintf(plotscript_file, "plot 0\n");
	fclose(plotscript_file);
#endif
}

void omp_event_print_profile_header() {
	printf("%*s    TOTAL     AVE(#Calls) Last Start Host/dev Measure\tDescription\n",
		   OMP_EVENT_NAME_LENGTH-1, "Name");
}

void omp_event_print_elapsed(omp_event_t *ev, double reference, double *start_time, double *elapsed) {
	omp_event_record_method_t record_method = ev->record_method;
	if (record_method == OMP_EVENT_HOST_RECORD) {
		printf("%*s%10.2f%10.2f(%d)%10.2f\thost\t%s\n",
			   OMP_EVENT_NAME_LENGTH, ev->event_name, ev->elapsed_host, ev->elapsed_host/ev->count, ev->count, ev->start_time_host - reference, ev->event_description);
		*start_time = ev->start_time_host - reference;
		*elapsed = ev->elapsed_host;
	} else if (record_method == OMP_EVENT_DEV_RECORD) {
		printf("%*s%10.2f%10.2f(%d)%10.2f\tdev\t%s\n",
			   OMP_EVENT_NAME_LENGTH, ev->event_name, ev->elapsed_dev, ev->elapsed_dev/ev->count, ev->count, ev->start_time_dev - reference, ev->event_description);
		*start_time = ev->start_time_dev - reference;;
		*elapsed = ev->elapsed_dev;
	} else {
		printf("%*s%10.2f%10.2f(%d)%10.2f\thost\t%s\n",
			   OMP_EVENT_NAME_LENGTH, ev->event_name, ev->elapsed_host, ev->elapsed_host/ev->count, ev->count, ev->start_time_host - reference, ev->event_description);
		printf("%*s%10.2f%10.2f(%d)%10.2f\tdev\t%s\n",
			   OMP_EVENT_NAME_LENGTH, ev->event_name, ev->elapsed_dev, ev->elapsed_dev/ev->count, ev->count, ev->start_time_dev - reference, ev->event_description);
		*start_time = ev->start_time_host - reference;;
		*elapsed = ev->elapsed_host;
	}
}
#endif

char * omp_get_device_typename(omp_device_t * dev) {
	int i;
	for (i=0; i<OMP_NUM_DEVICE_TYPES; i++) {
		if (omp_device_types[i].type == dev->type) return omp_device_types[i].shortname;
	}
	return NULL;
}


/* read timer in second */
double read_timer()
{
    return read_timer_ms()/1000.0;
}

/* read timer in ms */
double read_timer_ms()
{
    struct timespec ts;
#if defined(CLOCK_MONOTONIC_PRECISE)
	/* BSD. --------------------------------------------- */
	const clockid_t id = CLOCK_MONOTONIC_PRECISE;
#elif defined(CLOCK_MONOTONIC_RAW)
	/* Linux. ------------------------------------------- */
	const clockid_t id = CLOCK_MONOTONIC_RAW;
#elif defined(CLOCK_HIGHRES)
	/* Solaris. ----------------------------------------- */
	const clockid_t id = CLOCK_HIGHRES;
#elif defined(CLOCK_MONOTONIC)
	/* AIX, BSD, Linux, POSIX, Solaris. ----------------- */
	const clockid_t id = CLOCK_MONOTONIC;
#elif defined(CLOCK_REALTIME)
	/* AIX, BSD, HP-UX, Linux, POSIX. ------------------- */
	const clockid_t id = CLOCK_REALTIME;
#else
	const clockid_t id = (clockid_t)-1;	/* Unknown. */
#endif

	if ( id != (clockid_t)-1 && clock_gettime( id, &ts ) != -1 )
		return (double)ts.tv_sec * 1000.0 +
			(double)ts.tv_nsec / 1000000.0;

}
