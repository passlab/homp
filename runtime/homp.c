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
#include <unistd.h>
#include "inttypes.h"
#include "homp.h"
#include <assert.h>
#include <errno.h>
#include <utmpx.h>
#include <utmpx.h>
#include <numaif.h>

#define PAGEMAP_ENTRY 8
#define GET_BIT(X,Y) (X & ((uint64_t)1<<Y)) >> Y
#define GET_PFN(X) X & 0x7FFFFFFFFFFFFF

const int __endian_bit = 1;
#define is_bigendian() ( (*(char*)&__endian_bit) == 0 )

int i, c, pid, status;
unsigned long virt_addr; 
uint64_t read_val, file_offset;
char path_buf [0x100] = {};
FILE * f;
char *end;
//-----------
#define num_allowed_dist_policies 7
#define default_dist_policy_index 0
omp_dist_policy_argument_t omp_dist_policy_args[num_allowed_dist_policies] = {
		{OMP_DIST_POLICY_BLOCK,              -1,	0.0,	"even distribution",                                                                                                   "BLOCK"},
		{OMP_DIST_POLICY_SCHED_DYNAMIC,      100,	0.0, 	"each dev picks chunks and chunk size does not change",                                                                "SCHED_DYNAMIC"},
		{OMP_DIST_POLICY_SCHED_GUIDED,       100,	0.0,	"each dev picks chunks and chunk sizes reduce each time",                                                              "SCHED_GUIDED"},
		{OMP_DIST_POLICY_MODEL_1_AUTO,       -1,	0.0,	"dist all iterations using compute-based analytical model",                                                            "MODEL_1_AUTO"},
		{OMP_DIST_POLICY_MODEL_2_AUTO,       -1,	0.0,	"dist all iterations using compute/data-based analytical model",                                                       "MODEL_2_AUTO"},
		{OMP_DIST_POLICY_SCHED_PROFILE_AUTO, 100,	0.0,	"each dev pick the same amount of chunks, runtime profiles and then dist the rest based on profiling",                 "SCHED_PROFILE_AUTO"},
		{OMP_DIST_POLICY_MODEL_PROFILE_AUTO, 100,	0.0,	"dist the first chunk among devs using analytical model, runtime profiles, and then dist the rest based on profiling", "MODEL_PROFILE_AUTO"},
};

omp_dist_policy_t LOOP_DIST_POLICY;
float LOOP_DIST_CHUNK_SIZE;
float LOOP_DIST_CUTOFF_RATIO;

__attribute__((destructor))
void omp_print_homp_usage() {
	printf("==========================================================================================================================\n");
	printf("HOMP Usage: The OMP_DEV_SPEC_FILE should be set for pointing to a device specification file!\n");
	omp_print_dist_policy_options();
	omp_read_dist_policy_options(&LOOP_DIST_CHUNK_SIZE, &LOOP_DIST_CUTOFF_RATIO);
}

void omp_print_dist_policy_options() {
	int i;
	printf("==========================================================================================================================\n");
	printf("Loop distribution policy can be specified by setting LOOP_DIST_POLICY environment variable to one of the following:\n");
	printf("  The first number is the policy id, and the second number is the chunk size or chunk percentage. If it is -1, the chunk size is not used.\n");
	printf("  The second number is the chunk size or chunk percentage. If it is -1, the chunk size is not used.\n");
	printf("  The third number is the cutoff_ratio. If it is -0.0, it is not used. cutoff_ratio is used to decide whether to use a device\n");
	printf("      for offloading or not. If a dist length for a dev is less than the cutoff_ratio, the dev will not participate in offloading now on.\n");
	for (i=0; i < num_allowed_dist_policies; i++) {
		printf("\t%2d,<n>,<r>:\t%s: %s, default n: %d, default r: %.2f\n",
			   omp_dist_policy_args[i].type, omp_dist_policy_args[i].shortname, omp_dist_policy_args[i].name, (int)omp_dist_policy_args[i].chunk, omp_dist_policy_args[i].cutoff_ratio);
	}
	i= default_dist_policy_index;
	printf("-------------------------------------------------------------------------------------------------------------------------------\n");
	printf("\tExample: export LOOP_DIST_POLICY=10,200, or export LOOP_DIST_POLICY=8,10%, or export LOOP_DIST_POLICY=12,10%,0.01%\n");
	printf("\tDefault: %d,%d,%.2f%%:\t%s(%s)\n",
		   omp_dist_policy_args[i].type, (int)omp_dist_policy_args[i].chunk, omp_dist_policy_args[i].cutoff_ratio, omp_dist_policy_args[i].shortname, omp_dist_policy_args[i].name);
	printf("==========================================================================================================================\n");

}

/*
 * *chunk_size stores percentage as negative integer numbers
 *
 */
omp_dist_policy_t omp_read_dist_policy_options(float *chunk_size, float *cutoff_ratio) {
	omp_dist_policy_t p;
	char *env_str = getenv("LOOP_DIST_POLICY");
	int i = num_allowed_dist_policies;

	int use_percentage = 0;
	if (env_str != NULL) {
		char dist_policy_str[strlen(env_str)];
		strcpy(dist_policy_str, env_str);
		const char s[2] = ",";

		char *token = strtok(dist_policy_str, s);
		sscanf(token, "%d", &p);

		for (i = 0; i < num_allowed_dist_policies; i++) {
			if (omp_dist_policy_args[i].type == p) break;
		}
		if (i == num_allowed_dist_policies) {
			i = default_dist_policy_index;
			p = omp_dist_policy_args[i].type;
			*chunk_size = omp_dist_policy_args[i].chunk;
			*cutoff_ratio = omp_dist_policy_args[i].cutoff_ratio;
		} else {
			token = strtok(NULL, s);
			if (token != NULL) {
				if (token[strlen(token) - 1] == '%') {
					sscanf(token, "%f%%", chunk_size);
					use_percentage = 1;
				} else {
					sscanf(token, "%f", chunk_size);
				}
				token = strtok(NULL, s);

				if (token != NULL) {
					if (token[strlen(token) - 1] == '%') {
						sscanf(token, "%f%%", cutoff_ratio);
					} else {
						sscanf(token, "%f", cutoff_ratio);
					}
				} else {
					*cutoff_ratio = omp_dist_policy_args[i].cutoff_ratio;
				}
			} else {
				*chunk_size = omp_dist_policy_args[i].chunk;
				*cutoff_ratio = omp_dist_policy_args[i].cutoff_ratio;
			}
		}
	} else {
		i = default_dist_policy_index;
		p = omp_dist_policy_args[i].type;
		*chunk_size = omp_dist_policy_args[i].chunk;
		*cutoff_ratio = omp_dist_policy_args[i].cutoff_ratio;
	}

//	printf("Dist policy: %d,%d: %s, full name: %s\n", omp_dist_policy_args[i].type, *chunk_size, omp_dist_policy_args[i].shortname, omp_dist_policy_args[i].name);
	printf("--------------------------------------------------------------------------------------------------------------------\n");
	printf("Current dist policy is set and stored in the following global variables that can be used in the program.\n");
	printf("\tLOOP_DIST_POLICY       = %d %s(%s)\n", omp_dist_policy_args[i].type, omp_dist_policy_args[i].shortname, omp_dist_policy_args[i].name);
	printf("\tLOOP_DIST_CHUNK_SIZE   = %.2f", *chunk_size); if (use_percentage) printf("%%"); printf("\n");
	printf("\tLOOP_DIST_CUTOFF_RATIO = %.2f%%", *cutoff_ratio);
	printf("\n");

	printf("==========================================================================================================================\n\n");

	/* we use negative chunk_size for percentage */
	if (use_percentage) *chunk_size = 0-*chunk_size;
	*cutoff_ratio = *cutoff_ratio / 100.0;
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
	info->recurring = recurring;
	info->count = 0;
	info->type = off_type;
	if (off_type == OMP_OFFLOADING_DATA) { /* we handle offloading data as two steps, thus a recurring offloading */
		info->recurring = 1;
	}
	info->num_mapped_vars = num_maps;
	info->kernel_launcher = kernel_launcher;
	info->args = args;
	info->halo_x_info = NULL;
	info->loop_redist_needed = 0;
	info->start_time = 0;
	info->loop_depth = loop_nest_depth;
	for (i=0; i<loop_nest_depth; i++) {
		info->loop_dist_info[i].target_type = OMP_DIST_TARGET_LOOP_ITERATION;
		info->loop_dist_info[i].target = info;
		info->loop_dist_info[i].target_dim = i;
	}

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
		off->events == NULL;
		off->num_events = 0;
		off->loop_dist_done = 0;
		off->runtime_profile_elapsed = -1.0;
		off->last_total = 1; /* this serve as flag to see whether re-dist should be done or not */
	}

	info->nums_run = 1;

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
	info->recurring = recurring;
	info->count = 0;
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

void omp_init_dist_info(omp_dist_info_t *dist_info, omp_dist_policy_t dist_policy, long offset, long length,
						float chunk_size, int topdim) {
	dist_info->offset = offset;
	dist_info->start = offset;
	dist_info->length = length;
	dist_info->end = offset + length;
	dist_info->policy = dist_policy;
	dist_info->dim_index = topdim;
	dist_info->redist_needed = 0;
	dist_info->chunk_size = chunk_size;
}

void omp_data_map_dist_init_info(omp_data_map_info_t *map_info, int dim, omp_dist_policy_t dist_policy, long offset,
								 long length, float chunk_size, int topdim) {
	omp_dist_info_t * dist_info = &map_info->dist_info[dim];
	omp_init_dist_info(dist_info, dist_policy, offset, length, chunk_size, topdim);
}

/**
 * chunk_size store percentage as negative integer numbers
 */
void omp_loop_dist_init_info(omp_offloading_info_t *off_info, int level, omp_dist_policy_t dist_policy, long offset,
							 long length, float chunk_size, int topdim) {
	omp_dist_info_t * dist_info = &off_info->loop_dist_info[level];
	omp_init_dist_info(dist_info, dist_policy, offset, length, chunk_size, topdim);
	if (dist_policy == OMP_DIST_POLICY_SCHED_DYNAMIC ||
		dist_policy == OMP_DIST_POLICY_SCHED_GUIDED ||
		dist_policy == OMP_DIST_POLICY_SCHED_FEEDBACK ||
		dist_policy == OMP_DIST_POLICY_SCHED_PROFILE_AUTO ||
		dist_policy == OMP_DIST_POLICY_MODEL_PROFILE_AUTO) {
		dist_info->redist_needed = 1; /* multiple distributions are needed after the initial one */
		off_info->loop_redist_needed = dist_info->redist_needed;
	}
}

void omp_loop_dist_static_ratio(omp_offloading_info_t *off_info, int level, long offset, long length, float * ratios, int topdim) {
	omp_dist_info_t * dist_info = &off_info->loop_dist_info[level];
	omp_init_dist_info(dist_info, OMP_DIST_POLICY_SCHED_STATIC_RATIO, offset, length, 0, topdim);
	int i;
	for (i=0; i<off_info->top->dims[topdim]; i++) {
		/* TODO: */
	}
}

/* return whether realignment is needed or not for this align in the future */
static int omp_set_align_dist_policy(omp_dist_info_t * dist_info, omp_dist_target_type_t alignee_type, void * alignee, int alignee_dim, long offset) {
	omp_dist_info_t *alignee_dist_info = NULL;
	if (alignee_type == OMP_DIST_TARGET_DATA_MAP) {
		omp_data_map_info_t *alignee_map_info = (omp_data_map_info_t *) alignee;
		alignee_dist_info = &alignee_map_info->dist_info[alignee_dim];
	} else { /* OMP_DIST_TARGET_LOOP_ITERATION */
		alignee_dist_info = &((omp_offloading_info_t*)alignee)->loop_dist_info[alignee_dim];
	}

	offset = (offset == OMP_ALIGNEE_OFFSET ? alignee_dist_info->offset : offset);
	if (alignee_dist_info->policy == OMP_DIST_POLICY_ALIGN) {/* if there is a chain of alignment, make it point to the root */
		return omp_set_align_dist_policy(dist_info, alignee_dist_info->alignee_type, alignee_dist_info->alignee.data_map_info, alignee_dist_info->dim_index, offset);
	} else {
		omp_init_dist_info(dist_info, OMP_DIST_POLICY_ALIGN, offset, 0, 0, alignee_dim);
		dist_info->alignee_type = alignee_type;
		dist_info->alignee.data_map_info = (omp_data_map_info_t *) alignee;
		dist_info->redist_needed = alignee_dist_info->redist_needed;
	}

	return dist_info->redist_needed;
}

/* to align one data map with another data map, if dim>=0, align a specific dim, if dim<0, align all the dims */
void omp_data_map_dist_align_with_data_map(omp_data_map_info_t *map_info, int dim, long offset,
                                           omp_data_map_info_t *alignee, int alignee_dim) {
	int redist_needed = 0;
	if (dim >= 0 && alignee_dim >=0) {
		redist_needed = omp_set_align_dist_policy(&map_info->dist_info[dim], OMP_DIST_TARGET_DATA_MAP, alignee, alignee_dim, offset);
	} else if (dim == OMP_ALL_DIMENSIONS && alignee_dim == OMP_ALL_DIMENSIONS){ /* for all the dimensions that will be aligned */
		int i;
		for (i=0; i<map_info->num_dims;i++) {
			if (omp_set_align_dist_policy(&map_info->dist_info[i], OMP_DIST_TARGET_DATA_MAP, alignee, i, offset))
				redist_needed = 1;
		}
	} else if (dim == OMP_ALL_DIMENSIONS && alignee_dim >=0) {
		int i;
		for (i=0; i<map_info->num_dims;i++) {
			if (omp_set_align_dist_policy(&map_info->dist_info[i], OMP_DIST_TARGET_DATA_MAP, alignee, alignee_dim, offset))
				redist_needed = 1;
		}
	} else {
		abort();
	}
	if (redist_needed) map_info->remap_needed = 1;
}

/* to align one data map with another data map, if dim>=0, align a specific dim, if dim<0, align all the dims */
void omp_data_map_dist_align_with_data_map_with_halo(omp_data_map_info_t *map_info, int dim, long offset,
                                                     omp_data_map_info_t *alignee, int alignee_dim) {
	int redist_needed = 0;
	if (dim >= 0 && alignee_dim >=0) {
		redist_needed = omp_set_align_dist_policy(&map_info->dist_info[dim], OMP_DIST_TARGET_DATA_MAP, alignee, alignee_dim, offset);
		if (alignee->num_halo_dims) {
			omp_data_map_halo_region_info_t * halo_info = &alignee->halo_info[alignee_dim];
			omp_map_add_halo_region(map_info, dim, halo_info->left, halo_info->right, halo_info->edging);
		}
	} else if (dim == OMP_ALL_DIMENSIONS && alignee_dim == OMP_ALL_DIMENSIONS){ /* for all the dimensions that will be aligned */
		int i;
		for (i=0; i<map_info->num_dims;i++) {
			if (omp_set_align_dist_policy(&map_info->dist_info[i], OMP_DIST_TARGET_DATA_MAP, alignee, i, offset))
				redist_needed = 1;
			if (alignee->num_halo_dims) {
				omp_data_map_halo_region_info_t * halo_info = &alignee->halo_info[i];
				omp_map_add_halo_region(map_info, i, halo_info->left, halo_info->right, halo_info->edging);
			}
		}
	} else if (dim == OMP_ALL_DIMENSIONS && alignee_dim >=0) {
		int i;
		for (i=0; i<map_info->num_dims;i++) {
			if (omp_set_align_dist_policy(&map_info->dist_info[i], OMP_DIST_TARGET_DATA_MAP, alignee, alignee_dim, offset))
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

void omp_data_map_dist_align_with_loop(omp_data_map_info_t *map_info, int dim, long offset,
                                       omp_offloading_info_t *alignee, int alignee_level) {
	int redist_needed = 0;
	if (dim >= 0 && alignee_level >=0) {
		redist_needed = omp_set_align_dist_policy(&map_info->dist_info[dim], OMP_DIST_TARGET_LOOP_ITERATION, alignee, alignee_level, offset);
	} else if (dim == OMP_ALL_DIMENSIONS && alignee_level == OMP_ALL_DIMENSIONS){ /* for all the dimensions that will be aligned */
		int i;
		for (i=0; i<map_info->num_dims;i++) {
			if (omp_set_align_dist_policy(&map_info->dist_info[i], OMP_DIST_TARGET_LOOP_ITERATION, alignee, i, offset))
				redist_needed = 1;
		}
	} else if (dim == OMP_ALL_DIMENSIONS && alignee_level >=0) {
		int i;
		for (i=0; i<map_info->num_dims;i++) {
			if (omp_set_align_dist_policy(&map_info->dist_info[i], OMP_DIST_TARGET_LOOP_ITERATION, alignee, alignee_level, offset))
				redist_needed = 1;
		}
	} else {
		abort();
	}
	if (redist_needed) map_info->remap_needed = 1;
}

void omp_loop_dist_align_with_data_map(omp_offloading_info_t *loop_off_info, int level, long offset,
                                       omp_data_map_info_t *alignee, int alignee_dim) {
	if (level >= 0 && alignee_dim >=0) {
		omp_set_align_dist_policy(&loop_off_info->loop_dist_info[level], OMP_DIST_TARGET_DATA_MAP, alignee, alignee_dim, offset);
	} else if (level == OMP_ALL_DIMENSIONS && alignee_dim == OMP_ALL_DIMENSIONS){ /* for all the dimensions that will be aligned */
		int i;
		for (i=0; i<loop_off_info->loop_depth;i++) {
			omp_set_align_dist_policy(&loop_off_info->loop_dist_info[i], OMP_DIST_TARGET_DATA_MAP, alignee, i, offset);
		}
	} else if (level == OMP_ALL_DIMENSIONS && alignee_dim >=0) {
		int i;
		for (i=0; i<loop_off_info->loop_depth;i++) {
			omp_set_align_dist_policy(&loop_off_info->loop_dist_info[level], OMP_DIST_TARGET_DATA_MAP, alignee, alignee_dim, offset);
		}
	} else {
		abort();
	}
	loop_off_info->loop_redist_needed = alignee->remap_needed;
}

void omp_loop_dist_align_with_loop(omp_offloading_info_t *loop_off_info, int level, long offset,
                                   omp_offloading_info_t *alignee, int alignee_level) {
	if (level >= 0 && alignee_level >=0) {
		omp_set_align_dist_policy(&loop_off_info->loop_dist_info[level], OMP_DIST_TARGET_LOOP_ITERATION, alignee, alignee_level, offset);
	} else if (level == OMP_ALL_DIMENSIONS && alignee_level == OMP_ALL_DIMENSIONS){ /* for all the dimensions that will be aligned */
		int i;
		for (i=0; i<loop_off_info->loop_depth;i++) {
			omp_set_align_dist_policy(&loop_off_info->loop_dist_info[i], OMP_DIST_TARGET_LOOP_ITERATION, alignee, i, offset);
		}
	} else if (level == OMP_ALL_DIMENSIONS && alignee_level >=0) {
		int i;
		for (i=0; i<loop_off_info->loop_depth;i++) {
			omp_set_align_dist_policy(&loop_off_info->loop_dist_info[level], OMP_DIST_TARGET_LOOP_ITERATION, alignee, alignee_level, offset);
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
				//(map);
				return map;
			}
		}
	}

	/* STEP 3: seach the offloading stack if this inherits data map from previous data offloading */
	//printf("omp_map_get_map: off: %X, devid: %d, off_info: %X, host_ptr: %X\n", off, off->devseqid, off_info, host_ptr);
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
	map->dev = dev; /* mainly use as cache so we save one pointer deference */
	map->mem_noncontiguous = 0;
	map->map_type = info->map_type;
	map->dist_counter = 0;
	map->next = NULL;
	map->total_map_size = 0;
	map->total_map_wextra_size = 0;

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
		if (map->info == NULL) map->map_dist[i].acc_total_length = 0; /* a strange way to use this as a flag */
		map->map_dist[i].total_length = 0;
		map->map_dist[i].info = &info->dist_info[i];
	}
	map->info = info;
	map->access_level = OMP_DATA_MAP_ACCESS_LEVEL_INIT;
}

/* forward declaration to suppress compiler warning */
void omp_topology_get_neighbors(omp_grid_topology_t * top, int seqid, int topdim, int cyclic, int* left, int* right);


void omp_dist_block(long offset, long full_length, long position, int dim, long * offoffset, long *length) {
	/* partition the array region into subregion and save it to the map */
	long remaint = full_length % dim;
	long esize = full_length / dim;
	//	printf("n: %d, seqid: %d, map_dist: topdimsize: %d, remaint: %d, esize: %d\n", n, seqid, topdimsize, remaint, esize);
	long len, start;
	if (position < remaint) { /* each of the first remaint dev has one more element */
		len = esize + 1;
		start = (esize + 1) * position;
	} else {
		len = esize;
		start = esize * position + remaint;
	}
	*offoffset = start + offset;
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

static void omp_dist_with_cutoff(int ndev, float *ratios, float cutoff_ratio, long full_length, int seqid, long start,
								 long *myoffset, long *mylength) {
	int i;
	int last_dev = ndev - 1;
	float total_ratios = 0.0;
#ifdef DEBUG_CUTOFF
	if (seqid == 0) {
		printf("-----------------------------------------------------------------\n");
		printf("before applying cutoff %.2f: ", cutoff_ratio);
		for (i=0; i<ndev; i++) {
			printf("%0.2f     ", ratios[i]);
		}
		printf("\n");
	}
#endif
    int fatest = 0;
	float fatest_ratio = - 100.0;
	int num_cutoff = 0.0;
	for (i=0; i< ndev; i++) {
		float rat = ratios[i];
		if (rat > fatest_ratio) {
			fatest = i;
			fatest_ratio = rat;
		}
		if (rat < cutoff_ratio) {
			ratios [i] = 0.0;
			rat = 0;
			num_cutoff++;
		} else last_dev = i;
		total_ratios += rat;
	}

	/* in rare case, if the cutoff is inappropriately set and everybody is cut off, we will choose the fastest one */

	if (num_cutoff == ndev) {
		ratios[fatest] = fatest_ratio;
		total_ratios = fatest_ratio;
	}

#ifdef DEBUG_CUTOFF
	if (seqid == 0) {
		printf("after  applying cutoff %.2f: ", cutoff_ratio);
		for (i=0; i<ndev; i++) {
			printf("%0.2f     ", ratios[i]/total_ratios);
		}
		printf("\n");
		printf("-----------------------------------------------------------------\n");
	}
#endif
	long offset = 0;
	long length = 0;
	for (i=0; i<ndev; i++) {
		length = ratios[i]/total_ratios * full_length;
		if (length >= full_length) length = full_length;
		if (i == last_dev && offset + length != full_length) { /* fix rounding error */
			length = full_length - offset;
		}
		if (seqid == i) {
			*myoffset = offset + start;
			*mylength = length;
		}
		offset += length;
	}
}
/**
 * dist according to analytical model
 *
 * cutoff_ratio: if the ratio dist to the dev is less than this cutoff_ratio, this dev will be turned off, thus length = 0
 */
static void omp_dist_model(omp_dist_policy_t model_algorithm, omp_dist_info_t *dist_info, long start, long full_length,
						   omp_grid_topology_t *top, int *coords, int seqid, long *myoffset, long *mylength,
						   float cutoff_ratio) {
	long offset = 0;
	long length = 0;
	int i, j;
	omp_device_t *dev = &omp_devices[top->idmap[seqid]];

	/* in LINEAR_MODEL_1, only computation is considered */
	if (dist_info->target_type == OMP_DIST_TARGET_LOOP_ITERATION ) {
	} else {
		abort();
		/* error, so far AUTO is only applied to loop iteration */
	}
	omp_offloading_info_t * off_info = (omp_offloading_info_t*)dist_info->target;
	int ndev = off_info->top->nnodes;
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
    if (model_algorithm == OMP_DIST_POLICY_MODEL_1_AUTO) {
		/* compute the total capability */
		double total_flops = 0.0;
		double flops;
		double all_flops[ndev];
		float ratios[ndev];
		for (i = 0; i < ndev; i++) {
			flops = off_info->offloadings[i].dev->total_real_flopss;
			total_flops += flops;
			all_flops[i] = flops;
		}

		for (i=0; i<ndev; i++) {
			ratios[i] = all_flops[i]/total_flops;
		}

		omp_dist_with_cutoff(ndev, ratios, cutoff_ratio, full_length, seqid, start, myoffset, mylength);
		//printf("MODEL_AUTO: LINEAR_MODEL_1: Dev %d (%f GFlops/s): offset: %d, length: %d of total length: %d\n", i, flops, *myoffset, *mylength, full_length);
	} else { /* OMP_DIST_POLICY_MODEL_2_AUTO */
		/* in LINEAR_MODEL_2, computation and data movement cost are considered */
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
		struct align_map {
			omp_data_map_t *map;
			omp_dist_t *align_dist;
		} align_maps[off_info->num_mapped_vars];
		int num_aligned_maps = 0;
		double A = 0.0;
		double B = 0.0;
		int num_transfer = 0;
		for (i = 0; i < off_info->num_mapped_vars; i++) {
			omp_data_map_info_t *map_info = &off_info->data_map_info[i];
			if (map_info->map_type == OMP_DATA_MAP_FROM || map_info->map_type == OMP_DATA_MAP_TO) {
				num_transfer++;
			} else if (map_info->map_type == OMP_DATA_MAP_TOFROM) {
				num_transfer += 2;
			} else continue;

			omp_data_map_t *map = &map_info->maps[seqid];
			long map_size = map_info->sizeof_element;
			align_maps[i].align_dist = NULL;
			int num_aligned_dims = 0;
			int j;
			for (j = 0; j < map_info->num_dims; j++) { /* process each dimension */
				omp_dist_info_t *andist_info = &map_info->dist_info[j];
				omp_dist_t *map_dist = &map->map_dist[j];
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
				map_size = map_size * 2;
			}
			if (align_maps[i].align_dist != NULL) { /* we have an alignment */
				A += map_size;
			} else {
				B += map_size;
			}
		}

		A = A / (dev->bandwidth * 10.0e6); /* bandwidth in MB/s */
		B = B / (dev->bandwidth * 10.0e6);
		A += off_info->per_iteration_profile.num_fp_operations /
			 (dev->total_real_flopss * 10.0e9); /* FLOPs is in GFLOPs/s */
		B += num_transfer * (dev->latency * 10.0e-6); /* latency in us */
		/* here T = n*A+B --> n = T/A - B/A. We have then Ar = 1/A, and Br = -B/A*/

		double Ar = 1.0 / A;
		double Br = 0.0 - B / A;
		/* broadcast this to other device */
		off->Ar = Ar;
		off->Br = Br;
		/* sync so to make sure all received this info */
		pthread_barrier_wait(&off_info->inter_dev_barrier);
		/* solve the linear system */
		double allArs = 0.0;
		double allBrs = 0.0;
		for (i = 0; i < ndev; i++) {
			omp_offloading_t *anoff = &off_info->offloadings[i];
			allArs += anoff->Ar;
			allBrs += anoff->Br;
		}
		//printf("allArs: %f, allBrs: %f\n", allArs, allBrs);
		double T0 = (full_length - allBrs) / allArs; /* the predicted execution time by all the devices of the loop */
		/* now compute the offset and length for AUTO dist policy */
		float ratios[ndev];
		offset = 0;
		for (i = 0; i < ndev; i++) {
			omp_offloading_t *anoff = &off_info->offloadings[i];
			length = (long) (T0 * anoff->Ar + anoff->Br + 0.5);
			if (length <= 0) length = 0;
			if (length >= full_length) length = full_length;
			if (i == ndev - 1 && offset + length != full_length) { /* fix rounding error */
				length = full_length - offset;
			}
			ratios[i] = (float)length / (float) full_length;
			offset += length;
		}

		omp_dist_with_cutoff(ndev, ratios, cutoff_ratio, full_length, seqid, start, myoffset, mylength);
	//	printf("MODEL_AUTO: LINEAR_MODEL_2: Dev %d: offset: %d, length: %d of total length: %d, predicted exe time: %f\n", dev->id, *myoffset, *mylength, full_length, T0);
#if 0
        /* do the alignment map for arrays */
            for (i=0; i<off_info->num_mapped_vars; i++) {
                omp_dist_t * align_dist = align_maps[i].align_dist;
                if (align_dist != NULL) {
                    align_dist->length = *mylength;
                    align_dist->offset = *myoffset;
                }
            }
#endif
	}

#if defined (OMP_BREAKDOWN_TIMING)
	if (events != NULL)
		omp_event_record_stop(&events[runtime_dist_modeling_index]);
#endif
}

/**
 * distribute according to the profiled performance
 */
static void omp_dist_profile_auto(omp_dist_info_t *dist_info, long start, long full_length, int seqid, long *myoffset,
								  long *mylength, float *myratio, float *profile_performance, float cutoff_ratio) {
	int i;

	omp_offloading_info_t * off_info = (omp_offloading_info_t*)dist_info->target;
	int ndev = off_info->top->nnodes;
	*profile_performance = 0.0;
	*myratio = 0.0;
	*mylength = 0;
	float ratios[ndev];
	for (i =0; i <ndev; i++) {
		omp_offloading_t *anoff = &off_info->offloadings[i];
		double Ti = anoff->runtime_profile_elapsed;
		float ratio = 0.0;
		if (anoff->last_total == 0 || anoff->runtime_profile_elapsed <= 0.0) {
			/* this one will not participating */
		} else {
			int j;
			for (j = 0; j < ndev; j++) {
				anoff = &off_info->offloadings[j];
				if (anoff->last_total == 0 || anoff->runtime_profile_elapsed <= 0.0)
					continue; /* this one will not participating */
				ratio += Ti / anoff->runtime_profile_elapsed;
			}
			ratio = 1.0 / ratio;
		}
		ratios[i] = ratio;
		//	printf("Dev %d: Ti: %f, ratio: %f\n", dev->id, Ti, ratio);
	}

	omp_dist_with_cutoff(ndev, ratios, cutoff_ratio, full_length, seqid, start, myoffset, mylength);
}

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

		dist->offset = dist_info->offset + offset;
		dist->length = length;
	} else if (dist_info->policy == OMP_DIST_POLICY_FULL) { /* full rang dist_info */
		dist->length = full_length;
		dist->offset = dist_info->offset;
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
		dist->offset = alignee_dist->offset + dist_info->offset;
//		printf("aligned dist on dev %d: offset: %d, length: %d\n", seqid, alignee_dist->offset, alignee_dist->length);
	} else if (dist_info->policy == OMP_DIST_POLICY_MODEL_1_AUTO) {
		omp_dist_model(OMP_DIST_POLICY_MODEL_1_AUTO, dist_info, dist_info->start, full_length, top, coords, seqid,
					   &dist->offset, &dist->length, LOOP_DIST_CUTOFF_RATIO);
	} else if (dist_info->policy == OMP_DIST_POLICY_MODEL_2_AUTO) {
		omp_dist_model(OMP_DIST_POLICY_MODEL_2_AUTO, dist_info, dist_info->start, full_length, top, coords, seqid,
					   &dist->offset, &dist->length, LOOP_DIST_CUTOFF_RATIO);
	} else if (dist_info->policy == OMP_DIST_POLICY_SCHED_STATIC_RATIO) { /* simply distribute according to a user specified ratio */
		/* TODO: we do not have yet a user interface or runtime API for user to use this feature */
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
				dist->offset = offset + dist_info->offset;
				dist->length = length;
				break;
			}
			offset += length;
		}
		//printf("STATIC_RATIO: Dev %d: offset: %d, length: %d of total length: %d with ratio: %f\n", dev->id, dist->offset, dist->length, full_length);
	} else if (dist_info->policy == OMP_DIST_POLICY_SCHED_DYNAMIC) { /* only for loop */
		/* get next chunk */
		if (dist_info->chunk_size < 0)
			/* the percentage of total iterations, not the left-over */
			length = (0 - dist_info->chunk_size) * (dist_info->length) / 100;
		else
			length = dist_info->chunk_size;
		do {
			offset = dist_info->start;
			int new_start = offset + length;
			if (new_start > dist_info->end) {
				length = dist_info->end - offset;
				if (length <= 0) {
					length = 0;
					break;
				}
				new_start = dist_info->end;
			}
			if (__homp_cas(&dist_info->start, offset, new_start)) break;
		} while (1);
		dist->offset = offset;
		dist->length = length;
		//printf("SCHED_DYNAMIC: Dev %d: offset: %d, length: %d of total length: %d\n", dev->id, offset, length, full_length);
	} else if (dist_info->policy == OMP_DIST_POLICY_SCHED_GUIDED) { /* only for loop */
		full_length = dist_info->end - dist_info->offset;
		if (dist_info->chunk_size < 0)
			/* the percentage of left-over, not the total */
			length = (0 - dist_info->chunk_size) * full_length / 100;
		else
			length = dist_info->chunk_size;

		length = length / (dist->counter + 1);
		long min = full_length * 0.01; /* the min length */

		if (length < min) length = min; /* minimum 10 iterations per chunk */
		if (length > dist_info->length) {
			printf("Posibblly because of data race\n");
			abort();
		}
		do {
			offset = dist_info->start;
			int new_start = offset + length;
			if (new_start > dist_info->end) {
				length = dist_info->end - offset;
				if (length <= 0) {
					length = 0;
					break;
				}
				new_start = dist_info->end;
			}
			if (__homp_cas(&dist_info->start, offset, new_start)) break;
		} while (1);
		dist->offset = offset;
		dist->length = length;
		//printf("SCHED_GUIDE: Dev %d: offset: %d, length: %d of total length: %d\n", dev->id, offset, length, full_length);
	} else if (dist_info->policy == OMP_DIST_POLICY_SCHED_FEEDBACK) { /* only for loop */
		if (dist_info->chunk_size < 0)
			/* the percentage of left-over, not the total */
			length = (0 - dist_info->chunk_size) * (dist_info->end - dist_info->start) / 100;
		else
			length = dist_info->chunk_size/(dist->counter + 1);

		if (length < 100) length = 100; /* minimum 10 iterations per chunk */
		if (length > dist_info->length) {
			printf("Posibblly because of data race\n");
			abort();
		}
		do {
			offset = dist_info->start;
			int new_start = offset + length;
			if (new_start > dist_info->end) {
				length = dist_info->end - offset;
				if (length <= 0) {
					length = 0;
					break;
				}
				new_start = dist_info->end;
			}
			if (__homp_cas(&dist_info->start, offset, new_start)) break;
		} while (1);
		dist->offset = offset;
		dist->length = length;
//		printf("SCHED_GUIDE: Dev %d: offset: %d, length: %d of total length: %d\n", dev->id, offset, length, full_length);
	} else if (dist_info->policy == OMP_DIST_POLICY_SCHED_PROFILE_AUTO) { /* only for loop */
        /* two dists are needed for this policy and we use counter for that */
		if (dist->counter == 0) { /* SCHEDULE_STATIC policy */
			if (dist_info->chunk_size < 0)
				/* the percentage of total iterations, not the left-over */
				length = (0 - dist_info->chunk_size) * (dist_info->length) / 100;
			else
				length = dist_info->chunk_size;
			do {
				offset = dist_info->start;
				long new_start = offset + length;
				if (new_start > dist_info->end) {
					length = dist_info->end - offset;
					if (length <= 0) {
						length = 0;
						break;
					}
					new_start = dist_info->end;
				}
				if (__homp_cas(&dist_info->start, offset, new_start)) break;
			} while (1);
			dist->offset = offset;
			dist->length = length;
			//printf("SCHED_PROFILE_AUTO, PROFILE: Dev %d: offset: %d, length: %d of total length: %d\n", dev->id, dist->offset, dist->length, full_length);
		} else { /* similar to ratio policy */
			float profile_performance, myratio;

			omp_dist_profile_auto(dist_info, dist_info->start, full_length, seqid, &dist->offset, &dist->length,
								  &myratio, &profile_performance, LOOP_DIST_CUTOFF_RATIO);
			//printf("SCHED_PROFILE_AUTO, AUTO:    Dev %d: offset: %d, length: %d (%.2f%%) of total length: %d based on my profiling performance: %f\n", dev->id, dist->offset, dist->length, myratio*100.0, full_length, profile_performance);
			//dist->counter = 0; /* reset dist->counter for the future call of the same offloading */
		}
	} else if (dist_info->policy == OMP_DIST_POLICY_MODEL_PROFILE_AUTO) {
		if (dist_info->chunk_size < 0)
			/* the percentage of total iterations, not the left-over */
			length = (0 - dist_info->chunk_size) * (dist_info->length) / 100;
		else
			length = dist_info->chunk_size;
		if (dist->counter == 0) {
			omp_dist_model(OMP_DIST_POLICY_MODEL_2_AUTO, dist_info, dist_info->start, length, top, coords, seqid,
						   &dist->offset, &dist->length, LOOP_DIST_CUTOFF_RATIO);
			//printf("MODEL_PROFILE_AUTO, PROFILE: Dev %d: offset: %d, length: %d of total length: %d\n", dev->id, dist->offset, dist->length, length);
		} else {
			float profile_performance, myratio;

			full_length = full_length - length;
			offset = dist_info->start + length; /* here we did not update the dist_info->start. If to do it, we need a single thread to do that in the previous stage */

			omp_dist_profile_auto(dist_info, offset, full_length, seqid, &dist->offset, &dist->length, &myratio,
								  &profile_performance, LOOP_DIST_CUTOFF_RATIO);
			//printf("MODEL_PROFILE_AUTO, AUTO:    Dev %d: offset: %d, length: %d (%.2f%%) of total length: %d based on my profiling performance: %f\n", dev->id, dist->offset, dist->length, myratio*100.0, full_length, profile_performance);
			//dist->counter = 0; /* reset dist->counter for the future call of the same offloading */
		}
	} else {
		fprintf(stderr, "other dist_info type %d is not yet supported\n", dist_info->policy);
		abort();
		exit(1);
	}
	if (dist->length > 0) dist->counter++;
}

/* return the total amount of distribution */
long omp_loop_iteration_dist(omp_offloading_t *off) {
	omp_offloading_info_t *off_info = off->off_info;
	omp_grid_topology_t * top = off_info->top;

	int coords[top->ndims];
	omp_topology_get_coords(top, off->devseqid, top->ndims, coords);
	int i;

	//printf("omp_dist_call: %d\n", __LINE__);
	long total = 1;
	for (i = 0; i < off_info->loop_depth; i++) { /* process each dimension */
		omp_dist_info_t *dist_info = &off_info->loop_dist_info[i];
		omp_dist_t * dist = &off->loop_dist[i];

		if (dist_info->redist_needed || dist->counter < 1) {
			dist->info = dist_info;
			omp_dist(dist_info, dist, top, coords, off->devseqid, i); /* map_dist will increment the dist->counter */
		}
		dist->total_length += dist->length;
		dist->acc_total_length += dist->length;

		total *= off->loop_dist[i].length;
	}

	off->loop_dist_done = 1;
	off->last_total = total;
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
		omp_dist_t * dist = &map->map_dist[i];

		if (dist_info->redist_needed || dist->counter < 1) {
			omp_dist(dist_info, dist, top, coords, seqid, i); /* map_dist will increment the dist->counter */
			dist->total_length += dist->length;
		}
		dist->acc_total_length += dist->length;

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

		if (map_info->num_halo_dims) { omp_data_map_halo_region_info_t *halo = &map_info->halo_info[i];
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

	map->total_map_size += map_size;
	map->total_map_wextra_size += map_wextra_size;

	if (total > 0) map->dist_counter++;
	map->access_level = OMP_DATA_MAP_ACCESS_LEVEL_DIST;
	return total;
}

long omp_loop_get_range(omp_offloading_t *off, int loop_level, long *offset, long *length) {
	if (!off->loop_dist_done)
		omp_loop_iteration_dist(off);
	*offset = 0;
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

/*void omp_yed_graph_plot(omp_data_map_t * map)
{
omp_data_map_info_t * info = map->info;
	char *color,a;
       FILE *fp;

        fp = fopen("/home/aditi/homp/benchmarks/axpy_clean/graph.graphml", "a+");
    

 	if(strcmp(info->symbol,"x") == 0)
       	color= "#FF9900";
 	else
       	color="#99CCFF";

 fprintf(fp," <node id=\"%s-%d\">\n",info->symbol,map->dev->id);
 fprintf(fp,"<data key=\"d6\">\n  <y:GenericNode configuration=\"ShinyPlateNode3\"> \n <y:Geometry height=\"75.0\" width=\"190.0\" x=\"659.0\" y=\"233.0\"/>\n");
 fprintf(fp,"<y:Fill color=\"%s\" transparent=\"false\"/> \n <y:BorderStyle hasColor=\"false\"  type=\"line\" width=\"1.0\"/> \n",color);
 fprintf(fp," <y:NodeLabel alignment=\"center\" autoSizePolicy=\"content\" fontFamily=\"Dialog\" fontSize=\"12\" fontStyle=\"plain\" hasBackgroundColor=\"false\" hasLineColor=\"false\" height=\"17.96875\" horizontalTextPosition=\"center\" iconTextGap=\"4\" modelName=\"custom\" textColor=\"#000000\" verticalTextPosition=\"bottom\" visible=\"true\" width=\"70.171875\" x=\"42.9140625\" y=\"28.015625\"> (%s) %s[%d:%d]<y:LabelModel>\n",omp_get_device_typename(map->dev),info->symbol,map->map_dist[0].offset, map->map_dist[0].total_length);
//THREAD %d map->dev->id,
 fprintf(fp," <y:SmartNodeLabelModel distance=\"4.0\"/> \n </y:LabelModel> \n <y:ModelParameter> \n <y:SmartNodeLabelModelParameter labelRatioX=\"0.0\" labelRatioY=\"0.0\" nodeRatioX=\"0.0\" nodeRatioY=\"0.0\" offsetX=\"0.0\" offsetY=\"0.0\" upX=\"0.0\" upY=\"-1.0\"/>\n </y:ModelParameter>\n </y:NodeLabel> \n </y:GenericNode>\n </data>\n  </node>");

if(strcmp(info->symbol,"x") == 0)
{
fprintf(fp," <node id=\"(%d)\">\n",map->dev->id);
 fprintf(fp,"<data key=\"d5\"/> \n <data key=\"d6\">\n   <y:ShapeNode> \n <y:Geometry height=\"75.0\" width=\"190.0\" x=\"659.0\" y=\"233.0\"/>\n");
 fprintf(fp,"<y:Fill color=\"#FF9999\" transparent=\"false\"/> \n <y:BorderStyle hasColor=\"false\"  type=\"line\" width=\"1.0\"/> \n");
 fprintf(fp," <y:NodeLabel alignment=\"center\" autoSizePolicy=\"content\" fontFamily=\"Dialog\" fontSize=\"12\" fontStyle=\"plain\" hasBackgroundColor=\"false\" hasLineColor=\"false\" height=\"17.96875\" horizontalTextPosition=\"center\" iconTextGap=\"4\" modelName=\"custom\" textColor=\"#000000\" verticalTextPosition=\"bottom\" visible=\"true\" width=\"70.171875\" x=\"42.9140625\" y=\"28.015625\">THREAD %d <y:LabelModel>\n",map->dev->id);

 fprintf(fp," <y:SmartNodeLabelModel distance=\"4.0\"/> \n </y:LabelModel> \n <y:ModelParameter> \n <y:SmartNodeLabelModelParameter labelRatioX=\"0.0\" labelRatioY=\"0.0\" nodeRatioX=\"0.0\" nodeRatioY=\"0.0\" offsetX=\"0.0\" offsetY=\"0.0\" upX=\"0.0\" upY=\"-1.0\"/>\n </y:ModelParameter>\n </y:NodeLabel> \n  <y:Shape type=\"ellipse\"/>\n </y:ShapeNode> \n </data>\n  </node>\n");
  
}

fclose(fp);

}*/

void omp_print_map_info(omp_data_map_info_t * info) {
	int i;
 FILE *fp;

	
        fp = fopen("/home/aditi/homp/benchmarks/axpy_clean/graph.graphml", "a+");
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
		omp_offloading_info_t *off1;
		printf("\t%d, ", i);
		omp_print_data_map(&info->maps[i]);
		//omp_yed_graph_plot(&info->maps[i]);
		//omp_yed_edge(&info->maps[i]);	
		                
	}
}

/*omp_yed_edge(omp_data_map_t * map)
{
	FILE *fp;
	omp_data_map_info_t * info = map->info;
        fp = fopen("/home/aditi/homp/benchmarks/axpy_clean/graph.graphml", "a+");
	if(strcmp(info->symbol,"x") == 0)
	{
		fprintf(fp,"\n<edge id=\"%s:%d\" source=\"(%d)\" target=\"%s-%d\">\n",info->symbol,map->dev->id,map->dev->id,info->symbol,map->dev->id);
          	fprintf(fp,"<data key=\"d10\"/>\n");
 	     	fprintf(fp,"<data key=\"d11\">\n");
       		fprintf(fp,"<y:PolyLineEdge>\n");
          	fprintf(fp,"<y:Path sx=\"0.0\" sy=\"-56.5\" tx=\"0.0\" ty=\"37.5\"/>\n");
          	fprintf(fp,"<y:LineStyle color=\"#000000\" type=\"line\" width=\"1.0\"/>\n");
          	fprintf(fp,"<y:Arrows source=\"none\" target=\"standard\"/>\n");
          	fprintf(fp,"<y:BendStyle smoothed=\"false\"/>\n");
        	fprintf(fp,"</y:PolyLineEdge>\n");
      		fprintf(fp,"</data>\n");
    		fprintf(fp,"</edge>\n");
	}
	if(strcmp(info->symbol,"y") == 0)
	{
		fprintf(fp,"\n<edge id=\"%s:%d\" source=\"x-%d\" target=\"%s-%d\">\n",info->symbol,map->dev->id,map->dev->id,info->symbol,map->dev->id);
          	fprintf(fp,"<data key=\"d10\"/>\n");
 	     	fprintf(fp,"<data key=\"d11\">\n");
       		fprintf(fp,"<y:PolyLineEdge>\n");
          	fprintf(fp,"<y:Path sx=\"0.0\" sy=\"-56.5\" tx=\"0.0\" ty=\"37.5\"/>\n");
          	fprintf(fp,"<y:LineStyle color=\"#000000\" type=\"line\" width=\"1.0\"/>\n");
          	fprintf(fp,"<y:Arrows source=\"none\" target=\"standard\"/>\n");
          	fprintf(fp,"<y:BendStyle smoothed=\"false\"/>\n");
        	fprintf(fp,"</y:PolyLineEdge>\n");
      		fprintf(fp,"</data>\n");
    		fprintf(fp,"</edge>\n");
	}
 
fclose(fp);

}

*/

void omp_print_data_map(omp_data_map_t * map) {
	omp_data_map_info_t * info = map->info;
	char * mem = "COPY";
        
 
	if (map->map_type == OMP_DATA_MAP_SHARED) {
		mem = "SHARED";
	}
	printf("dev %d(%s), %s, ", map->dev->id, omp_get_device_typename(map->dev), mem);
     
 	int soe = info->sizeof_element;
	int i;
	//for (i=0; i<info->num_dims;i++) printf("[%d:%d]", map->map_dist[i].offset, map->map_dist[i].offset+map->map_dist[i].length-1);
	for (i=0; i<info->num_dims;i++) {
        printf("[%d:%d]", map->map_dist[i].offset, map->map_dist[i].total_length);
 	}

	printf(", size: %d, size wextra: %d (accumulated %d times)\n",map->total_map_size, map->total_map_wextra_size, map->dist_counter);
	printf("\t\tsrc ptr: %X, src wextra ptr: %X, dev ptr: %X, dev wextra ptr: %X (last)\n", map->map_source_ptr, map->map_source_wextra_ptr, map->map_dev_ptr, map->map_dev_wextra_ptr);
 
	if (info->num_halo_dims) {
		//printf("\t\thalo memory:\n");
		omp_data_map_halo_region_mem_t * all_halo_mems = map->halo_mem;
		for (i=0; i<info->num_dims; i++) {
			omp_data_map_halo_region_mem_t * halo_mem = &all_halo_mems[i];
			printf("\t\t%d-d halo, L_IN: %X[%d], L_OUT: %X[%d], R_OUT: %X[%d], R_IN: %X[%d]", i,
				   halo_mem->left_in_ptr, halo_mem->left_in_size/soe, halo_mem->left_out_ptr, halo_mem->left_out_size/soe,
				   halo_mem->right_out_ptr, halo_mem->right_out_size/soe, halo_mem->right_in_ptr, halo_mem->right_in_size/soe);
            
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

void omp_topology_pretty_print(omp_grid_topology_t * top, char * buffer) {
	int i, offset;
	for (i=0; i<top->nnodes; i++) {
		int devid = top->idmap[i];
		omp_device_t * dev = &omp_devices[devid];
		offset = strlen(buffer);
		sprintf(buffer + offset, "%d(%s:%d)-", devid, omp_get_device_typename(dev), dev->sysid);
	}
	offset = strlen(buffer);
	buffer[offset-1] = '\0'; /* remove the last - */
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


void omp_yed_start1()
{
      FILE *fp1;

        fp1 = fopen("/home/aditi/homp/graph.graphml", "w");
        
        fprintf(fp1,"<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n");
fprintf(fp1,"<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\" xmlns:java=\"http://www.yworks.com/xml/yfiles-common/1.0/java\" xmlns:sys=\"http://www.yworks.com/xml/yfiles-common/markup/primitives/2.0\" xmlns:x=\"http://www.yworks.com/xml/yfiles-common/markup/2.0\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xmlns:y=\"http://www.yworks.com/xml/graphml\" xmlns:yed=\"http://www.yworks.com/xml/yed/3\" xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns http://www.yworks.com/xml/schema/graphml/1.1/ygraphml.xsd\">\n");
 fprintf(fp1," <!--Created by yEd 3.16.2.1-->");
 fprintf(fp1," <key for=\"port\" id=\"d0\" yfiles.type=\"portgraphics\"/>\n");
 fprintf(fp1," <key for=\"port\" id=\"d1\" yfiles.type=\"portgeometry\"/>\n");
 fprintf(fp1," <key for=\"port\" id=\"d2\" yfiles.type=\"portuserdata\"/>\n");
 fprintf(fp1," <key attr.name=\"color\" attr.type=\"string\" for=\"node\" id=\"d3\">\n");
 fprintf(fp1,"<default><![CDATA[yellow]]></default>\n");
 fprintf(fp1," </key>\n");
 fprintf(fp1,"<key attr.name=\"url\" attr.type=\"string\" for=\"node\" id=\"d4\"/>\n");
 fprintf(fp1,"<key attr.name=\"description\" attr.type=\"string\" for=\"node\" id=\"d5\"/>\n");
 fprintf(fp1," <key for=\"node\" id=\"d6\" yfiles.type=\"nodegraphics\"/>\n");
 fprintf(fp1,"<key for=\"graphml\" id=\"d7\" yfiles.type=\"resources\"/>\n");
 fprintf(fp1,"<key attr.name=\"weight\" attr.type=\"double\" for=\"edge\" id=\"d8\"/>\n");
 fprintf(fp1,"<key attr.name=\"url\" attr.type=\"string\" for=\"edge\" id=\"d9\"/>\n");
 fprintf(fp1,"<key attr.name=\"description\" attr.type=\"string\" for=\"edge\" id=\"d10\"/>\n");
 fprintf(fp1,"<key for=\"edge\" id=\"d11\" yfiles.type=\"edgegraphics\"/>\n");
 fprintf(fp1,"<graph edgedefault=\"directed\" id=\"G\">\n");
 fclose(fp1);      

}
void omp_yed_end1()
{
FILE *fp1;

        fp1 = fopen("/home/aditi/homp/graph.graphml", "a+");
        fprintf(fp1,"\n</graph>\n"); 
	fprintf(fp1,"<data key=\"d7\">\n");
    	fprintf(fp1,"<y:Resources/>\n");
  	fprintf(fp1,"</data>\n");
	fprintf(fp1,"</graphml>\n");
	fclose(fp1);

}
void omp_yed_xml(omp_offloading_info_t *info,int num)
{
	
	FILE *fp1;
int i,j;
 fp1 = fopen("/home/aditi/homp/graph.graphml", "a+");
//-------------------------------------

sprintf(path_buf, "/proc/self/maps");				//to get the address range
freopen("/home/aditi/homp/mapinfo.txt", "w", stdout);
	 f = fopen(path_buf, "r");      // open the specified file
    if (f != NULL)
    {
        int c;

        while ((c = fgetc(f)) != EOF)     // read character from file until EOF
        {
            putchar(c); 
	 	                  // output character
        }
        fclose(f);
    }
fclose (stdout);
freopen("/dev/tty", "w", stdout);
  fprintf(fp1,"<data key=\"d0\"/>");
  fprintf(fp1,"\n<node id=\"n0\" yfiles.foldertype=\"group\">\n");
  fprintf(fp1,"<data key=\"d4\"/>\n");
  fprintf(fp1,"<data key=\"d5\"/>\n");
  fprintf(fp1,"<data key=\"d6\">\n");
  fprintf(fp1,"<y:ProxyAutoBoundsNode>\n");
  fprintf(fp1,"<y:Realizers active=\"0\">\n");
  fprintf(fp1,"<y:GroupNode>\n");
  fprintf(fp1,"<y:Geometry height=\"573\" width=\"639.0\" x=\"70.0\" y=\"-75.96093749999999\"/>\n");
  fprintf(fp1,"<y:Fill color=\"#F5F5F5\" transparent=\"false\"/>\n");
  fprintf(fp1,"<y:BorderStyle color=\"#000000\" type=\"dashed\" width=\"1.0\"/>\n");
  fprintf(fp1,"<y:NodeLabel alignment=\"right\" autoSizePolicy=\"node_width\" backgroundColor=\"#EBEBEB\" borderDistance=\"0.0\" fontFamily=\"Dialog\"\n"); 
  fprintf(fp1,"fontSize=\"15\" fontStyle=\"plain\" hasLineColor=\"false\" height=\"21.4609375\" horizontalTextPosition=\"center\" iconTextGap=\"4\"\n"); 
  fprintf(fp1,"modelName=\"internal\" modelPosition=\"t\" textColor=\"#000000\" verticalTextPosition=\"bottom\" visible=\"true\"\n"); 
  fprintf(fp1,"width=\"639.0\" x=\"0.0\" y=\"0.0\">cpu 0</y:NodeLabel>\n");
  fprintf(fp1,"<y:Shape type=\"roundrectangle\"/>\n");
  fprintf(fp1,"<y:State closed=\"false\" closedHeight=\"50.0\" closedWidth=\"50.0\" innerGraphDisplayEnabled=\"false\"/>\n");
  fprintf(fp1,"<y:Insets bottom=\"15\" bottomF=\"15.0\" left=\"15\" leftF=\"15.0\" right=\"15\" rightF=\"15.0\" top=\"15\" topF=\"15.0\"/>\n");
  fprintf(fp1,"<y:BorderInsets bottom=\"726\" bottomF=\"726.0\" left=\"0\" leftF=\"0.0\" right=\"1416\" rightF=\"416.064453125\" top=\"0\" topF=\"0.0\"/>\n");
  fprintf(fp1,"</y:GroupNode>\n");
  fprintf(fp1,"<y:GroupNode>\n");
  fprintf(fp1,"<y:Geometry height=\"50.0\" width=\"50.0\" x=\"0.0\" y=\"60.0\"/>\n");
  fprintf(fp1,"<y:Fill color=\"#F5F5F5\" transparent=\"false\"/>\n");
  fprintf(fp1,"<y:BorderStyle color=\"#000000\" type=\"dashed\" width=\"1.0\"/>\n");
  fprintf(fp1,"<y:NodeLabel alignment=\"right\" autoSizePolicy=\"node_width\" backgroundColor=\"#EBEBEB\" borderDistance=\"0.0\"\n"); 
  fprintf(fp1,"fontFamily=\"Dialog\" fontSize=\"15\" fontStyle=\"plain\" hasLineColor=\"false\" height=\"21.4609375\" horizontalTextPosition=\"center\"\n"); 
  fprintf(fp1,"iconTextGap=\"4\" modelName=\"internal\" modelPosition=\"t\" textColor=\"#000000\" verticalTextPosition=\"bottom\" visible=\"true\"\n"); 
  fprintf(fp1,"width=\"65.201171875\" x=\"-7.6005859375\" y=\"0.0\">Folder 1</y:NodeLabel>\n");
  fprintf(fp1,"<y:Shape type=\"roundrectangle\"/>\n");
  fprintf(fp1,"<y:State closed=\"true\" closedHeight=\"50.0\" closedWidth=\"50.0\" innerGraphDisplayEnabled=\"false\"/>\n");
  fprintf(fp1,"<y:Insets bottom=\"5\" bottomF=\"5.0\" left=\"5\" leftF=\"5.0\" right=\"5\" rightF=\"5.0\" top=\"5\" topF=\"5.0\"/>\n");
  fprintf(fp1,"<y:BorderInsets bottom=\"0\" bottomF=\"0.0\" left=\"0\" leftF=\"0.0\" right=\"0\" rightF=\"0.0\" top=\"0\" topF=\"0.0\"/>\n");
  fprintf(fp1,"</y:GroupNode>\n");
  fprintf(fp1,"</y:Realizers>\n");
  fprintf(fp1,"</y:ProxyAutoBoundsNode>\n");
  fprintf(fp1,"</data>\n");
  fprintf(fp1,"<graph edgedefault=\"directed\" id=\"n0:\">\n");
i=0;
system("cut -d' ' -f1 /home/aditi/homp/mapinfo.txt > /home/aditi/homp/mapinfo1.txt");		//to select just the first column of the mapinfo.txt file
FILE* file = fopen("/home/aditi/homp/mapinfo1.txt", "r"); /* should check the result */
    char line[256];
    while (fgets(line, sizeof(line), file)) {
               printf("%s", line); 	
	fprintf(fp1," \n<node id= \"n0::n%d\">\n",i);
 fprintf(fp1,"<data key=\"d6\">\n  <y:GenericNode configuration=\"ShinyPlateNode3\"> \n <y:Geometry height=\"75.0\" width=\"190.0\" x=\"659.0\" y=\"233.0\"/>\n");
 fprintf(fp1,"<y:Fill color=\"#99CC00\" transparent=\"false\"/> \n <y:BorderStyle hasColor=\"false\"  type=\"line\" width=\"1.0\"/> \n");
 fprintf(fp1," <y:NodeLabel alignment=\"center\" autoSizePolicy=\"content\" fontFamily=\"Dialog\" fontSize=\"12\" fontStyle=\"plain\" hasBackgroundColor=\"false\" hasLineColor=\"false\" height=\"17.96875\" horizontalTextPosition=\"center\" iconTextGap=\"4\" modelName=\"custom\" textColor=\"#000000\" verticalTextPosition=\"bottom\" visible=\"true\" width=\"70.171875\" x=\"42.9140625\" y=\"28.015625\"> %s <y:LabelModel>\n" ,line );
 fprintf(fp1," <y:SmartNodeLabelModel distance=\"4.0\"/> \n </y:LabelModel> \n <y:ModelParameter> \n <y:SmartNodeLabelModelParameter labelRatioX=\"0.0\" labelRatioY=\"0.0\" nodeRatioX=\"0.0\" nodeRatioY=\"0.0\" offsetX=\"0.0\" offsetY=\"0.0\" upX=\"0.0\" upY=\"-1.0\"/>\n </y:ModelParameter>\n </y:NodeLabel> \n </y:GenericNode>\n </data>\n </node>"); 
i++;
    }
fprintf(fp1,"\n</graph>");
fprintf(fp1,"\n</node>\n");
     fclose(file);

//-------------------------------------
   for (i=0; i<info->top->nnodes; i++)
   	{
	omp_offloading_t *off=&info->offloadings[i]; 
	
	fprintf(fp1,"<data key=\"d0\"/>\n<node id=\"(%d)\">\n<data key=\"d5\"/>\n<data key=\"d6\">\n<y:ShapeNode>\n<y:Geometry height=\"88.0\" width=\"161.0\" x=\"555.0\" y=\"195.0\"/>\n",off->dev->id);
        fprintf(fp1,"<y:Fill color=\"#FFCC00\" transparent=\"false\"/><y:BorderStyle color=\"#000000\" raised=\"false\" type=\"line\" width=\"1.0\"/>\n");
        fprintf(fp1,"<y:NodeLabel alignment=\"center\" autoSizePolicy=\"content\" fontFamily=\"Dialog\" fontSize=\"12\" fontStyle=\"plain\" hasBackgroundColor=\"false\" hasLineColor=\"false\" height=\"17.96875\" horizontalTextPosition=\"center\" iconTextGap=\"4\" modelName=\"custom\" textColor=\"#000000\" verticalTextPosition=\"bottom\" visible=\"true\" width=\"55.046875\" x=\"52.9765625\" y=\"35.015625\">THREAD %d <y:LabelModel>\n",off->dev->id);
            fprintf(fp1,"<y:SmartNodeLabelModel distance=\"4.0\"/>\n</y:LabelModel>\n<y:ModelParameter>\n");
            fprintf(fp1,"<y:SmartNodeLabelModelParameter labelRatioX=\"0.0\" labelRatioY=\"0.0\" nodeRatioX=\"0.0\" nodeRatioY=\"0.0\" offsetX=\"0.0\" offsetY=\"0.0\" upX=\"0.0\" upY=\"-1.0\"/>\n");
            fprintf(fp1," </y:ModelParameter>\n </y:NodeLabel>\n<y:Shape type=\"ellipse\"/>\n</y:ShapeNode>\n</data>\n </node>\n");              
	 }
 	fclose(fp1);

	omp_offloading_t *off=&info->offloadings;
	omp_data_map_info_t *info1 = info->data_map_info;
	omp_data_map_t *map = info1->maps;

for(j=0;j <= off->num_maps+1;j++)
{
omp_data_map_info_t *info1 = &info->data_map_info[j];
for (i=0; i<info1->off_info->top->nnodes; i++)   				
		{
	omp_offloading_t *off=&info->offloadings[i]; 
	omp_yed_node(&info1->maps[i],off);  
	
	        }
	
}
}
//------------------

int read_pagemap(char * path_buf, unsigned long virt_addr){
   printf("Big endian? %d\n", is_bigendian());
   f = fopen(path_buf, "rb");
   if(!f){
      printf("Error! Cannot open %s\n", path_buf);
      return -1;
   }
   
   //Shifting by virt-addr-offset number of bytes
   //and multiplying by the size of an address (the size of an entry in pagemap file)
   file_offset = virt_addr / getpagesize() * PAGEMAP_ENTRY;
   printf("Vaddr: 0x%lx, Page_size: %d, Entry_size: %d\n", virt_addr, getpagesize(), PAGEMAP_ENTRY);
   printf("Reading %s at 0x%llx\n", path_buf, (unsigned long long) file_offset);
   status = fseek(f, file_offset, SEEK_SET);
   if(status){
      perror("Failed to do fseek!");
      return -1;
   }
   errno = 0;
   read_val = 0;
   unsigned char c_buf[PAGEMAP_ENTRY];
   for(i=0; i < PAGEMAP_ENTRY; i++){
      c = getc(f);
      if(c==EOF){
         printf("\nReached end of the file\n");
         return 0;
      }
      if(is_bigendian())
           c_buf[i] = c;
      else
           c_buf[PAGEMAP_ENTRY - i - 1] = c;
      printf("[%d]0x%x ", i, c);
   }
   for(i=0; i < PAGEMAP_ENTRY; i++){
      //printf("%d ",c_buf[i]);
      read_val = (read_val << 8) + c_buf[i];
   }
   printf("\n");
   printf("Result: 0x%llx\n", (unsigned long long) read_val);
   //if(GET_BIT(read_val, 63))
   if(GET_BIT(read_val, 63))
      printf("PFN: 0x%llx\n",(unsigned long long) GET_PFN(read_val));
   else
      printf("Page not present\n");
   if(GET_BIT(read_val, 62))
      printf("Page swapped\n");
   fclose(f);
   return 0;
}

void getData(char buff[])		//read csv file
{
   char *token = strtok(buff,",");
   int counter=0;
 
   while( token != NULL ) 
   {
 counter++; 
printf( " %s\n",token);
      token = strtok(NULL,",");
   }	  
}

void omp_yed_node(omp_data_map_t * map, omp_offloading_t *off)			/* to display nodes in graphml */
{
		FILE *fp1;
	omp_data_map_info_t * info1 = map->info;
        fp1 = fopen("/home/aditi/homp/graph.graphml", "a+");
	
	fprintf(fp1," \n<node id=\"%s-%d\">\n",info1->symbol,off->dev->id);
 fprintf(fp1,"<data key=\"d6\">\n  <y:GenericNode configuration=\"ShinyPlateNode3\"> \n <y:Geometry height=\"75.0\" width=\"190.0\" x=\"659.0\" y=\"233.0\"/>\n");
 fprintf(fp1,"<y:Fill color=\"#99CCFF\" transparent=\"false\"/> \n <y:BorderStyle hasColor=\"false\"  type=\"line\" width=\"1.0\"/> \n");
 fprintf(fp1," <y:NodeLabel alignment=\"center\" autoSizePolicy=\"content\" fontFamily=\"Dialog\" fontSize=\"12\" fontStyle=\"plain\" hasBackgroundColor=\"false\" hasLineColor=\"false\" height=\"17.96875\" horizontalTextPosition=\"center\" iconTextGap=\"4\" modelName=\"custom\" textColor=\"#000000\" verticalTextPosition=\"bottom\" visible=\"true\" width=\"70.171875\" x=\"42.9140625\" y=\"28.015625\"> (%s) %s[%d:%d]<y:LabelModel>\n",omp_get_device_typename(map->dev),info1->symbol,map->map_dist[0].offset, map->map_dist[0].total_length);
 fprintf(fp1," <y:SmartNodeLabelModel distance=\"4.0\"/> \n </y:LabelModel> \n <y:ModelParameter> \n <y:SmartNodeLabelModelParameter labelRatioX=\"0.0\" labelRatioY=\"0.0\" nodeRatioX=\"0.0\" nodeRatioY=\"0.0\" offsetX=\"0.0\" offsetY=\"0.0\" upX=\"0.0\" upY=\"-1.0\"/>\n </y:ModelParameter>\n </y:NodeLabel> \n </y:GenericNode>\n </data>\n </node>"); 


/*---------------edge printing for each node with the thread node------------------*/

		fprintf(fp1,"\n<edge id=\"%s:%d\" source=\"(%d)\" target=\"%s-%d\">\n",info1->symbol,off->dev->id,map->dev->id,info1->symbol,off->dev->id);
          	fprintf(fp1,"<data key=\"d10\"/>\n");
 	     	fprintf(fp1,"<data key=\"d11\">\n");
       		fprintf(fp1,"<y:PolyLineEdge>\n");
          	fprintf(fp1,"<y:Path sx=\"0.0\" sy=\"-56.5\" tx=\"0.0\" ty=\"37.5\"/>\n");
          	fprintf(fp1,"<y:LineStyle color=\"#000000\" type=\"line\" width=\"1.0\"/>\n");
          	fprintf(fp1,"<y:Arrows source=\"none\" target=\"standard\"/>\n");
          	fprintf(fp1,"<y:BendStyle smoothed=\"false\"/>\n");
        	fprintf(fp1,"</y:PolyLineEdge>\n");
      		fprintf(fp1,"</data>\n");
    		fprintf(fp1,"</edge>\n");

/*-----------print source pointer node------------*/

fprintf(fp1," \n<node id=\"s%s-%d\">\n",info1->symbol,off->dev->id);
 fprintf(fp1,"<data key=\"d6\">\n  <y:GenericNode configuration=\"ShinyPlateNode3\"> \n <y:Geometry height=\"75.0\" width=\"190.0\" x=\"659.0\" y=\"233.0\"/>\n");
 fprintf(fp1,"<y:Fill color=\"#FF9999\" transparent=\"false\"/> \n <y:BorderStyle hasColor=\"false\"  type=\"line\" width=\"1.0\"/> \n");
 fprintf(fp1," <y:NodeLabel alignment=\"center\" autoSizePolicy=\"content\" fontFamily=\"Dialog\" fontSize=\"12\" fontStyle=\"plain\" hasBackgroundColor=\"false\" hasLineColor=\"false\" height=\"17.96875\" horizontalTextPosition=\"center\" iconTextGap=\"4\" modelName=\"custom\" textColor=\"#000000\" verticalTextPosition=\"bottom\" visible=\"true\" width=\"70.171875\" x=\"42.9140625\" y=\"28.015625\"> Map Source Ptr (%s): %X <y:LabelModel>\n",info1->symbol,map->map_source_ptr);
 fprintf(fp1," <y:SmartNodeLabelModel distance=\"4.0\"/> \n </y:LabelModel> \n <y:ModelParameter> \n <y:SmartNodeLabelModelParameter labelRatioX=\"0.0\" labelRatioY=\"0.0\" nodeRatioX=\"0.0\" nodeRatioY=\"0.0\" offsetX=\"0.0\" offsetY=\"0.0\" upX=\"0.0\" upY=\"-1.0\"/>\n </y:ModelParameter>\n </y:NodeLabel> \n </y:GenericNode>\n </data>\n </node>"); 

//-------------------

int numa_node = -1;
get_mempolicy(&numa_node, NULL, 0, (void*)map->map_source_ptr, MPOL_F_NODE | MPOL_F_ADDR);
printf("\n%d",numa_node);

//-------------------
/*
void * ptr_to_check = map->map_source_ptr;
 //here you should align ptr_to_check to page boundary 
 int status[1];
 int ret_code;
 status[0]=-1;
 ret_code=move_pages(0 , 1, &ptr_to_check, NULL, status, 0);
 printf("Memory at %p is at %d node (retcode %d)\n", ptr_to_check, status[0], ret_code);
*/
//-------------------

/*uintptr_t vaddr = map->map_source_ptr; 
 FILE *pagemap, *numa;
    intptr_t paddr = 0;
    int offset = (vaddr / sysconf(_SC_PAGESIZE)) * sizeof(uint64_t);
    uint64_t e;

    // https://www.kernel.org/doc/Documentation/vm/pagemap.txt
    if ((pagemap = fopen("/proc/self/pagemap", "r"))) {
        if (lseek(fileno(pagemap), offset, SEEK_SET) == offset) {
            if (fread(&e, sizeof(uint64_t), 1, pagemap)) {
                if (e & (1ULL << 63)) { // page present ?
                    paddr = e & ((1ULL << 54) - 1); // pfn mask
                    paddr = paddr * sysconf(_SC_PAGESIZE);
                    // add offset within page
                    paddr = paddr | (vaddr & (sysconf(_SC_PAGESIZE) - 1));
                }   
            }   
        }   
        fclose(pagemap);
    }
numa = fopen("/proc/self/numa_maps", "r");

printf("--------------%ld", paddr); */
/*
//----------------------------
char buffer[12];

   
   if(!memcmp("self" ,"self",sizeof("self"))){
      sprintf(path_buf, "/proc/self/pagemap");
      pid = -1;
   }
   else{
         pid = strtol("self" ,&end, 10);
         if (end == "self"  || *end != '\0' || pid<=0){ 
            printf("PID must be a positive number or 'self'\n");
            return -1;
            }
       }
   virt_addr = strtol(map->map_source_ptr, NULL, 16);
   if(pid!=-1)
      sprintf(path_buf, "/proc/%u/pagemap", pid);
   
   read_pagemap(path_buf, virt_addr);

 */     
/*---------------edge printing for each source pointer node with the array node------------------*/

		fprintf(fp1,"\n<edge id=\"s%s:%d\" source=\"%s-%d\" target=\"s%s-%d\">\n",info1->symbol,off->dev->id,info1->symbol,map->dev->id,info1->symbol,off->dev->id);
          	fprintf(fp1,"<data key=\"d10\"/>\n");
 	     	fprintf(fp1,"<data key=\"d11\">\n");
       		fprintf(fp1,"<y:PolyLineEdge>\n");
          	fprintf(fp1,"<y:Path sx=\"0.0\" sy=\"-56.5\" tx=\"0.0\" ty=\"37.5\"/>\n");
          	fprintf(fp1,"<y:LineStyle color=\"#000000\" type=\"line\" width=\"1.0\"/>\n");
          	fprintf(fp1,"<y:Arrows source=\"none\" target=\"standard\"/>\n");
          	fprintf(fp1,"<y:BendStyle smoothed=\"false\"/>\n");
        	fprintf(fp1,"</y:PolyLineEdge>\n");
      		fprintf(fp1,"</data>\n");
    		fprintf(fp1,"</edge>\n");
       
/*-----------print device pointer node------------*/

fprintf(fp1," \n<node id=\"d%s-%d\">\n",info1->symbol,off->dev->id);
 fprintf(fp1,"<data key=\"d6\">\n  <y:GenericNode configuration=\"ShinyPlateNode3\"> \n <y:Geometry height=\"75.0\" width=\"190.0\" x=\"659.0\" y=\"233.0\"/>\n");
 fprintf(fp1,"<y:Fill color=\"#00CCCC\" transparent=\"false\"/> \n <y:BorderStyle hasColor=\"false\"  type=\"line\" width=\"1.0\"/> \n");
 fprintf(fp1," <y:NodeLabel alignment=\"center\" autoSizePolicy=\"content\" fontFamily=\"Dialog\" fontSize=\"12\" fontStyle=\"plain\" hasBackgroundColor=\"false\" hasLineColor=\"false\" height=\"17.96875\" horizontalTextPosition=\"center\" iconTextGap=\"4\" modelName=\"custom\" textColor=\"#000000\" verticalTextPosition=\"bottom\" visible=\"true\" width=\"70.171875\" x=\"42.9140625\" y=\"28.015625\"> Map Device Ptr (%s): %X <y:LabelModel>\n",info1->symbol,map->map_dev_ptr);
 fprintf(fp1," <y:SmartNodeLabelModel distance=\"4.0\"/> \n </y:LabelModel> \n <y:ModelParameter> \n <y:SmartNodeLabelModelParameter labelRatioX=\"0.0\" labelRatioY=\"0.0\" nodeRatioX=\"0.0\" nodeRatioY=\"0.0\" offsetX=\"0.0\" offsetY=\"0.0\" upX=\"0.0\" upY=\"-1.0\"/>\n </y:ModelParameter>\n </y:NodeLabel> \n </y:GenericNode>\n </data>\n </node>"); 
      
/*---------------edge printing for each device pointer node with the array node------------------*/

		fprintf(fp1,"\n<edge id=\"d%s:%d\" source=\"%s-%d\" target=\"d%s-%d\">\n",info1->symbol,off->dev->id,info1->symbol,map->dev->id,info1->symbol,off->dev->id);
          	fprintf(fp1,"<data key=\"d10\"/>\n");
 	     	fprintf(fp1,"<data key=\"d11\">\n");
       		fprintf(fp1,"<y:PolyLineEdge>\n");
          	fprintf(fp1,"<y:Path sx=\"0.0\" sy=\"-56.5\" tx=\"0.0\" ty=\"37.5\"/>\n");
          	fprintf(fp1,"<y:LineStyle color=\"#000000\" type=\"line\" width=\"1.0\"/>\n");
          	fprintf(fp1,"<y:Arrows source=\"none\" target=\"standard\"/>\n");
          	fprintf(fp1,"<y:BendStyle smoothed=\"false\"/>\n");
        	fprintf(fp1,"</y:PolyLineEdge>\n");
      		fprintf(fp1,"</data>\n");
    		fprintf(fp1,"</edge>\n");
fclose(fp1);	


}

void omp_yed_for_loop(omp_offloading_info_t *info1)
{
	FILE *fp1,*fp2;
	int i;
 	long start,offset;
    	long len;
 	fp1 = fopen("/home/aditi/homp/graph.graphml", "a+");
fp2 = fopen("/home/aditi/homp/benchmarks/axpy_clean/test2.graphml", "w+");
//printf("CPU: %d\n", sched_getcpu());

	for (i=0; i<info1->top->nnodes; i++)   				
		{
	omp_offloading_t *off= &info1->offloadings[i];
	omp_dist_t *loopdist = &off->loop_dist[i];	
	offset = omp_loop_get_range(off, 0, &start, &len);
 fprintf(fp1," \n<node id=\"%lu\">\n",off->devseqid);
 fprintf(fp1,"<data key=\"d6\">\n  <y:GenericNode configuration=\"ShinyPlateNode3\"> \n <y:Geometry height=\"75.0\" width=\"190.0\" x=\"659.0\" y=\"233.0\"/>\n");
 fprintf(fp1,"<y:Fill color=\"#99CCFF\" transparent=\"false\"/> \n <y:BorderStyle hasColor=\"false\"  type=\"line\" width=\"1.0\"/> \n");
 fprintf(fp1," <y:NodeLabel alignment=\"center\" autoSizePolicy=\"content\" fontFamily=\"Dialog\" fontSize=\"12\" fontStyle=\"plain\" hasBackgroundColor=\"false\" hasLineColor=\"false\" height=\"17.96875\" horizontalTextPosition=\"center\" iconTextGap=\"4\" modelName=\"custom\" textColor=\"#000000\" verticalTextPosition=\"bottom\" visible=\"true\" width=\"70.171875\" x=\"42.9140625\" y=\"28.015625\">for loop : %lu : %lu (Total: %lu) <y:LabelModel>\n",offset, offset+off->last_total,off->last_total);
 fprintf(fp1," <y:SmartNodeLabelModel distance=\"4.0\"/> \n </y:LabelModel> \n <y:ModelParameter> \n <y:SmartNodeLabelModelParameter labelRatioX=\"0.0\" labelRatioY=\"0.0\" nodeRatioX=\"0.0\" nodeRatioY=\"0.0\" offsetX=\"0.0\" offsetY=\"0.0\" upX=\"0.0\" upY=\"-1.0\"/>\n </y:ModelParameter>\n </y:NodeLabel> \n </y:GenericNode>\n </data>\n </node>");              
	
/*---------------edge printing for each for loop node with the thread node------------------*/

		fprintf(fp1,"\n<edge id=\"%lu\" source=\"(%d)\" target=\"%lu\">\n",off->devseqid,off->dev->id,off->devseqid);
          	fprintf(fp1,"<data key=\"d10\"/>\n");
 	     	fprintf(fp1,"<data key=\"d11\">\n");
       		fprintf(fp1,"<y:PolyLineEdge>\n");
          	fprintf(fp1,"<y:Path sx=\"0.0\" sy=\"-56.5\" tx=\"0.0\" ty=\"37.5\"/>\n");
          	fprintf(fp1,"<y:LineStyle color=\"#000000\" type=\"line\" width=\"1.0\"/>\n");
          	fprintf(fp1,"<y:Arrows source=\"none\" target=\"standard\"/>\n");
          	fprintf(fp1,"<y:BendStyle smoothed=\"false\"/>\n");
        	fprintf(fp1,"</y:PolyLineEdge>\n");
      		fprintf(fp1,"</data>\n");
    		fprintf(fp1,"</edge>\n");

	
	long *length = off->loop_dist[i].length;
	
	   
	//fprintf(fp2,"\n----%lu----%lu --%d",offset,length,info1->loop_depth);

	}

  	
	fclose(fp1);

}



void omp_offloading_info_report_profile(omp_offloading_info_t *info, int num) {

	int i, j;
	//omp_offloading_t *off = &info->offloadings[0];
#if defined(PROFILE_PLOT)
	char plotscript_filename[128];
	omp_yed_start1();	/* to print the start of graphml file   */    
	omp_yed_xml(info,num);   /* to print the mid-content of graphml file   */
	omp_yed_for_loop(info);
	omp_yed_end1();		/* to print the end of graphml file   */

//------------------------------------------
//int pid1 = getpid();
//printf("---%d----",pid1);
/*
 sprintf(path_buf, "/proc/%d/numa_maps",getpid());
	FILE *f = fopen(path_buf, "r");      // open the specified file
    if (f != NULL)
    {
        int c;

        while ((c = fgetc(f)) != EOF)     // read character from file until EOF
        {
            putchar(c);                   // output character
        }
        fclose(f);
    }
	
printf("\n----------------------------\n");
//-------------------------------------------
*/
//sprintf(path_buf, "/proc/self/smaps");
//printf("\n----------------------------\n");
//sprintf(path_buf, "/proc/self/status");
//sprintf(path_buf, "/proc/self/mem");



/*
//------------------------------------------
//sprintf(path_buf, "/proc/kpagecount");


sprintf(path_buf, "/proc/meminfo");
	 f = fopen(path_buf, "r");      // open the specified file
    if (f != NULL)
    {
        unsigned long c;

        while ((c = fgetc(f)) != EOF)     // read character from file until EOF
        {
            putchar(c);                   // output character
        }
        fclose(f);
    }
	
//-------------------------------------------
char buf[100];
sprintf(buf, "numastat -c -m -n -p %d", getpid());
int status = system(buf);
*/
//----------------------------
		
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

	int count = num; //info->count * num;
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
				omp_event_print_elapsed(ev, info->start_time, &start_time, &elapsed, ev->count<count?ev->count:count);
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

	long full_length = info->loop_dist_info[0].length; // - info->loop_dist_info[0].offset;

	/* write the report to a CSV file */
	char report_csv_filename[256];
	char dist_policy_str[128];
	char targets_str[128]; targets_str[0] = '\0';
	omp_topology_pretty_print(info->top, targets_str);
	for (i=0; i<num_allowed_dist_policies; i++) {
		if (omp_dist_policy_args[i].type == LOOP_DIST_POLICY)
			break;
	}
	if (LOOP_DIST_CHUNK_SIZE < 0)
		sprintf(dist_policy_str, "%s,%.2f%%,%.2f%%", omp_dist_policy_args[i].shortname, 0 - LOOP_DIST_CHUNK_SIZE, LOOP_DIST_CUTOFF_RATIO);
	else sprintf(dist_policy_str, "%s,%.2f,%.2f%%", omp_dist_policy_args[i].shortname, LOOP_DIST_CHUNK_SIZE, LOOP_DIST_CUTOFF_RATIO);

	sprintf(report_csv_filename, "%s-%d-%s.csv\0", info->name, full_length, dist_policy_str);
	FILE * report_csv_file = fopen(report_csv_filename, "a+");
	char time_buff[100];
	time_t now = time (0);
	strftime (time_buff, 100, "%Y-%m-%d %H:%M:%S.000", localtime (&now));
	fprintf(report_csv_file, "\"%s size: %d on %d devs(%s), %s policy, %s\"\n", info->name, full_length, info->top->nnodes, targets_str,
			dist_policy_str, time_buff);
	for (i=0; i<omp_num_devices; i++) {
		omp_device_t * dev = &omp_devices[i];
		int devid = dev->id;
		int devsysid = dev->sysid;
		char * type = omp_get_device_typename(dev);
		fprintf(report_csv_file, ",\"%s:%d(sysid:%d)\"", type, devid, devsysid);
	}
	fprintf(report_csv_file, ",%d DEVs ACCU, %% ACCU", info->top->nnodes);
	fprintf(report_csv_file, "\n");
	int lastdevid = 0;
	double acc_total = 0.0;
	double acc_time[misc_event_index_start];
	for (i=0; i<misc_event_index_start; i++) {
		int j;
		lastdevid = 0;
		double acc_temp = 0.0;
		for (j=0; j<info->top->nnodes; j++) {
			omp_offloading_t *off = &info->offloadings[j];
			int devid = off->dev->id;
			omp_event_t * ev = &off->events[i];
			if (ev->event_name == NULL) continue;
			int thiscount = ev->count<count?ev->count:count;
			double time_ms = thiscount > 0 ? omp_event_get_elapsed(ev)/thiscount : 0;
			acc_temp += time_ms;
			if (j == 0) fprintf(report_csv_file, "%s(%d)", ev->event_name, thiscount);
			while(lastdevid <= devid) {
				fprintf(report_csv_file, ",\t");
				lastdevid++;
			}
			fprintf(report_csv_file, "%.2f", time_ms);
		}
		acc_time[i] = acc_temp;
		fprintf(report_csv_file, ",%.2f", acc_temp);
		if (i == total_event_index) acc_total = acc_temp;
		fprintf(report_csv_file, ",%.2f%%", 100.0 * acc_temp / acc_total);
		fprintf(report_csv_file, "\n");
	}

	lastdevid = 0;
	for (j=0; j<info->top->nnodes; j++) {
		omp_offloading_t *off = &info->offloadings[j];

		int devid = off->dev->id;
		if (j == 0) fprintf(report_csv_file, "DIST(%d)", full_length);
		while(lastdevid <= devid) {
			fprintf(report_csv_file, ",\t");
			lastdevid++;
		}
		fprintf(report_csv_file, "%d", off->loop_dist[0].acc_total_length/count);
	}
	fprintf(report_csv_file, "\n");
	lastdevid = 0;
	for (j=0; j<info->top->nnodes; j++) {
		omp_offloading_t *off = &info->offloadings[j];

		int devid = off->dev->id;
		if (j == 0) fprintf(report_csv_file, "DIST(\%)");
		while(lastdevid <= devid) {
			fprintf(report_csv_file, ",\t");
			lastdevid++;
		}
//		printf("%d:%d,", off->loop_dist[0].length, length);
		float percentage = 100*((float)off->loop_dist[0].acc_total_length/((float) count * full_length));
		fprintf(report_csv_file, "%.1f%%", percentage);
	}
	fprintf(report_csv_file, "\n");
	lastdevid = 0;
	for (j=0; j<info->top->nnodes; j++) {
		omp_offloading_t *off = &info->offloadings[j];
		int devid = off->dev->id;
		if (j == 0) fprintf(report_csv_file, "DEV GFLOPS");
		while(lastdevid <= devid) {
			fprintf(report_csv_file, ",\t");
			lastdevid++;
		}
		fprintf(report_csv_file, "%.1f", off->dev->total_real_flopss);
	}
	fprintf(report_csv_file, "\n");

	fprintf(report_csv_file, "-------------------------------------------------------------------------------------------------\n\n");
	fclose(report_csv_file);

	/* another form of the csv file, it is the transposed version of the previous one */
	char report_csv_transpose[256];
	sprintf(report_csv_transpose, "%s-%d-%ddevs(%s).csv\0", info->name, full_length, info->top->nnodes, targets_str);
	FILE * report_csv_transpose_file = fopen(report_csv_transpose, "a+");
	fprintf(report_csv_transpose_file, "\"%s, size: %d on %d devices, policy: %s, %s\"\n", info->name, full_length, info->top->nnodes,
			dist_policy_str, time_buff);

	//off = &info->offloadings[0];
	fprintf(report_csv_transpose_file, "DIST POLICY,Device");
	for (j=0; j<misc_event_index_start; j++) {
		omp_event_t * ev = &off->events[j];
		if (ev->event_name == NULL) continue;
		int thiscount = ev->count<count?ev->count:count;
		fprintf(report_csv_transpose_file, ",%s(%d)", ev->event_name, thiscount);
	}
	fprintf(report_csv_transpose_file, ", DIST(%d), DIST(%%)\n", count);

	for (i=0; i<info->top->nnodes;i++) {
		int j;
		omp_offloading_t *off = &info->offloadings[i];
		int devid = off->dev->id;
		int devsysid = off->dev->sysid;
		char * type = omp_get_device_typename(off->dev);

		fprintf(report_csv_transpose_file, ", %s dev %d sysid: %d %.1f GFLOPS", type, devid, devsysid, off->dev->total_real_flopss);
		for (j=0; j<misc_event_index_start; j++) {
			omp_event_t * ev = &off->events[j];
			if (ev->event_name == NULL) continue;
			int thiscount = ev->count<count?ev->count:count;
			double time_ms = thiscount > 0 ? omp_event_get_elapsed(ev)/thiscount : 0;
			fprintf(report_csv_transpose_file, ",%.2f", time_ms);
		}
		float percentage = 100*((float)off->loop_dist[0].acc_total_length/(float)(count * full_length));
		fprintf(report_csv_transpose_file, ",%d(%d),%.1f%%(%d)", off->loop_dist[0].acc_total_length/count, count, percentage, count);
		fprintf(report_csv_transpose_file, "\n");
	}

	fprintf(report_csv_transpose_file, "\"%s\", %d DEVs ACCU", dist_policy_str, info->top->nnodes);
	for (j=0; j<misc_event_index_start; j++) {
		fprintf(report_csv_transpose_file, ",%.2f(%.2f%%)", acc_time[j], 100.0*acc_time[j]/acc_total);
	}
	fprintf(report_csv_transpose_file, "\n\n");
	/* percentge only */
	fprintf(report_csv_transpose_file, "\"%s\", %d DEVs ACCU %%", dist_policy_str, info->top->nnodes);
	fprintf(report_csv_transpose_file, ",%.2f", acc_total);
	for (j=1; j<misc_event_index_start; j++) {
		fprintf(report_csv_transpose_file, ",%.2f%%", 100.0*acc_time[j]/acc_total);
	}

	fprintf(report_csv_transpose_file, "\n\n");
	fclose(report_csv_transpose_file);

#if defined(PROFILE_PLOT)
	fprintf(plotscript_file, "plot 0\n");
	fclose(plotscript_file);
#endif
 /*-------------------------------------------------------------------------------------------------------*/


}

void omp_event_print_profile_header() {
	printf("%*s    TOTAL     AVE(#Calls) Last Start Host/dev Measure\tDescription\n",
		   OMP_EVENT_NAME_LENGTH-1, "Name");
}

void omp_event_print_elapsed(omp_event_t *ev, double reference, double *start_time, double *elapsed, int count) {
	omp_event_record_method_t record_method = ev->record_method;
	if (record_method == OMP_EVENT_HOST_RECORD) {
		printf("%*s%10.2f%10.2f(%d)%10.2f\thost\t%s\n",
			   OMP_EVENT_NAME_LENGTH, ev->event_name, ev->elapsed_host, ev->elapsed_host/count, count, ev->start_time_host - reference, ev->event_description);
		*start_time = ev->start_time_host - reference;
		*elapsed = ev->elapsed_host;
	} else if (record_method == OMP_EVENT_DEV_RECORD) {
		printf("%*s%10.2f%10.2f(%d)%10.2f\tdev\t%s\n",
			   OMP_EVENT_NAME_LENGTH, ev->event_name, ev->elapsed_dev, ev->elapsed_dev/count, count, ev->start_time_dev - reference, ev->event_description);
		*start_time = ev->start_time_dev - reference;;
		*elapsed = ev->elapsed_dev;
	} else {
		printf("%*s%10.2f%10.2f(%d)%10.2f\thost\t%s\n",
			   OMP_EVENT_NAME_LENGTH, ev->event_name, ev->elapsed_host, ev->elapsed_host/count, count, ev->start_time_host - reference, ev->event_description);
		printf("%*s%10.2f%10.2f(%d)%10.2f\tdev\t%s\n",
			   OMP_EVENT_NAME_LENGTH, ev->event_name, ev->elapsed_dev, ev->elapsed_dev/count, count, ev->start_time_dev - reference, ev->event_description);
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
