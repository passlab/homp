#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/timeb.h>

#include "homp.h"

/**
 * dist_info->start is the only variable that are modified by multiple dev threads
 * we need to reset this to off_info->offset in order to rerun the same offloading_info;
 *
 */
static void omp_reset_dist_for_rerun(omp_offloading_info_t *off_info) {
	int i;
	omp_dist_info_t * dist_info;
	for (i=0; i<off_info->loop_depth; i++) {
		dist_info = &off_info->loop_dist_info[i];
		dist_info->start = dist_info->offset;
	}

	for (i=0; i<off_info->num_mapped_vars; i++) {
		omp_data_map_info_t * map_info = &off_info->data_map_info[i];
		int j;
		for (j=0; j<map_info->num_dims; j++) {
			dist_info = &map_info->dist_info[j];
			dist_info->start = dist_info->offset;
		}
	}
}

static inline void omp_off_init_before_run(omp_offloading_t * off) {
    off->loop_dist_done = 0;
    off->runtime_profile_elapsed = -1.0;
    off->last_total = 1; /* this serve as flag to see whether re-dist should be done or not */
    off->loop_dist[0].total_length = 0;
    off->loop_dist[1].total_length = 0;
    off->loop_dist[2].total_length = 0;
    off->loop_dist[0].counter = 0;
    off->loop_dist[1].counter = 0;
    off->loop_dist[2].counter = 0;
}

/**
 * notifying the helper threads to work on the offloading specified in off_info arg
 * It always start with copyto and may stops after copyto for target data
 * master is just the thread that will store
 */
void omp_offloading_start(omp_offloading_info_t *off_info) {
	omp_grid_topology_t * top = off_info->top;
    /* generate master trace file */

    if (!off_info->recurring) off_info->count = 0; /* if this is not recurring one, but need to rerun mulitple times for e.g. performance measurement, we reset the counter to 0 */
	off_info->start_time = read_timer_ms(); /* only for the first time */

	int i;
	for (i = 0; i < top->nnodes; i++) {
		omp_device_t * dev = &omp_devices[top->idmap[i]];
		if (dev->offload_request != NULL) {
			fprintf(stderr, "device %d is not ready for answering your request, offloading_start: %X. It is a bug so far\n", dev->id, off_info);
		}
		dev->offload_request = off_info;
		//printf("offloading to device: %d, %X\n", i, off_info);
		/* TODO: this is data race if multiple host threads try to offload to the same devices,
		 * FIX is to use cas operation to update this field
		 */
	}

	pthread_barrier_wait(&off_info->barrier);

    off_info->count++;
	off_info->compl_time = read_timer_ms();
    if (off_info->loop_redist_needed) omp_reset_dist_for_rerun(off_info); /* for the next sample run by reset the start in dist_info */

#if defined (OMP_BREAKDOWN_TIMING)
	pthread_barrier_wait(&off_info->barrier); /* this one make sure the profiling is collected */
#endif
}

long secondary_offload_cycle(omp_offloading_info_t *off_info, omp_offloading_t *off, omp_event_t *events, int seqid) {
	int i;

#if defined (OMP_BREAKDOWN_TIMING)
	omp_event_record_start(&events[map_dist_alloc_event_index]);
#endif
    long total = 0;
	if (off_info->loop_redist_needed) total = omp_loop_iteration_dist(off);
	if (total == 0) return 0;
	//case OMP_OFFLOADING_MAPMEM:
	/* init data map and dev memory allocation */
	/***************** for each mapped variable that has to and tofrom, if it has region mapped to this __ndev_i__ id, we need code here *******************************/
	for (i = 0; i < off->num_maps; i++) {
		int inherited = 1;
		omp_data_map_t *map = omp_map_offcache_iterator(off, i, &inherited);
		if (inherited) continue;

		if (map->info->remap_needed){
			omp_data_map_dist(map, seqid);
			if (map->access_level != OMP_DATA_MAP_ACCESS_LEVEL_MALLOC) omp_map_malloc(map, off);
		}
		//omp_print_data_map(map);
	}

#if defined (OMP_BREAKDOWN_TIMING)
	omp_event_record_stop(&events[map_dist_alloc_event_index]);
#endif

	if (off_info->type == OMP_OFFLOADING_STANDALONE_DATA_EXCHANGE) goto data_exchange;

//	case OMP_OFFLOADING_COPYTO:
	{
#if defined (OMP_BREAKDOWN_TIMING)
		if (off_info->num_mapped_vars > 0)
			omp_event_record_start(&events[acc_mapto_event_index]);
#endif
		for (i = 0; i < off->num_maps; i++) {
			int inherited;
			omp_data_map_t *map = omp_map_offcache_iterator(off, i, &inherited);
			if (inherited) continue;
			omp_data_map_info_t *map_info = map->info;
			if (map_info->map_direction == OMP_DATA_MAP_TO || map_info->map_direction == OMP_DATA_MAP_TOFROM) {
				if (map_info->remap_needed) {
#if defined (OMP_BREAKDOWN_TIMING)
					omp_event_t *ev = &events[misc_event_index_start+i];
//				if (ev->event_name == NULL) omp_event_set_attribute(ev, off->stream, "MAPTO_", "Time for mapto data movement for array %s", map_info->symbol);
//				printf("Dev: %d: MAPTO_%s, %d\n", off->dev->id, map_info->symbol, misc_event_index-1);
					omp_event_record_start(ev);
#endif
					omp_map_mapto_async(map, off->stream);
					//omp_map_memcpy_to_async((void*)map->map_dev_ptr, dev, (void*)map->map_buffer, map->map_size, off->stream); /* memcpy from host to device */
#if defined (OMP_BREAKDOWN_TIMING)
					omp_event_record_stop(ev);
#endif
				}
			}
		}
#if defined (OMP_BREAKDOWN_TIMING)
		if (off_info->num_mapped_vars > 0)
			omp_event_record_stop(&events[acc_mapto_event_index]);
#endif
	}

//	case OMP_OFFLOADING_KERNEL:
	{
#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_start(&events[acc_kernel_exe_event_index]);
#endif
		/* launching the kernel */
		void *args = off_info->args;
		void (*kernel_launcher)(omp_offloading_t *, void *) = off_info->kernel_launcher;
		if (args == NULL) args = off->args;
		if (kernel_launcher == NULL) kernel_launcher = off->kernel_launcher;
		kernel_launcher(off, args);
		off->loop_dist_done = 0; /* reset for the next dist if there is */
#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_stop(&events[acc_kernel_exe_event_index]);
#endif
	}

//	case OMP_OFFLOADING_EXCHANGE:
	data_exchange:;
	/* for data exchange, either a standalone or an appended exchange */
	if (off_info->halo_x_info != NULL) {
		omp_stream_sync(off->stream);/* make sure previous operation are complete, should NOT be timed for exchange */
#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_start(&events[acc_ex_pre_barrier_event_index]);
#endif
		pthread_barrier_wait(
				&off_info->inter_dev_barrier); /* make sure everybody is completed so we can exchange now */
#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_stop(&events[acc_ex_pre_barrier_event_index]);
#endif

#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_start(&events[acc_ex_event_index]);
#endif
		for (i = 0; i < off_info->num_maps_halo_x; i++) {
			omp_data_map_halo_exchange_info_t *x_halos = &off_info->halo_x_info[i];
			omp_data_map_info_t *map_info = x_halos->map_info;
			//int devseqid = omp_grid_topology_get_seqid(map_info->top, dev->id);

			omp_data_map_t *map = &map_info->maps[seqid];
			omp_halo_region_pull(map, x_halos->x_dim, x_halos->x_direction);
		}
#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_stop(&events[acc_ex_event_index]);
		omp_event_record_start(&events[acc_ex_post_barrier_event_index]);
#endif
//		dev->offload_request = NULL; /* release this dev */
		pthread_barrier_wait(&off_info->inter_dev_barrier);
		//printf("dev: %d (seqid: %d) holo region pull\n", dev->id, seqid);

#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_stop(&events[acc_ex_post_barrier_event_index]);
#endif
//		if (off_info->type == OMP_OFFLOADING_STANDALONE_DATA_EXCHANGE) goto omp_offloading_sync_cleanup;
	}

//	case OMP_OFFLOADING_COPYFROM:
	{
		omp_offloading_copyfrom:;
#if defined (OMP_BREAKDOWN_TIMING)
		if (off_info->num_mapped_vars > 0)
			omp_event_record_start(&events[acc_mapfrom_event_index]);
#endif
		/* copy back results */
		for (i = 0; i < off->num_maps; i++) {
			int inherited;
			omp_data_map_t *map = omp_map_offcache_iterator(off, i, &inherited);
			if (inherited) continue;
			omp_data_map_info_t *map_info = map->info;
			if (map_info->map_direction == OMP_DATA_MAP_FROM || map_info->map_direction == OMP_DATA_MAP_TOFROM) {
				if (map_info->remap_needed) {
#if defined (OMP_BREAKDOWN_TIMING)
					/* TODO bug here if this is reached from the above goto, since events is not available */
					omp_event_t *ev = &events[misc_event_index_start+i];
//				if (ev->event_name == NULL) omp_event_set_attribute(ev, off->stream, "MAPFROM_", "Time for mapfrom data movement for array %s", map_info->symbol);
					omp_event_record_start(ev);
#endif
					omp_map_mapfrom_async(map, off->stream);
					//omp_map_memcpy_from_async((void*)map->map_buffer, (void*)map->map_dev_ptr, dev, map->map_size, off->stream); /* memcpy from host to device */
#if defined (OMP_BREAKDOWN_TIMING)
					omp_event_record_stop(ev);
#endif
				}
			}
		}
#if defined (OMP_BREAKDOWN_TIMING)
		if (off_info->num_mapped_vars > 0)
			omp_event_record_stop(&events[acc_mapfrom_event_index]);
#endif
	}

	/* sync stream to wait for completion of the initial offloading */
	omp_stream_sync(off->stream); /*NOTE: we should NOT time this call as the event system already count in as previous async kernel or async memcpy */
#if defined (OMP_BREAKDOWN_TIMING)
	omp_event_record_start(&events[sync_cleanup_event_index]);
#endif
	for (i=0; i<off->num_maps; i++) {
		int inherited;
		omp_data_map_t *map = omp_map_offcache_iterator(off, i, &inherited);
		if (inherited) continue;
		if (map->info->remap_needed) omp_map_free(map, off);
	}

#if defined (OMP_BREAKDOWN_TIMING)
	omp_event_record_stop(&events[sync_cleanup_event_index]);
#endif
	return total;
}

/**
 * return the acc_time of this call only, and this time will be added upon the previous accu_time in the event
 */
double omp_accumulate_elapsed_ms(omp_event_t *events, int num_events) {
	int i;
	double accu_time = 0.0;
	omp_event_accumulate_elapsed_ms(&events[total_event_index], 0);
	omp_event_accumulate_elapsed_ms(&events[timing_init_event_index], 0);
	accu_time += omp_event_accumulate_elapsed_ms(&events[map_init_event_index], 0);
	accu_time += omp_event_accumulate_elapsed_ms(&events[map_dist_alloc_event_index], 0);
	omp_event_accumulate_elapsed_ms(&events[runtime_dist_modeling_index], 0);
	omp_event_accumulate_elapsed_ms(&events[sync_cleanup_event_index], 0);
	omp_event_accumulate_elapsed_ms(&events[barrier_wait_event_index], 0);
	omp_event_accumulate_elapsed_ms(&events[profiling_barrier_wait_event_index], 0);
	accu_time += omp_event_accumulate_elapsed_ms(&events[acc_mapto_event_index], 0);
	accu_time += omp_event_accumulate_elapsed_ms(&events[acc_kernel_exe_event_index], 0);
	accu_time += omp_event_accumulate_elapsed_ms(&events[acc_mapfrom_event_index], 0);
	accu_time += omp_event_accumulate_elapsed_ms(&events[acc_ex_pre_barrier_event_index], 0);
	accu_time += omp_event_accumulate_elapsed_ms(&events[acc_ex_event_index], 0);
	accu_time += omp_event_accumulate_elapsed_ms(&events[acc_ex_post_barrier_event_index], 0);
	for (i=misc_event_index_start; i<num_events; i++) {
		omp_event_accumulate_elapsed_ms(&events[i], 0);
	}
	omp_event_record_start(&events[total_event_accumulated_index]);
	omp_event_record_stop(&events[total_event_accumulated_index]);
	omp_event_accumulate_elapsed_ms(&events[total_event_accumulated_index], accu_time);
	return accu_time;
}

/* the num_mapped_vars * 2 +4 is the rough number of events needed */
/* the event (if mapto var is num_mapto, and mapfrom var is num_mapfrom (both including tofrom);
 * 0: The whole measured time from host side, measured from host
 * 1: The init time (stream, event, etc), this is the overhead for the breakdown timing, measured from host
 * 2: The time for map init, data dist, buffer allocation and data marshalling, measured from host
 * 3: The accumulated time for mapto datamovement, measured from dev
 * 4 - acc_kernel_exe_event_index-1: The time for each mapto datamovement, measured from dev (total num_mapto events)
 * acc_kernel_exe_event_index: kernel exe time
 * acc_kernel_exe_event_index+1: The accumulated time for mapfrom datamovement, measured from dev
 *     acc_kernel_exe_event_index+2 - xxxx: The time for each mapfrom datamovement, measured from dev (total num_mapfrom events)
 * xxxx: The time for cleanup resources (stream, event, data unmarshalling, etc), measured from host
 * xxxx: The time for barrier wait (for other kernel to complete), measured from host
 */
static int omp_init_off_stream_events(omp_offloading_t * off) {
	omp_device_t * dev = off->dev;
	int devid = dev->id;
	omp_offloading_info_t * off_info = off->off_info;
	int i;

	int num_events = off_info->num_mapped_vars * 2 + misc_event_index_start; /* the max posibble # of events to be used */
	omp_event_t * events = (omp_event_t *) malloc(sizeof(omp_event_t) * num_events); /**TODO: free this memory somewhere later */
	off->num_events = num_events;
	off->events = events;

	omp_event_init(&events[timing_init_event_index], dev, OMP_EVENT_HOST_RECORD, NULL, "INIT_0", "Time for initialization of stream and event", devid);
	omp_event_record_start(&events[timing_init_event_index]);
#if defined USING_PER_OFFLOAD_STREAM
	omp_stream_create(dev, &off->mystream);
    off->stream = &off->mystream;
#else
	off->stream = &dev->default_stream;
#endif
	omp_dev_stream_t *stream = off->stream;

	omp_event_init(&events[total_event_index], dev, OMP_EVENT_HOST_RECORD, NULL, "OFF_TOTAL", "Total offloading time (everything) on dev: %d", devid);
	omp_event_init(&events[total_event_accumulated_index], dev, OMP_EVENT_HOST_RECORD, NULL, "ACCU_TOTAL",
				   "Total ACCUMULATED time on dev: %d", devid);
	omp_event_init(&events[map_init_event_index], dev, OMP_EVENT_HOST_RECORD, NULL, "INIT_0.1",
				   "Time for init map data structure");
	omp_event_init(&events[map_dist_alloc_event_index], dev, OMP_EVENT_HOST_RECORD, NULL, "INIT_1",
				   "Time for data dist, memory allocation, and data marshalling");
	omp_event_init(&events[runtime_dist_modeling_index], dev, OMP_EVENT_HOST_RECORD, NULL, "MODELING",
				   "Runtime modeling cost");
	omp_event_init(&events[sync_cleanup_event_index], dev, OMP_EVENT_HOST_RECORD, NULL, "FINI_1",
				   "Time for dev sync and cleaning (event/stream/map, deallocation/unmarshalling)");
	omp_event_init(&events[barrier_wait_event_index], dev, OMP_EVENT_HOST_RECORD, NULL, "BAR_FINI_2",
				   "Time for barrier wait for other to complete");
	omp_event_init(&events[profiling_barrier_wait_event_index], dev, OMP_EVENT_HOST_RECORD, NULL, "PROF_BAR",
				   "Time for barrier wait to make sure profiling by all dev is done");
	omp_event_init(&events[acc_mapto_event_index], dev, OMP_EVENT_DEV_RECORD, stream, "ACC_MAPTO",
				   "Accumulated time for mapto data movement for all array");
	omp_event_init(&events[acc_kernel_exe_event_index], dev, OMP_EVENT_DEV_RECORD, stream, "KERN",
				   "Time for kernel (%s) execution", off_info->name);
	omp_event_init(&events[acc_mapfrom_event_index], dev, OMP_EVENT_DEV_RECORD, stream, "ACC_MAPFROM",
				   "Accumulated time for mapfrom data movement for all array");
	omp_event_init(&events[acc_ex_pre_barrier_event_index], dev, OMP_EVENT_HOST_RECORD, NULL, "PRE_BAR_X",
				   "Time for barrier sync before data exchange between devices");
	omp_event_init(&events[acc_ex_event_index], dev, OMP_EVENT_HOST_RECORD, NULL, "DATA_X",
				   "Time for data exchange between devices");
	omp_event_init(&events[acc_ex_post_barrier_event_index], dev, OMP_EVENT_HOST_RECORD, NULL, "POST_BAR_X",
				   "Time for barrier sync after data exchange between devices");

	for (i = misc_event_index_start; i < num_events; i++) {
		//printf("init misc event: %d by dev: %d\n", i, dev->id);
		omp_event_init(&events[i], dev, OMP_EVENT_DEV_RECORD, NULL, NULL, NULL, 0);
	}

	omp_event_record_stop(&events[timing_init_event_index]);
	return num_events;
}

static inline void omp_off_init_maps(omp_offloading_info_t * off_info, omp_offloading_t * off, int seqid) {
	int i;
	for (i = 0; i < off_info->num_mapped_vars; i++) {
		/* we handle inherited map here, by each helper thread, and we only update the off object (not off_info)*/
		omp_data_map_info_t *map_info = &off_info->data_map_info[i];

		int inherited = 1;
		omp_data_map_t *map = omp_map_offcache_iterator(off, i, &inherited);

		if (map == NULL) {
			map = omp_map_get_map_inheritance(off->dev, map_info->source_ptr);
			if (map == NULL) { /* here we basically ignore any map specification if it can inherit from ancestor (upper level nested target data) */
				map = &map_info->maps[seqid];
				inherited = 0;
			} else inherited = 1;
			omp_map_append_map_to_offcache(off, map, inherited);
			//omp_print_data_map(map);
		}
		if (!inherited) {
			if (map_info->remap_needed || off_info->count < 1) omp_data_map_init_map(map, map_info, off->dev);
		}
	}
}

static inline void omp_off_map_dist(omp_offloading_info_t * off_info, omp_offloading_t * off, int seqid) {
	int i;
	for (i = 0; i < off_info->num_mapped_vars; i++) {
		/* we handle inherited map here, by each helper thread, and we only update the off object (not off_info)*/
		int inherited = 1;
		omp_data_map_t *map = omp_map_offcache_iterator(off, i, &inherited);
		if (inherited) continue;
		if (map->info->remap_needed || off_info->count < 1) omp_data_map_dist(map, seqid); /* handle all unmapped variable */
		if (map->access_level != OMP_DATA_MAP_ACCESS_LEVEL_MALLOC) omp_map_malloc(map, off);
	}
}

static inline void omp_off_map_copyto(omp_offloading_t *off, omp_event_t *events) {
	int i;
	int misc_event_index = misc_event_index_start;
	for (i = 0; i < off->num_maps; i++) {
		int inherited;
		omp_data_map_t *map = omp_map_offcache_iterator(off, i, &inherited);
		if (inherited) continue;
		omp_data_map_info_t *map_info = map->info;
		if (map_info->map_direction == OMP_DATA_MAP_TO || map_info->map_direction == OMP_DATA_MAP_TOFROM) {
#if defined (OMP_BREAKDOWN_TIMING)
			omp_event_t * ev = &events[misc_event_index+i];
			if (ev->event_name == NULL) {
				//if (devid == 0)
				//printf("set misc event att: %d by dev: %d\n", misc_event_index, dev->id);
				omp_event_set_attribute(ev, off->stream, "MAPTO_", "Time for mapto data movement for array %s", map_info->symbol);
			}
			//if (devid == 0) printf("Dev: %d: MAPTO_%s, %d: off count: %d\n", dev->id, map_info->symbol, misc_event_index-1, off_info->count);
			omp_event_record_start(ev);
#endif
			//omp_print_data_map(map);
			omp_map_mapto_async(map, off->stream);
			//omp_map_memcpy_to_async((void*)map->map_dev_ptr, dev, (void*)map->map_buffer, map->map_size, off->stream); /* memcpy from host to device */
#if defined (OMP_BREAKDOWN_TIMING)
			omp_event_record_stop(ev);
#endif
		}
	}
}

static inline void omp_off_map_copyfrom(omp_offloading_t * off, omp_event_t * events) {
	int i;
	int	misc_event_index = misc_event_index_start;
	for (i = 0; i < off->num_maps; i++) {
		int inherited;
		omp_data_map_t *map = omp_map_offcache_iterator(off, i, &inherited);
		if (inherited) continue;
		omp_data_map_info_t *map_info = map->info;
		if (map_info->map_direction == OMP_DATA_MAP_FROM || map_info->map_direction == OMP_DATA_MAP_TOFROM) {
#if defined (OMP_BREAKDOWN_TIMING)
			/* TODO bug here if this is reached from the above goto, since events is not available */
			omp_event_t * ev = &events[misc_event_index+i];
			if (ev->event_name == NULL) omp_event_set_attribute(ev, off->stream, "MAPFROM_", "Time for mapfrom data movement for array %s", map_info->symbol);
			omp_event_record_start(ev);
#endif
			omp_map_mapfrom_async(map, off->stream);
			//omp_map_memcpy_from_async((void*)map->map_buffer, (void*)map->map_dev_ptr, dev, map->map_size, off->stream); /* memcpy from host to device */
#if defined (OMP_BREAKDOWN_TIMING)
			omp_event_record_stop(ev);
#endif
		}
	}
}

static inline void omp_off_map_free(omp_offloading_t * off) {
	int i;
	for (i=0; i<off->num_maps; i++) {
		int inherited;
		omp_data_map_t *map = omp_map_offcache_iterator(off, i, &inherited);
		if (inherited) continue;
		omp_map_free(map, off);
	}
}

static void omp_post_off_barrier(omp_offloading_t * off) {
	int num_events = off->num_events;
	omp_event_t *events = off->events;
	omp_offloading_info_t * off_info = off->off_info;
#if defined (OMP_BREAKDOWN_TIMING)
	omp_event_record_start(&events[barrier_wait_event_index]);
#endif
	off->dev->offload_request = NULL; /* release this dev */
	pthread_barrier_wait(&off_info->barrier);
#if defined (OMP_BREAKDOWN_TIMING)
	omp_event_record_stop(&events[barrier_wait_event_index]);
#endif

#if defined (OMP_BREAKDOWN_TIMING)
	omp_event_record_stop(&events[total_event_index]);
#endif

	/* print out the timing info */
#if defined (OMP_BREAKDOWN_TIMING)
	/* do timing accumulation if this is a recurring kernel */
	omp_accumulate_elapsed_ms(events, num_events);
	pthread_barrier_wait(&off_info->barrier);
#endif
}

/**
 *
 * off_info->type == OMP_OFFLOADING_DATA should be true when calling this func by the proxy thread
 *
 * we use off_info->count to check whether this call is for copyto or copyfrom
 * odd:  copyto
 * even: copyfrom
 */
void omp_offloading_data_copyto(omp_device_t * dev) {
	omp_offloading_info_t * off_info = dev->offload_request;
	omp_grid_topology_t * top = off_info->top;
	int seqid = omp_grid_topology_get_seqid(top, dev->id); /* we assume this is tiny for timing, so not included */
	omp_offloading_t * off = &off_info->offloadings[seqid];
	//printf("devid: %d --> seqid: %d in top: %X, off: %X, off_info: %X\n", dev->id, seqid, top, off, off_info);

#if defined (OMP_BREAKDOWN_TIMING)
	if (off->events == NULL) { /* the first time of recurring offloading or a non-recurring offloading */
		omp_init_off_stream_events(off);
        //off_info->count <= 1; /* assertation */
	}
    omp_off_init_before_run(off);
	omp_event_t * events = off->events;
	omp_event_record_start(&events[total_event_index]);
#endif

#if defined (OMP_BREAKDOWN_TIMING)
	omp_event_record_start(&events[map_init_event_index]);
#endif
	omp_off_init_maps(off_info, off, seqid);
#if defined (OMP_BREAKDOWN_TIMING)
	omp_event_record_stop(&events[map_init_event_index]);
#endif

#if defined (OMP_BREAKDOWN_TIMING)
	omp_event_record_start(&events[map_dist_alloc_event_index]);
#endif
	/* init data map and dev memory allocation */
	omp_off_map_dist(off_info, off, seqid);

#if defined (OMP_BREAKDOWN_TIMING)
	omp_event_record_stop(&events[map_dist_alloc_event_index]);
#endif

#if defined (OMP_BREAKDOWN_TIMING)
	if (off_info->num_mapped_vars > 0)
		omp_event_record_start(&events[acc_mapto_event_index]);
#endif
	omp_off_map_copyto(off, events);

#if defined (OMP_BREAKDOWN_TIMING)
	if (off_info->num_mapped_vars > 0)
		omp_event_record_stop(&events[acc_mapto_event_index]);
#endif

	dev->offload_stack_top++;
	dev->offload_stack[dev->offload_stack_top] = off;
	omp_stream_sync(off->stream); /*NOTE: we should NOT time this call as the event system already count in as previous async kernel or async memcpy */

	omp_post_off_barrier(off);
}

/* copyto should already happen for the same off_info
 */
void omp_offloading_data_copyfrom(omp_device_t * dev) {
	omp_offloading_info_t * off_info = dev->offload_request;
	omp_grid_topology_t * top = off_info->top;
	int seqid = omp_grid_topology_get_seqid(top, dev->id); /* we assume this is tiny for timing, so not included */
	omp_offloading_t * off = &off_info->offloadings[seqid];
	//printf("devid: %d --> seqid: %d in top: %X, off: %X, off_info: %X\n", dev->id, seqid, top, off, off_info);

#if defined (OMP_BREAKDOWN_TIMING)
	omp_event_t *events = off->events;
	omp_event_record_start(&events[total_event_index]);
#endif
#if defined (OMP_BREAKDOWN_TIMING)
	if (off_info->num_mapped_vars > 0)
		omp_event_record_start(&events[acc_mapfrom_event_index]);
#endif

	omp_off_map_copyfrom(off, events);

#if defined (OMP_BREAKDOWN_TIMING)
	if (off_info->num_mapped_vars > 0)
		omp_event_record_stop(&events[acc_mapfrom_event_index]);
#endif

	omp_stream_sync(off->stream); /*NOTE: we should NOT time this call as the event system already count in as previous async kernel or async memcpy */
	dev->offload_stack_top--;
	//printf("pop an off %X onto offload stack at position %d\n", off, dev->offload_stack_top+1);
#if defined (OMP_BREAKDOWN_TIMING)
	omp_event_record_start(&events[sync_cleanup_event_index]);
#endif
	omp_off_map_free(off);
#if defined USING_PER_OFFLOAD_STREAM
	omp_stream_destroy(&off->mystream);
#endif
#if defined (OMP_BREAKDOWN_TIMING)
	omp_event_record_stop(&events[sync_cleanup_event_index]);
#endif

	omp_post_off_barrier(off);
}

/**
 * called by the shepherd thread
 */
void omp_offloading_run(omp_device_t * dev) {
	omp_offloading_info_t * off_info = dev->offload_request;
	omp_grid_topology_t * top = off_info->top;
	int seqid = omp_grid_topology_get_seqid(top, dev->id); /* we assume this is tiny for timing, so not included */
	omp_offloading_t * off = &off_info->offloadings[seqid];
	//printf("devid: %d --> seqid: %d in top: %X, off: %X, off_info: %X\n", dev->id, seqid, top, off, off_info);

	int devid = dev->id;
	int i = 0;

#if defined (OMP_BREAKDOWN_TIMING)
	if (off->events == NULL) { /* the first time of recurring offloading or a non-recurring offloading */
		omp_init_off_stream_events(off);
	}
    int num_events = off->num_events;
    omp_event_t *events = off->events;
	omp_event_record_start(&events[total_event_index]);
    omp_off_init_before_run(off);
#endif

	double runtime_profile_elapsed =read_timer_ms();
#if defined (OMP_BREAKDOWN_TIMING)
	omp_event_record_start(&events[map_init_event_index]);
#endif
	omp_off_init_maps(off_info, off, seqid);
#if defined (OMP_BREAKDOWN_TIMING)
	omp_event_record_stop(&events[map_init_event_index]);
#endif

#if defined (OMP_BREAKDOWN_TIMING)
	omp_event_record_start(&events[map_dist_alloc_event_index]);
#endif
	long total = 0;
	if (!off->loop_dist_done) total = omp_loop_iteration_dist(off);
	else total = off->last_total;
	if (total == 0) {
		off->loop_dist_done = 0;
		goto second_offloading;
	}

	//case OMP_OFFLOADING_MAPMEM:
	off->stage = OMP_OFFLOADING_MAPMEM;
	omp_off_map_dist(off_info, off, seqid);

#if defined (OMP_BREAKDOWN_TIMING)
	omp_event_record_stop(&events[map_dist_alloc_event_index]);
#endif

	int misc_event_index = misc_event_index_start;
	if (off_info->type == OMP_OFFLOADING_STANDALONE_DATA_EXCHANGE) goto data_exchange;

//	case OMP_OFFLOADING_COPYTO:
#if defined (OMP_BREAKDOWN_TIMING)
	if (off_info->num_mapped_vars > 0)
		omp_event_record_start(&events[acc_mapto_event_index]);
#endif
	omp_off_map_copyto(off, events);
#if defined (OMP_BREAKDOWN_TIMING)
	if (off_info->num_mapped_vars > 0)
		omp_event_record_stop(&events[acc_mapto_event_index]);
#endif

	{
#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_start(&events[acc_kernel_exe_event_index]);
#endif
		/* launching the kernel */
		void *args = off_info->args;
		void (*kernel_launcher)(omp_offloading_t *, void *) = off_info->kernel_launcher;
		if (args == NULL) args = off->args;
		if (kernel_launcher == NULL) kernel_launcher = off->kernel_launcher;
		kernel_launcher(off, args);
        if (off_info->loop_redist_needed) off->loop_dist_done = 0; /* reset for the next if there is */
#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_stop(&events[acc_kernel_exe_event_index]);
#endif
	}

//	case OMP_OFFLOADING_EXCHANGE:
	data_exchange:;
	/* for data exchange, either a standalone or an appended exchange */
	if (off_info->halo_x_info != NULL) {
		omp_stream_sync(off->stream);/* make sure previous operation are complete, should NOT be timed for exchange */
#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_start(&events[acc_ex_pre_barrier_event_index]);
#endif
		pthread_barrier_wait(&off_info->inter_dev_barrier); /* make sure everybody is completed so we can exchange now */
#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_stop(&events[acc_ex_pre_barrier_event_index]);
#endif

#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_start(&events[acc_ex_event_index]);
#endif
		for (i = 0; i < off_info->num_maps_halo_x; i++) {
			omp_data_map_halo_exchange_info_t *x_halos = &off_info->halo_x_info[i];
			omp_data_map_info_t *map_info = x_halos->map_info;
			//int devseqid = omp_grid_topology_get_seqid(map_info->top, dev->id);

			omp_data_map_t *map = &map_info->maps[seqid];
			omp_halo_region_pull(map, x_halos->x_dim, x_halos->x_direction);
		}
#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_stop(&events[acc_ex_event_index]);
		omp_event_record_start(&events[acc_ex_post_barrier_event_index]);
#endif
//		dev->offload_request = NULL; /* release this dev */
		pthread_barrier_wait(&off_info->inter_dev_barrier);
		//printf("dev: %d (seqid: %d) holo region pull\n", dev->id, seqid);

#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_stop(&events[acc_ex_post_barrier_event_index]);
#endif
//		if (off_info->type == OMP_OFFLOADING_STANDALONE_DATA_EXCHANGE) goto omp_offloading_sync_cleanup;
	}

//	case OMP_OFFLOADING_COPYFROM:
	omp_offloading_copyfrom:;
	off->stage = OMP_OFFLOADING_COPYFROM;
#if defined (OMP_BREAKDOWN_TIMING)
	if (off_info->num_mapped_vars > 0)
		omp_event_record_start(&events[acc_mapfrom_event_index]);
#endif
	/* copy back results */
	omp_off_map_copyfrom(off, events);

#if defined (OMP_BREAKDOWN_TIMING)
	if (off_info->num_mapped_vars > 0)
		omp_event_record_stop(&events[acc_mapfrom_event_index]);
#endif

	/* sync stream to wait for completion of the initial offloading */
	omp_stream_sync(off->stream); /*NOTE: we should NOT time this call as the event system already count in as previous async kernel or async memcpy */
#if defined (OMP_BREAKDOWN_TIMING)
	omp_event_record_start(&events[sync_cleanup_event_index]);
#endif
	for (i=0; i<off->num_maps; i++) {
		int inherited;
		omp_data_map_t *map = omp_map_offcache_iterator(off, i, &inherited);
		if (inherited) continue;
		if (map->info->remap_needed) omp_map_free(map, off);
	}

#if defined (OMP_BREAKDOWN_TIMING)
	omp_event_record_stop(&events[sync_cleanup_event_index]);
#endif

second_offloading: ;
	off->runtime_profile_elapsed = runtime_profile_elapsed;
	{
		omp_dist_policy_t loop_dist_policy = off_info->loop_dist_info[0].policy;
		if (loop_dist_policy == OMP_DIST_POLICY_SCHED_DYNAMIC || loop_dist_policy == OMP_DIST_POLICY_SCHED_GUIDED || loop_dist_policy == OMP_DIST_POLICY_SCHED_FEEDBACK) {
			while (total > 0) {
				runtime_profile_elapsed =read_timer_ms() - runtime_profile_elapsed;
#if defined (OMP_BREAKDOWN_TIMING)
				off->runtime_profile_elapsed = omp_accumulate_elapsed_ms(events, num_events);
#endif
				off->runtime_profile_elapsed = off->runtime_profile_elapsed/total;

				runtime_profile_elapsed =read_timer_ms();
				total = secondary_offload_cycle(off_info, off, events, seqid);
			}
		} else if (loop_dist_policy == OMP_DIST_POLICY_SCHED_PROFILE_AUTO || loop_dist_policy == OMP_DIST_POLICY_MODEL_PROFILE_AUTO) {
			runtime_profile_elapsed =read_timer_ms() - runtime_profile_elapsed;
#if defined (OMP_BREAKDOWN_TIMING)
			off->runtime_profile_elapsed = omp_accumulate_elapsed_ms(events, num_events);
//			printf("Device %d, profile time: %f using events vs %f using timer\n", dev->id, off->runtime_profile_elapsed, runtime_profile_elapsed);
#endif
            //printf("elapsed: %f, per iteration: %f of total iteration: %d\n", off->runtime_profile_elapsed, off->runtime_profile_elapsed/off->loop_dist[0].length, off->loop_dist[0].length);
			off->runtime_profile_elapsed = off->runtime_profile_elapsed/total;
			/* we need barrier here to make sure every device finishes its portion for collective profiling and ratio modeling */

#if defined (OMP_BREAKDOWN_TIMING)
			omp_event_record_start(&events[profiling_barrier_wait_event_index]);
#endif
			pthread_barrier_wait(&off_info->inter_dev_barrier);
#if defined (OMP_BREAKDOWN_TIMING)
			omp_event_record_stop(&events[profiling_barrier_wait_event_index]);
#endif
//			printf("dev %d wait in barrier for the secondary offloading\n", dev->id);
			if (total) secondary_offload_cycle(off_info, off, events, seqid);
		}
	}

	{
		/* sync stream to wait for completion */
#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_start(&events[sync_cleanup_event_index]);
#endif
		if (total) {
			for (i=0; i<off->num_maps; i++) {
				int inherited;
				omp_data_map_t *map = omp_map_offcache_iterator(off, i, &inherited);
				if (inherited) continue;
				if (!map->info->remap_needed) omp_map_free(map, off);
			}
		}
#if defined USING_PER_OFFLOAD_STREAM
		omp_stream_destroy(&off->mystream);
#endif
#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_stop(&events[sync_cleanup_event_index]);
#endif
		off->stage = OMP_OFFLOADING_MDEV_BARRIER;
	}

	omp_post_off_barrier(off);
}

/* helper thread main */
void helper_thread_main(void * arg) {
	omp_device_t * dev = (omp_device_t*)arg;

	omp_set_current_device_dev(dev);
	omp_warmup_device(dev);
	omp_stream_create(dev, &dev->default_stream);
	omp_stream_sync(&dev->default_stream);
//	omp_set_num_threads(dev->num_cores);
	pthread_barrier_wait(&all_dev_sync_barrier);
//	printf("helper threading (devid: %s) loop ....\n", dev->name);
	/*************** loop *******************/
	while (omp_device_complete == 0) {
//		printf("helper threading (devid: %X) waiting ....\n", dev);
		while (dev->offload_request == NULL) {
			if (omp_device_complete) return;
		}
//		printf("helper threading (devid: %X) offloading  ....\n", dev);
		omp_offloading_info_t * off_info = dev->offload_request;
		if (off_info->type == OMP_OFFLOADING_DATA) {
			if (off_info->count % 2 == 1) /* for copyto */
				omp_offloading_data_copyto(dev);
			else omp_offloading_data_copyfrom(dev);
		} else omp_offloading_run(dev);
	}

	omp_stream_destroy(&dev->default_stream);
}
