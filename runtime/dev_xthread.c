#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/timeb.h>

#include "homp.h"

/**
 * notifying the helper threads to work on the offloading specified in off_info arg
 * It always start with copyto and may stops after copyto for target data
 * master is just the thread that will store
 */
void omp_offloading_start(omp_offloading_info_t * off_info) {
	omp_device_t ** targets = off_info->targets;
	int num_targets = off_info->top->nnodes;
    /* generate master trace file */

#if defined (OMP_BREAKDOWN_TIMING)
	if (off_info->count <= 1) off_info->start_time = read_timer_ms(); /* only for the first time */
#endif

	int i;
	for (i = 0; i < num_targets; i++) {
		if (targets[i]->offload_request != NULL) {
			fprintf(stderr, "device %d is not ready for answering your request, offloading_start: %X. It is a bug so far\n", targets[i]->id, off_info);
		}
		targets[i]->offload_request = off_info;
		/* TODO: this is data race if multiple host threads try to offload to the same devices,
		 * FIX is to use cas operation to update this field
		 */
	}
	pthread_barrier_wait(&off_info->barrier);

	if (off_info->type != OMP_OFFLOADING_STANDALONE_DATA_EXCHANGE && off_info->halo_x_info != NULL) { /* appended halo exchange */
		pthread_barrier_wait(&off_info->barrier);
	}
	if (off_info->count) off_info->count++; /* recurring, increment the number of offloading */

#if defined (OMP_BREAKDOWN_TIMING)
	pthread_barrier_wait(&off_info->barrier); /* this one make sure the profiling is collected */
	off_info->compl_time = read_timer_ms();
#endif

}

#if defined (OMP_BREAKDOWN_TIMING)
/* making it global so other can use those values */
int total_event_index = 0;       		/* host event */
int timing_init_event_index = 1; 		/* host event */
int map_init_event_index = 2;  			/* host event */

int acc_mapto_event_index = 3; 			/* dev event */
int kernel_exe_event_index = 4;			/* dev event */
int acc_ex_event_index = 5;  			/* host event for data exchange such as halo xchange */
int acc_ex_barrier_event_index = 6;  	/* host event */
int acc_mapfrom_event_index = 7;		/* dev event */

int sync_cleanup_event_index = 8;		/* host event */
int barrier_wait_event_index = 9;		/* host event */

int misc_event_index_start = 10;        /* other events, e.g. mapto/from for each array, start with 9*/
#endif

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
	/* the num_mapped_vars * 2 +4 is the rough number of events needed */
	/* the event (if mapto var is num_mapto, and mapfrom var is num_mapfrom (both including tofrom);
	 * 0: The whole measured time from host side, measured from host
	 * 1: The init time (stream, event, etc), this is the overhead for the breakdown timing, measured from host
	 * 2: The time for map init, data dist, buffer allocation and data marshalling, measured from host
	 * 3: The accumulated time for mapto datamovement, measured from dev
	 *     4 - kernel_exe_event_index-1: The time for each mapto datamovement, measured from dev (total num_mapto events)
	 * kernel_exe_event_index: kernel exe time
	 * kernel_exe_event_index+1: The accumulated time for mapfrom datamovement, measured from dev
	 *     kernel_exe_event_index+2 - xxxx: The time for each mapfrom datamovement, measured from dev (total num_mapfrom events)
	 * xxxx: The time for cleanup resources (stream, event, data unmarshalling, etc), measured from host
	 * xxxx: The time for barrier wait (for other kernel to complete), measured from host
	 */

	int num_events;
	omp_event_t *events;
	if (off_info->count <= 1) { /* the first time of recurring offloading or a non-recurring offloading */
		num_events = off_info->num_mapped_vars * 2 + 10; /* the max posibble # of events to be used */
		events = (omp_event_t *) malloc(sizeof(omp_event_t) * num_events); /**TODO: free this memory somewhere later */
		off->num_events = num_events;
		off->events = events;
	} else { /* second time and later recurring offloading */
		num_events = off->num_events;
		events = off->events;
	}

	int misc_event_index = misc_event_index_start;

	/* set up stream and event */
	if (off_info->count <= 1) omp_event_init(&events[total_event_index], omp_host_dev, OMP_EVENT_HOST_RECORD);
	omp_event_record_start(&events[total_event_index], NULL, "OFF_TOTAL", "Total offloading time (everything) on dev: %d", devid);
#endif

//	case OMP_OFFLOADING_INIT:
	if (off_info->count <= 1) /* the first time of recurring offloading or a non-recurring offloading */
	{
		off->stage = OMP_OFFLOADING_INIT;
#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_init(&events[timing_init_event_index], omp_host_dev, OMP_EVENT_HOST_RECORD);
		omp_event_record_start(&events[timing_init_event_index], NULL, "INIT_0", "Time for initialization of stream and event", devid);
#endif

#if defined USING_PER_OFFLOAD_STREAM
		omp_stream_create(dev, &off->mystream, 0);
		off->stream = &off->mystream;
#else
		off->stream = &dev->devstream;
#endif
		off->devseqid = seqid;
		off->dev = dev;
		off->off_info = off_info;
		off->num_maps = 0;

#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_init(&events[map_init_event_index], omp_host_dev, OMP_EVENT_HOST_RECORD);
		omp_event_init(&events[sync_cleanup_event_index], omp_host_dev, OMP_EVENT_HOST_RECORD);
		omp_event_init(&events[barrier_wait_event_index], omp_host_dev, OMP_EVENT_HOST_RECORD);
		omp_event_init(&events[acc_mapto_event_index], dev, OMP_EVENT_DEV_RECORD);
		omp_event_init(&events[kernel_exe_event_index], dev, OMP_EVENT_DEV_RECORD);
		omp_event_init(&events[acc_mapfrom_event_index], dev, OMP_EVENT_DEV_RECORD);
		omp_event_init(&events[acc_ex_event_index], omp_host_dev, OMP_EVENT_HOST_RECORD);
		omp_event_init(&events[acc_ex_barrier_event_index], omp_host_dev, OMP_EVENT_HOST_RECORD);

		for (i=misc_event_index_start; i<num_events; i++) {
			omp_event_init(&events[i], dev, OMP_EVENT_DEV_RECORD);
		}

		omp_event_record_stop(&events[timing_init_event_index]);
#endif

	    //case OMP_OFFLOADING_MAPMEM:
		off->stage = OMP_OFFLOADING_MAPMEM;
		/* init data map and dev memory allocation */
		/***************** for each mapped variable has to and tofrom, if it has region mapped to this __ndev_i__ id, we need code here *******************************/
#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_start(&events[map_init_event_index], NULL, "INIT_1", "Time for init map, data dist, buffer allocation, and data marshalling");
#endif
		for (i=0; i<off_info->num_mapped_vars; i++) {
			/* we handle inherited map here, by each helper thread, and we only update the off object (not off_info)*/
			omp_data_map_info_t * map_info = &off_info->data_map_info[i];

			omp_data_map_t * map = omp_map_get_map_inheritance (dev, map_info->source_ptr);
			int inherited = 1;
			if (map == NULL) { /* here we basically ignore any map specification if it can inherit from ancestor (upper level nested target data) */
				map = &map_info->maps[seqid];
				omp_data_map_init_map(map, map_info, dev, off->stream, off);
				omp_data_map_dist(map, seqid, off);
				omp_map_buffer(map, off);
				inherited = 0;
			}
			omp_offload_append_map_to_cache(off, map, inherited);
			//omp_print_data_map(map);
		}
#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_stop(&events[map_init_event_index]);
#endif
	}

	omp_dev_stream_t *stream = off->stream;

	if (off_info->type == OMP_OFFLOADING_STANDALONE_DATA_EXCHANGE) goto data_exchange;
	if (off_info->type == OMP_OFFLOADING_DATA && off_info->count > 1) {
		goto omp_offloading_copyfrom;
	}

//	case OMP_OFFLOADING_COPYTO:
	{
omp_offloading_copyto: ;
		off->stage = OMP_OFFLOADING_COPYTO;
#if defined (OMP_BREAKDOWN_TIMING)
		if (off_info->num_mapped_vars > 0)
			omp_event_record_start(&events[acc_mapto_event_index], stream, "ACC_MAPTO", "Accumulated time for mapto data movement for all array");
#endif
		for (i=0; i<off_info->num_mapped_vars; i++) {
			omp_data_map_info_t * map_info = &off_info->data_map_info[i];
			omp_data_map_t * map = &map_info->maps[seqid];
			if (omp_map_is_map_inherited(off, map)) continue;

			if (map_info->map_direction == OMP_DATA_MAP_TO || map_info->map_direction == OMP_DATA_MAP_TOFROM) {
#if defined (OMP_BREAKDOWN_TIMING)
				omp_event_record_start(&events[misc_event_index], stream, "MAPTO_", "Time for mapto data movement for array %s", map_info->symbol);
#endif
				omp_map_mapto_async(map, off->stream);
				//omp_map_memcpy_to_async((void*)map->map_dev_ptr, dev, (void*)map->map_buffer, map->map_size, off->stream); /* memcpy from host to device */
#if defined (OMP_BREAKDOWN_TIMING)
				omp_event_record_stop(&events[misc_event_index++]);
#endif
			}
		}
#if defined (OMP_BREAKDOWN_TIMING)
		if (off_info->num_mapped_vars > 0)
			omp_event_record_stop(&events[acc_mapto_event_index]);
#endif
	}

	if (off_info->type == OMP_OFFLOADING_DATA) { /* only data offloading, i.e., OMP_OFFLOADING_DATA */
		//assert (off_info->recurring == 1);
		off->stage = OMP_OFFLOADING_SYNC;
		goto omp_offloading_sync_cleanup;
	} else {
		off->stage = OMP_OFFLOADING_KERNEL;
	}

//	case OMP_OFFLOADING_KERNEL:
	{
#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_start(&events[kernel_exe_event_index], stream, "KERN", "Time for kernel (%s) execution", off_info->name);
#endif
		/* launching the kernel */
		void * args = off_info->args;
		void (*kernel_launcher)(omp_offloading_t *, void *) = off_info->kernel_launcher;
		if (args == NULL) args = off->args;
		if (kernel_launcher == NULL) kernel_launcher = off->kernel_launcher;
		kernel_launcher(off, args);
#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_stop(&events[kernel_exe_event_index]);
#endif
	}
//	case OMP_OFFLOADING_EXCHANGE:
	{

	}

//	case OMP_OFFLOADING_COPYFROM:
	{
omp_offloading_copyfrom: ;
		off->stage = OMP_OFFLOADING_COPYFROM;
#if defined (OMP_BREAKDOWN_TIMING)
		if (off_info->num_mapped_vars > 0)
			omp_event_record_start(&events[acc_mapfrom_event_index], stream,  "ACC_MAPFROM", "Accumulated time for mapfrom data movement for all array");
#endif
		/* copy back results */
		for (i=0; i<off_info->num_mapped_vars; i++) {
			omp_data_map_info_t * map_info = &off_info->data_map_info[i];
			omp_data_map_t * map = &map_info->maps[seqid];
			if (omp_map_is_map_inherited(off, map)) continue;

			if (map_info->map_direction == OMP_DATA_MAP_FROM || map_info->map_direction == OMP_DATA_MAP_TOFROM) {
#if defined (OMP_BREAKDOWN_TIMING)
				/* TODO bug here if this is reached from the above goto, since events is not available */
				omp_event_record_start(&events[misc_event_index], stream, "MAPFROM_", "Time for mapfrom data movement for array %s", map_info->symbol);
#endif
				omp_map_mapfrom_async(map, off->stream);
				//omp_map_memcpy_from_async((void*)map->map_buffer, (void*)map->map_dev_ptr, dev, map->map_size, off->stream); /* memcpy from host to device */
#if defined (OMP_BREAKDOWN_TIMING)
				omp_event_record_stop(&events[misc_event_index++]);
#endif
			}
		}
#if defined (OMP_BREAKDOWN_TIMING)
		if (off_info->num_mapped_vars > 0)
			omp_event_record_stop(&events[acc_mapfrom_event_index]);
#endif
	}

	//case OMP_OFFLOADING_SYNC:
	//case OMP_OFFLOADING_SYNC_CLEANUP:
	{
omp_offloading_sync_cleanup: ;
		/* sync stream to wait for completion */
#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_start(&events[sync_cleanup_event_index], NULL, "FINI_1", "Time for dev sync and cleaning (event/stream/map, deallocation/unmarshalling)");
#endif
		omp_stream_sync(off->stream);
		if (off->stage == OMP_OFFLOADING_SYNC) {
			if (off_info->type == OMP_OFFLOADING_DATA) { /* this should be just an assertation */
				/* put in the offloading stack */
				dev->offload_stack_top++;
				dev->offload_stack[dev->offload_stack_top] = off;
				//printf("pushing an off %X onto offload stack at position %d\n", off, dev->offload_stack_top);
			}
		} else {
			if (off_info->type == OMP_OFFLOADING_DATA) { /* pop up this offload stack */
				dev->offload_stack_top--;
				//printf("pop an off %X onto offload stack at position %d\n", off, dev->offload_stack_top+1);
			}

			off->stage = OMP_OFFLOADING_SYNC_CLEANUP;
			omp_cleanup(off);
		}
#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_stop(&events[sync_cleanup_event_index]);
#endif
		off->stage = OMP_OFFLOADING_MDEV_BARRIER;
	}
//	case OMP_OFFLOADING_MDEV_BARRIER:
	{
#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_start(&events[barrier_wait_event_index], NULL, "BAR_FINI_2", "Time for barrier wait for other to complete");
#endif
		dev->offload_request = NULL; /* release this dev */
		pthread_barrier_wait(&off_info->barrier);
#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_stop(&events[barrier_wait_event_index]);
#endif
		//off_info->stage = OMP_OFFLOADING_COMPLETE; /* data race for any access to off_info
	}
data_exchange:;
	/* for data exchange, either a standalone or an appended exchange */
	if (off_info->halo_x_info != NULL) {
#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_start(&events[acc_ex_event_index], NULL, "DATA_X", "Time for data exchange between devices");
#endif
		for (i=0; i<off_info->num_maps_halo_x; i++) {
			omp_data_map_halo_exchange_info_t * x_halos = &off_info->halo_x_info[i];
			omp_data_map_info_t * map_info = x_halos->map_info;
			//int devseqid = omp_grid_topology_get_seqid(map_info->top, dev->id);

			omp_data_map_t * map = &map_info->maps[seqid];
			//printf("dev: %d (seqid: %d) holo region pull\n", dev->id, devseqid);
			omp_halo_region_pull(map, x_halos->x_dim, x_halos->x_direction);
		}
#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_stop(&events[acc_ex_event_index]);
		omp_event_record_start(&events[acc_ex_barrier_event_index], NULL, "BAR_DATA_X", "Time for barrier sync for data exchange between devices");
#endif
		dev->offload_request = NULL; /* release this dev */
		pthread_barrier_wait(&off_info->barrier);

#if defined (OMP_BREAKDOWN_TIMING)
		omp_event_record_stop(&events[acc_ex_barrier_event_index]);
#endif
	}

#if defined (OMP_BREAKDOWN_TIMING)
	omp_event_record_stop(&events[total_event_index]);
#endif

	/* print out the timing info */
#if defined (OMP_BREAKDOWN_TIMING)
	/* do timing accumulation if this is a recurring kernel */
	for (i=0; i<num_events; i++) {
		omp_event_accumulate_elapsed_ms(&events[i]);
	}
	pthread_barrier_wait(&off_info->barrier);
#endif
}

/* helper thread main */
void helper_thread_main(void * arg) {
	omp_device_t * dev = (omp_device_t*)arg;
	omp_set_current_device_dev(dev);
	omp_stream_create(dev, &dev->devstream, 1);

	/*************** loop *******************/
	while (omp_device_complete == 0) {
		//	printf("helper threading (devid: %d) waiting ....\n", devid);
		while (dev->offload_request == NULL) {
			if (omp_device_complete) return;
		}
		omp_offloading_run(dev);
	}
}
