/*
 * homp_dev.c
 *
 * contains dev-specific implementation of homp.h functions, mainly those with
 *
 *  Created on: Oct 4, 2014
 *      Author: yy8
 */
/**
 * an easy way for defining dev-specific code:
#if defined (DEVICE_NVGPU_SUPPORT)

#elif defined (DEVICE_LOCALTH)

#else

#endif
 */
#include "homp.h"


inline void devcall_errchk(int code, char *file, int line, int abort) {
#if defined (DEVICE_NVGPU_SUPPORT)
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
#elif defined (DEVICE_LOCALTH)
	if (code != 0) {
		fprintf(stderr, "devcal_assert: %d %s %d\n", code, file, line);
		if (abort) exit(code);
	}
#endif
}

/* init the device objects, num_of_devices, default_device_var ICV etc
 *
 */
void omp_init_devices() {
	/* query hardware device */
	int num_gpudevs = 0;
	omp_num_devices = 0;
#if defined (DEVICE_NVGPU_SUPPORT)
	cudaError_t result = cudaGetDeviceCount(&num_gpudevs);
	devcall_assert(result);
	omp_num_devices += num_gpudevs;
#endif
	/* query other type of device */

	/* the thread-simulated devices */
	int num_thsimdev;

	char * num_thsimdev_str = getenv("NUM_THSIM_DEVICES");
	if (num_thsimdev_str != NULL ) sscanf(num_thsimdev_str, "%d", &num_thsimdev);
	if (num_thsimdev < 0) num_thsimdev = 0;
	omp_num_devices += num_thsimdev;

	omp_devices = malloc(sizeof(omp_device_t) * omp_num_devices);
	int i;

	/* the helper thread setup */
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	/* initialize attr with default attributes */
	pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
	pthread_setconcurrency(omp_num_devices);

	for (i=0; i<omp_num_devices; i++)
	{
		omp_device_t * dev = &omp_devices[i];
		dev->id = i;
		if (i < num_gpudevs) dev->type = OMP_DEVICE_NVGPU;
		else dev->type  = OMP_DEVICE_LOCALTH;
		dev->status = 1;
		dev->sysid = i;
		dev->resident_data_maps = NULL;
		dev->next = &omp_devices[i+1];
		dev->notification_counter = -1;

		int rt = pthread_create(&dev->helperth, &attr, (void *(*)(void *))helper_thread_main, (void *) dev);
		if (rt) {fprintf(stderr, "cannot create helper threads for devices.\n"); exit(1); }
	}
	if (omp_num_devices) {
		default_device_var = 0;
		omp_devices[omp_num_devices-1].next = NULL;
	}
	printf("System has total %d devices, %d GPU, %d THSIM devices, and the number of THSIM devices can be controlled by setting the NUM_THSIM_DEVICES env,\
	  and number of active (enabled) devices can be controlled by setting OMP_NUM_ACTIVE_DEVICES variable\n", omp_num_devices, num_gpudevs, num_thsimdev);
}


void omp_set_current_device(omp_device_t * d) {
#if defined (DEVICE_NVGPU_SUPPORT)
    int result;
	if (d->type == OMP_DEVICE_NVGPU) {
		result = cudaSetDevice(d->sysid);
		devcall_assert (result);
	}
#endif
}

void omp_map_malloc_dev(omp_data_map_t * map) {
	omp_device_type_t devtype = map->dev->type;
#if defined (DEVICE_NVGPU_SUPPORT)
	if (devtype == OMP_DEVICE_NVGPU) {
		if (cudaErrorMemoryAllocation == cudaMalloc(&map->mem_dev_ptr, map->map_size)) {
			fprintf(stderr, "cudaMalloc error to allocate mem on device\n");
		}
	} else
#endif
	if (devtype == OMP_DEVICE_LOCALTH) {
		map->mem_dev_ptr = malloc(map->map_size);
	} else {
		fprintf(stderr, "device type is not supported for this call\n");
	}
}

void omp_map_memcpy_to(omp_data_map_t * map) {
	omp_device_type_t devtype = map->dev->type;
#if defined (DEVICE_NVGPU_SUPPORT)
	if (devtype == OMP_DEVICE_NVGPU) {
	    cudaError_t result;
	    result = cudaMemcpy((void *)map->map_dev_ptr,(const void *)map->map_buffer,map->map_size, cudaMemcpyHostToDevice);
	    devcall_assert(result);
	} else
#endif
	if (devtype == OMP_DEVICE_LOCALTH) {
		memcpy((void *)map->map_dev_ptr,(const void *)map->map_buffer,map->map_size);
	} else {
		fprintf(stderr, "device type is not supported for this call\n");
	}
}

void omp_map_memcpy_to_async(omp_data_map_t * map) {
	omp_device_type_t devtype = map->dev->type;
#if defined (DEVICE_NVGPU_SUPPORT)
	if (devtype == OMP_DEVICE_NVGPU) {
		cudaError_t result;
		result = cudaMemcpyAsync((void *)map->map_dev_ptr,(const void *)map->map_buffer,map->map_size, cudaMemcpyHostToDevice, map->stream->systream.cudaStream);
		devcall_assert(result);
	} else
#endif
	if (devtype == OMP_DEVICE_LOCALTH) {
		fprintf(stderr, "no async call support, use sync memcpy call\n");
		memcpy((void *)map->map_dev_ptr, (const void *)map->map_buffer, map->map_size);
	} else {
		fprintf(stderr, "device type is not supported for this call\n");
	}
}

void omp_map_memcpy_from(omp_data_map_t * map) {
	omp_device_type_t devtype = map->dev->type;
#if defined (DEVICE_NVGPU_SUPPORT)
	if (devtype == OMP_DEVICE_NVGPU) {
		cudaError_t result;
	    result = cudaMemcpy((void *)map->map_buffer,(const void *)map->map_dev_ptr,map->map_size, cudaMemcpyDeviceToHost);
		devcall_assert(result);
	} else
#endif
	if (devtype == OMP_DEVICE_LOCALTH) {
		memcpy((void *)map->map_buffer, (const void *)map->map_dev_ptr, map->map_size);
	} else {
		fprintf(stderr, "device type is not supported for this call\n");
	}
}

/**
 *  device to host, async */
void omp_map_memcpy_from_async(omp_data_map_t * map) {
	omp_device_type_t devtype = map->dev->type;
#if defined (DEVICE_NVGPU_SUPPORT)
	if (devtype == OMP_DEVICE_NVGPU) {
		cudaError_t result;
        result = cudaMemcpyAsync((void *)map->map_buffer,(const void *)map->map_dev_ptr,map->map_size, cudaMemcpyDeviceToHost, map->stream->systream.cudaStream);
		devcall_assert(result);
	} else
#endif
	if (devtype == OMP_DEVICE_LOCALTH) {
		fprintf(stderr, "no async call support, use sync memcpy call\n");
		memcpy((void *)map->map_buffer, (const void *)map->map_dev_ptr, map->map_size);
	} else {
		fprintf(stderr, "device type is not supported for this call\n");
	}
}

void omp_map_memcpy_DeviceToDevice(omp_data_map_t * dst, omp_data_map_t * src, int size) {
	omp_device_type_t dst_devtype = dst->dev->type;
	omp_device_type_t src_devtype = src->dev->type;

#if defined (DEVICE_NVGPU_SUPPORT)
	if (dst_devtype == OMP_DEVICE_NVGPU && src_devtype == OMP_DEVICE_NVGPU) {
		cudaError_t result;
	    result = cudaMemcpy((void *)dst->map_dev_ptr,(const void *)src->map_dev_ptr,size, cudaMemcpyDeviceToDevice);
		devcall_assert(result);
	} else
#endif
	if (dst_devtype == OMP_DEVICE_LOCALTH && src_devtype == OMP_DEVICE_LOCALTH) {
		memcpy((void *)dst->map_dev_ptr, (const void *)src->map_dev_ptr, size);
	} else {
		fprintf(stderr, "device type is not supported for this call, currently we only support p2p copy between GPU-GPU and TH-TH\n");
	}
}

void omp_map_memcpy_DeviceToDeviceAsync(omp_data_map_t * dst, omp_data_map_t * src, int size) {
	omp_device_type_t dst_devtype = dst->dev->type;
	omp_device_type_t src_devtype = src->dev->type;

#if defined (DEVICE_NVGPU_SUPPORT)
	if (dst_devtype == OMP_DEVICE_NVGPU && src_devtype == OMP_DEVICE_NVGPU) {
		cudaError_t result;
	    result = cudaMemcpyAsync((void *)dst->map_dev_ptr,(const void *)src->map_dev_ptr,size, cudaMemcpyDeviceToDevice,src->stream->systream.cudaStream);
		devcall_assert(result);
	} else
#endif
	if (dst_devtype == OMP_DEVICE_LOCALTH && src_devtype == OMP_DEVICE_LOCALTH) {
		memcpy((void *)dst->map_dev_ptr, (const void *)src->map_dev_ptr, size);
	} else {
		fprintf(stderr, "device type is not supported for this call, currently we only support p2p copy between GPU-GPU and TH-TH\n");
	}
}

#if defined (DEVICE_NVGPU_SUPPORT)
void xomp_beyond_block_reduction_float_stream_callback(cudaStream_t stream,  cudaError_t status, void*  userData ) {
	omp_reduction_float_t * rdata = (omp_reduction_float_t*)userData;
	float result = 0.0;
	int i;
	for (i=0; i<rdata->num; i++)
		result += rdata->input[i];
	rdata->result = result;
}
#endif

void omp_init_stream(omp_device_t * d, omp_dev_stream_t * stream) {
	stream->dev = d;
	int i;

#if defined (DEVICE_NVGPU_SUPPORT)
	cudaError_t result;
	if (d->type == OMP_DEVICE_NVGPU) {
		result = cudaStreamCreate(&stream->systream.cudaStream);
		devcall_assert(result);
		for (i=0; i<OMP_DEV_STREAM_NUM_EVENTS; i++) {
			result = cudaEventCreateWithFlags(&stream->start_event[i], cudaEventBlockingSync);
			devcall_assert(result);
			result = cudaEventCreateWithFlags(&stream->stop_event[i], cudaEventBlockingSync);
			devcall_assert(result);
			stream->elapsed[i] = 0.0;
		}
	} else {
		fprintf(stderr, "device type (%d) is not yet supported!\n", d->type);
	}
#else
	/** other type of device stream support */
#endif
	for (i=0; i<OMP_DEV_STREAM_NUM_EVENTS; i++) {
		stream->elapsed[i] = 0.0;
	}
}

#if defined (DEVICE_NVGPU_SUPPORT)
#ifdef USE_STREAM_HOST_CALLBACK_4_TIMING
void omp_stream_host_timer_callback(cudaStream_t stream,  cudaError_t status, void*  userData ) {
	float * time = (float*)userData;
	*time = read_timer_ms();
}
#endif
#endif

void omp_stream_start_event_record(omp_dev_stream_t * stream, int event) {
#if defined (DEVICE_NVGPU_SUPPORT)
    cudaError_t result;
#ifdef USE_STREAM_HOST_CALLBACK_4_TIMING
	result = cudaStreamAddCallback(stream->systream.cudaStream, omp_stream_host_timer_callback, &stream->start_time[event], 0);
#else
	result = cudaEventRecord(stream->start_event[event], stream->systream.cudaStream);
#endif
    devcall_assert(result);
#else
    stream->start_time[event] = read_timer_ms();
#endif
}

void omp_stream_stop_event_record(omp_dev_stream_t * stream, int event) {
#if defined (DEVICE_NVGPU_SUPPORT)
	cudaError_t result;
#ifdef USE_STREAM_HOST_CALLBACK_4_TIMING
	result = cudaStreamAddCallback(stream->systream.cudaStream, omp_stream_host_timer_callback, &stream->stop_time[event], 0);
#else
	result = cudaEventRecord(stream->stop_event[event], stream->systream.cudaStream);
#endif
	devcall_assert(result);
#else
    stream->stop_time[event] = read_timer_ms();
#endif
}

/**
 * Computes the elapsed time between two events (in milliseconds with a resolution of around 0.5 microseconds).
 */
float omp_stream_event_elapsed_ms(omp_dev_stream_t * stream, int event) {
	float elapse;
#if defined (DEVICE_NVGPU_SUPPORT)
#ifdef USE_STREAM_HOST_CALLBACK_4_TIMING
	elapse = stream->stop_time[event] - stream->start_time[event];
#else
	cudaError_t result;
	result = cudaEventSynchronize(stream->start_event[event]);
	devcall_assert(result);
	result = cudaEventSynchronize(stream->stop_event[event]);
	devcall_assert(result);
	result = cudaEventElapsedTime(&elapse, stream->start_event[event], stream->stop_event[event]);
	devcall_assert(result);
#endif
	stream->elapsed[event] = elapse;
#else
	elapse = stream->stop_time[event] - stream->start_time[event];
#endif
	return elapse;
}

/* accumulate the elapsed time of the event to the stream object and return the elapsed of this event
 */
float omp_stream_event_elapsed_accumulate_ms(omp_dev_stream_t * stream, int event) {
	float elapse;
#if defined (DEVICE_NVGPU_SUPPORT)
#ifdef USE_STREAM_HOST_CALLBACK_4_TIMING
	elapse = stream->stop_time[event] - stream->start_time[event];
#else
	cudaEventElapsedTime(&elapse, stream->start_event[event], stream->stop_event[event]);
#endif
#else
	elapse = stream->stop_time[event] - stream->start_time[event];
#endif
	stream->elapsed[event] += elapse;
	return elapse;
}

/**
 * sync device by syncing the stream so all the pending calls the stream are completed
 *
 * if destroy_stream != 0; the stream will be destroyed.
 */
void omp_sync_stream(omp_dev_stream_t *st, int destroy_stream) {
#if defined (DEVICE_NVGPU_SUPPORT)
	cudaError_t result;
	if (destroy_stream) {
		result = cudaStreamSynchronize(st->systream.cudaStream);
		devcall_assert(result);
		result = cudaStreamDestroy(st->systream.cudaStream);
		devcall_assert(result);
	} else {
		result = cudaStreamSynchronize(st->systream.cudaStream);
		devcall_assert(result);
	}
#else
#endif
}

void omp_sync_cleanup(int num_devices, int num_maps, omp_dev_stream_t dev_stream[num_devices], omp_data_map_t data_map[]) {
	int i, j;
	omp_dev_stream_t * st;
        cudaError_t result;

	for (i=0; i<num_devices; i++) {
		st = &dev_stream[i];
		result = cudaSetDevice(st->dev->sysid);
                devcall_assert(result);
		result = cudaStreamSynchronize(st->systream.cudaStream);
                devcall_assert(result);
		result = cudaStreamDestroy(st->systream.cudaStream);
                devcall_assert(result);
	    for (j=0; j<num_maps; j++) {
	    	omp_data_map_t * map = &data_map[i*num_maps+j];
	    	result = cudaFree(map->mem_dev_ptr);
                devcall_assert(result);
	    	if (map->marshalled_or_not) { /* if this is marshalled and need to free space since this is not useful anymore */
	    		omp_data_map_unmarshal(map);
	    		free(map->map_buffer);
	    	}
	    }
	}
}


