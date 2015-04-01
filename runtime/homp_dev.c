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

#elif defined (DEVICE_THSIM)

#else

#endif
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <math.h>
#include "homp.h"

inline void devcall_errchk(int code, char *file, int line, int ab) {
#if defined (DEVICE_NVGPU_SUPPORT)
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (ab) { abort();}
	}
#elif defined (DEVICE_THSIM)
	if (code != 0) {
		fprintf(stderr, "devcal_assert: %d %s %d\n", code, file, line);
		if (ab) { abort();}
	}
#endif
}

double addmul(double add, double mul, int ops){
	// need to initialise differently otherwise compiler might optimise away
	double sum1=0.1, sum2=-0.1, sum3=0.2, sum4=-0.2, sum5=0.0;
	double mul1=1.0, mul2= 1.1, mul3=1.2, mul4= 1.3, mul5=1.4;
	int loops=ops/10;          // we have 10 floating point ops inside the loop
	double expected = 5.0*add*loops + (sum1+sum2+sum3+sum4+sum5)
			+ pow(mul,loops)*(mul1+mul2+mul3+mul4+mul5);

	int i;
	for(i=0; i<loops; i++) {
		mul1*=mul; mul2*=mul; mul3*=mul; mul4*=mul; mul5*=mul;
		sum1+=add; sum2+=add; sum3+=add; sum4+=add; sum5+=add;
	}
	return  sum1+sum2+sum3+sum4+sum5+mul1+mul2+mul3+mul4+mul5 - expected;
}

double cpu_sustain_gflopss (double * flopss) {
	double x=M_PI;
	double y=1.0+1e-8;
	int n = 1000000;
	double timer = read_timer();
	x=addmul(x,y,n);
	timer = read_timer() - timer;
	*flopss = n/timer/1e9;
}

void * omp_init_dev_specific(omp_device_t * dev) {
	omp_device_type_t devtype = dev->type;
	dev->devstream.dev = dev;
	dev->devstream.systream.myStream = NULL;
#if defined (DEVICE_NVGPU_SUPPORT)
	if (devtype == OMP_DEVICE_NVGPU) {
		dev->dev_properties = (struct cudaDeviceProp*)malloc(sizeof(struct cudaDeviceProp));
		cudaSetDevice(dev->sysid);
		cudaGetDeviceProperties(dev->dev_properties, dev->sysid);
		dev->devstream.systream.cudaStream = 0;

		/* warm up the device */
		void * dummy_dev;
		char dummy_host[1024];
		cudaMalloc(&dummy_dev, 1024);
		cudaMemcpy(dummy_dev, dummy_host, 1024, cudaMemcpyHostToDevice);
		cudaMemcpy(dummy_host, dummy_dev, 1024, cudaMemcpyDeviceToHost);
		cudaFree(dummy_dev);
	} else
#endif
	if (devtype == OMP_DEVICE_THSIM) {
		dev->dev_properties = &dev->helperth; /* make it point to the thread id */
		dev->num_cores = omp_host_dev->num_cores;
		dev->flopss_percore = omp_host_dev->flopss_percore;
	} else {
	}

	dev->total_real_flopss = dev->num_cores * (dev->flopss_percore*(1+dev->id));
	dev->bandwidth = (1+dev->id)*omp_host_dev->bandwidth / 100;
	dev->latency = (1+dev->id)*omp_host_dev->latency * 1000;
	return dev->dev_properties;
}

/* init the device objects, num_of_devices, helper threads, default_device_var ICV etc
 *
 */
int omp_init_devices() {
	/* query hardware device */
	omp_num_devices = 0; /* we always have at least host device */

	/* the thread-simulated devices */
	int num_thsim_dev;
	int i;

	char * num_thsim_dev_str = getenv("OMP_NUM_THSIM_DEVICES");
	if (num_thsim_dev_str != NULL ) {
		sscanf(num_thsim_dev_str, "%d", &num_thsim_dev);
		if (num_thsim_dev < 0) num_thsim_dev = 1;
	} else num_thsim_dev = 1;

	omp_num_devices += num_thsim_dev;
	omp_device_types[OMP_DEVICE_THSIM].num_devs = num_thsim_dev;

	/* for NVDIA GPU devices */
	int num_nvgpu_dev = 0;
	int total_gpudevs = 0;

#if defined (DEVICE_NVGPU_SUPPORT)
	cudaError_t result = cudaGetDeviceCount(&total_gpudevs);
	devcall_assert(result);
#endif

	int gpu_selection[total_gpudevs];
	for (i=0; i<total_gpudevs;i++) gpu_selection[i] = 0;

#if defined (DEVICE_NVGPU_SUPPORT)
	if (total_gpudevs > 0) {
		char * nvgpu_dev_str = getenv("OMP_NVGPU_DEVICES");
		if (nvgpu_dev_str != NULL ) {
			char * token = strtok(nvgpu_dev_str, ",");
			while(token != NULL) {
				int gpuid;
				sscanf(token, "%d", &gpuid);
				gpu_selection[gpuid] = 1;
				num_nvgpu_dev ++;
				token = strtok(NULL, ",");
			}
		} else {
			char * num_nvgpu_dev_str = getenv("OMP_NUM_NVGPU_DEVICES");
			if (num_nvgpu_dev_str != NULL ) {
				sscanf(num_nvgpu_dev_str, "%d", &num_nvgpu_dev);
				if (num_nvgpu_dev > total_gpudevs || num_nvgpu_dev < 0) num_nvgpu_dev = total_gpudevs;
			} else num_nvgpu_dev = total_gpudevs;
			for (i=0; i<num_nvgpu_dev;i++) gpu_selection[i] = 1;
		}

		omp_num_devices += num_nvgpu_dev;
		omp_device_types[OMP_DEVICE_NVGPU].num_devs = num_nvgpu_dev;
	}
#endif

	omp_host_dev = malloc(sizeof(omp_device_t) * (omp_num_devices+1));
	omp_devices = &omp_host_dev[1];
	omp_host_dev->id = -1;
	omp_host_dev->type = OMP_DEVICE_HOST;
	omp_host_dev->sysid = 0;
	omp_host_dev->status = 1;
	omp_host_dev->resident_data_maps = NULL;
	omp_host_dev->next = omp_devices;
	omp_host_dev->offload_request = NULL;
	omp_host_dev->offload_stack_top = -1;
	/** compute the performance */
	omp_host_dev->num_cores = sysconf( _SC_NPROCESSORS_ONLN );
	double dummy = cpu_sustain_gflopss(&omp_host_dev->flopss_percore);
	omp_host_dev->total_real_flopss = omp_host_dev->num_cores * omp_host_dev->flopss_percore;
	omp_host_dev->num_cores = omp_host_dev->num_cores - num_nvgpu_dev; /* reserve the cores for dev thread */
	omp_host_dev->bandwidth = 600*1000; /* GB/s */
	omp_host_dev->latency = 0.02; /* us, i.e. 20 ns */

	/* the helper thread setup */
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	/* initialize attr with default attributes */
	pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	pthread_setconcurrency(omp_num_devices+1);

	int j = 0;
	for (i=0; i<omp_num_devices; i++) {
		omp_device_t * dev = &omp_devices[i];
		dev->id = i;
		if (i < num_thsim_dev) {
			dev->type  = OMP_DEVICE_THSIM;
			dev->mem_type = OMP_DEVICE_MEM_SHARED_CC_NUMA;
			dev->sysid = i;
		} else {
			dev->type = OMP_DEVICE_NVGPU;
			dev->mem_type = OMP_DEVICE_MEM_DISCRETE;
			for (; j<total_gpudevs; j++) {
				if (gpu_selection[j]) {
					break;
				}
			}
			dev->sysid = j;
			j++;
		}
		dev->status = 1;
		dev->resident_data_maps = NULL;
		dev->next = &omp_devices[i+1];
		dev->offload_request = NULL;
		dev->offload_stack_top = -1;
		omp_init_dev_specific(dev);

		printf("Dev %d, %dcores,%fGFLOPs/s, %fMB/s, %fus\n",
			   i, dev->num_cores, dev->total_real_flopss, dev->bandwidth,dev->latency);

		int rt = pthread_create(&dev->helperth, &attr, (void *(*)(void *))helper_thread_main, (void *) dev);
		if (rt) {fprintf(stderr, "cannot create helper threads for devices.\n"); exit(1); }
	}
	if (omp_num_devices) {
		default_device_var = 0;
		omp_devices[omp_num_devices-1].next = NULL;
	}

	printf("System has total %d devices(%d GPU and %d THSIM devices).\n", omp_num_devices, num_nvgpu_dev, num_thsim_dev);
	printf("The number of each type of devices can be controlled by environment variables:\n");
	printf("\tOMP_NUM_THSIM_DEVICES for THSIM devices (default 0)\n");
	printf("\tOMP_NVGPU_DEVICES for selecting specific NVGPU devices (e.g., \"0,2,3\", i.e. ,separated list with no spaces)\n");
	printf("\tOMP_NUM_NVGPU_DEVICES for selecting a number of NVIDIA GPU devices from dev 0 (default, total available, overwritten by OMP_NVGPU_DEVICES)\n");
	printf("\tTo make a specific number of devices available, use OMP_NUM_ACTIVE_DEVICES (default, total number of system devices)\n");
	return omp_num_devices;
}
// terminate helper threads
void omp_fini_devices() {
	int i;

	omp_device_complete = 1;
	for (i=0; i<omp_num_devices; i++) {
		omp_device_t * dev = &omp_devices[i];
		int rt = pthread_join(dev->helperth, NULL);
		omp_device_type_t devtype = dev->type;
#if defined (DEVICE_NVGPU_SUPPORT)
		if (devtype == OMP_DEVICE_NVGPU) {
			free(dev->dev_properties);
		}
#endif
	}

	free(omp_host_dev);
}

int omp_set_current_device_dev(omp_device_t * d) {
#if defined (DEVICE_NVGPU_SUPPORT)
    int result;
	if (d->type == OMP_DEVICE_NVGPU) {
		result = cudaSetDevice(d->sysid);
		devcall_assert (result);
	}
#endif
	return d->id;
}

void omp_map_mapto(omp_data_map_t * map) {
	if (map->map_type == OMP_DATA_MAP_COPY) omp_map_memcpy_to((void*)map->map_dev_ptr, map->dev, (void*)map->map_buffer, map->map_size);
}

void omp_map_mapto_async(omp_data_map_t * map, omp_dev_stream_t * stream) {
	if (map->map_type == OMP_DATA_MAP_COPY) omp_map_memcpy_to_async((void*)map->map_dev_ptr, map->dev, (void*)map->map_buffer, map->map_size, stream);
}

void omp_map_mapfrom(omp_data_map_t * map) {
	if (map->map_type == OMP_DATA_MAP_COPY)
		omp_map_memcpy_from((void*)map->map_buffer, (void*)map->map_dev_ptr, map->dev, map->map_size); /* memcpy from host to device */
}

void omp_map_mapfrom_async(omp_data_map_t * map, omp_dev_stream_t * stream) {
	if (map->map_type == OMP_DATA_MAP_COPY)
		omp_map_memcpy_from_async((void*)map->map_buffer, (void*)map->map_dev_ptr, map->dev, map->map_size, stream); /* memcpy from host to device */
}

void * omp_map_malloc_dev(omp_device_t * dev, long size) {
	omp_device_type_t devtype = dev->type;
	void * ptr = NULL;
#if defined (DEVICE_NVGPU_SUPPORT)
	if (devtype == OMP_DEVICE_NVGPU) {
		if (cudaErrorMemoryAllocation == cudaMalloc(&ptr, size)) {
			fprintf(stderr, "cudaMalloc error to allocate mem on device\n");
		}
	} else
#endif
	if (devtype == OMP_DEVICE_THSIM) {
		ptr = malloc(size);
	} else {
		fprintf(stderr, "device type is not supported for this call\n");
		abort();
	}
	return ptr;
}

void omp_map_free_dev(omp_device_t * dev, void * ptr) {
	omp_device_type_t devtype = dev->type;
#if defined (DEVICE_NVGPU_SUPPORT)
	if (devtype == OMP_DEVICE_NVGPU) {
	    cudaError_t result = cudaFree(ptr);
	    devcall_assert(result);
	} else
#endif
	if (devtype == OMP_DEVICE_THSIM) {
		free(ptr);
	} else {
		fprintf(stderr, "device type is not supported for this call\n");
		abort();
	}
}

void omp_map_memcpy_to(void * dst, omp_device_t * dstdev, const void * src, long size) {
	omp_device_type_t devtype = dstdev->type;
#if defined (DEVICE_NVGPU_SUPPORT)
	if (devtype == OMP_DEVICE_NVGPU) {
	    cudaError_t result;
	    result = cudaMemcpy((void *)dst,(const void *)src,size, cudaMemcpyHostToDevice);
	    devcall_assert(result);
	} else
#endif
	if (devtype == OMP_DEVICE_THSIM) {
		memcpy((void *)dst,(const void *)src,size);
	} else {
		fprintf(stderr, "device type is not supported for this call\n");
		abort();
	}
}

void omp_map_memcpy_to_async(void * dst, omp_device_t * dstdev, const void * src, long size, omp_dev_stream_t * stream) {
	omp_device_type_t devtype = dstdev->type;
#if defined (DEVICE_NVGPU_SUPPORT)
	if (devtype == OMP_DEVICE_NVGPU) {
		cudaError_t result;
		result = cudaMemcpyAsync((void *)dst,(const void *)src,size, cudaMemcpyHostToDevice, stream->systream.cudaStream);
		devcall_assert(result);
	} else
#endif
	if (devtype == OMP_DEVICE_THSIM) {
//		fprintf(stderr, "no async call support, use sync memcpy call\n");
		memcpy((void *)dst,(const void *)src,size);
	} else {
		fprintf(stderr, "device type is not supported for this call\n");
		abort();
	}
}

void omp_map_memcpy_from(void * dst, const void * src, omp_device_t * srcdev, long size) {
	omp_device_type_t devtype = srcdev->type;
#if defined (DEVICE_NVGPU_SUPPORT)
	if (devtype == OMP_DEVICE_NVGPU) {
		cudaError_t result;
	    result = cudaMemcpy((void *)dst,(const void *)src,size, cudaMemcpyDeviceToHost);
		devcall_assert(result);
	} else
#endif
	if (devtype == OMP_DEVICE_THSIM) {
		memcpy((void *)dst,(const void *)src,size);
	} else {
		fprintf(stderr, "device type is not supported for this call\n");
		abort();
	}
}

/**
 *  device to host, async */
void omp_map_memcpy_from_async(void * dst, const void * src, omp_device_t * srcdev, long size, omp_dev_stream_t * stream) {
	omp_device_type_t devtype = srcdev->type;
#if defined (DEVICE_NVGPU_SUPPORT)
	if (devtype == OMP_DEVICE_NVGPU) {
		cudaError_t result;
		result = cudaMemcpyAsync((void *)dst,(const void *)src,size, cudaMemcpyDeviceToHost, stream->systream.cudaStream);
		devcall_assert(result);
	} else
#endif
	if (devtype == OMP_DEVICE_THSIM) {
//		fprintf(stderr, "no async call support, use sync memcpy call\n");
		memcpy((void *)dst,(const void *)src,size);
//		printf("memcpy from: dest: %X, src: %X, size: %d\n", map->map_buffer, map->map_dev_ptr);
	} else {
		fprintf(stderr, "device type is not supported for this call\n");
		abort();
	}
}

/**
 * this should be calling from src for NGVPU implementation
 */
int omp_map_enable_memcpy_DeviceToDevice(omp_device_t * dstdev, omp_device_t * srcdev) {
	omp_device_type_t dst_devtype = dstdev->type;
	omp_device_type_t src_devtype = srcdev->type;

#if defined (DEVICE_NVGPU_SUPPORT)
	if (dst_devtype == OMP_DEVICE_NVGPU && src_devtype == OMP_DEVICE_NVGPU) {
		int can_access = 0;
		cudaError_t result;
		result = cudaDeviceCanAccessPeer(&can_access, srcdev->sysid, dstdev->sysid);
		devcall_assert(result);
		if (can_access) {
			result = cudaDeviceEnablePeerAccess(dstdev->sysid, 0);
		    if(result != cudaErrorPeerAccessAlreadyEnabled) {
		    	return 0;
		    } else return 1;
		} else return 1;
	} else
#endif
	if (dst_devtype == OMP_DEVICE_THSIM && src_devtype == OMP_DEVICE_THSIM) {
#if defined EXPERIMENT_RELAY_BUFFER_FOR_HALO_EXCHANGE
		return 0;
#else
		return 1;
#endif
	} else {
		fprintf(stderr, "device type is not supported for this call, currently we only support p2p copy between GPU-GPU and TH-TH\n");
		abort();
	}
	return 0;
}

void omp_map_memcpy_DeviceToDevice(void * dst, omp_device_t * dstdev, void * src, omp_device_t * srcdev, int size) {
	omp_device_type_t dst_devtype = dstdev->type;
	omp_device_type_t src_devtype = srcdev->type;

#if defined (DEVICE_NVGPU_SUPPORT)
	if (dst_devtype == OMP_DEVICE_NVGPU && src_devtype == OMP_DEVICE_NVGPU) {
		cudaError_t result;
	    result = cudaMemcpy((void *)dst,(const void *)src,size, cudaMemcpyDeviceToDevice);
//		result = cudaMemcpyPeer(dst, dstdev->sysid, src, srcdev->sysid, size);
		devcall_assert(result);
	} else
#endif
	if (dst_devtype == OMP_DEVICE_THSIM && src_devtype == OMP_DEVICE_THSIM) {
		memcpy((void *)dst, (const void *)src, size);
	} else {
		fprintf(stderr, "device type is not supported for this call, currently we only support p2p copy between GPU-GPU and TH-TH\n");
		abort();
	}
}

/** it is a push operation, i.e. src push data to dst */
void omp_map_memcpy_DeviceToDeviceAsync(void * dst, omp_device_t * dstdev, void * src, omp_device_t * srcdev, int size, omp_dev_stream_t * srcstream) {
	omp_device_type_t dst_devtype = dstdev->type;
	omp_device_type_t src_devtype = srcdev->type;

#if defined (DEVICE_NVGPU_SUPPORT)
	if (dst_devtype == OMP_DEVICE_NVGPU && src_devtype == OMP_DEVICE_NVGPU) {
		cudaError_t result;
	    result = cudaMemcpyAsync((void *)dst,(const void *)src,size, cudaMemcpyDeviceToDevice,srcstream->systream.cudaStream);
	    //result = cudaMemcpyPeerAsync(dst, dstdev->sysid, src, srcdev->sysid, size, srcstream->systream.cudaStream);

		devcall_assert(result);
	} else
#endif
	if (dst_devtype == OMP_DEVICE_THSIM && src_devtype == OMP_DEVICE_THSIM) {
		memcpy((void *)dst, (const void *)src, size);
	} else {
		fprintf(stderr, "device type is not supported for this call, currently we only support p2p copy between GPU-GPU and TH-TH\n");
		abort();
	}
}

/* In the current implementation of the runtime, we will NOT use stream callback to do the timing and others such as reduction operation.
 * The reason is because CUDA use a driver thread to handle callback, which become not necessnary since we have a dedicated helper thread
 * for each GPU and the helper thread could do this kind of work
 */
#if defined (DEVICE_NVGPU_SUPPORT)

#if 0
void xomp_beyond_block_reduction_float_stream_callback(cudaStream_t stream,  cudaError_t status, void*  userData ) {
	omp_reduction_float_t * rdata = (omp_reduction_float_t*)userData;
	float result = 0.0;
	int i;
	for (i=0; i<rdata->num; i++)
		result += rdata->input[i];
	rdata->result = result;
}
#endif

void omp_stream_host_timer_callback(cudaStream_t stream,  cudaError_t status, void*  userData ) {
	double * time = (double*)userData;
	*time = read_timer_ms();
}
#endif

void omp_stream_create(omp_device_t * d, omp_dev_stream_t * stream, int using_dev_default) {
	stream->dev = d;
	int i;

#if defined (DEVICE_NVGPU_SUPPORT)
	if (d->type == OMP_DEVICE_NVGPU) {
		if (using_dev_default) stream->systream.cudaStream = 0;
		else {
			cudaError_t result;
			result = cudaStreamCreate(&stream->systream.cudaStream);
			devcall_assert(result);
		}
	} else
#endif
	if (d->type == OMP_DEVICE_THSIM){
		/* do nothing */
	} else {

	}
}

/**
 * sync device by syncing the stream so all the pending calls the stream are completed
 *
 * if destroy_stream != 0; the stream will be destroyed.
 */
void omp_stream_sync(omp_dev_stream_t *st) {
	omp_device_type_t devtype = st->dev->type;
#if defined (DEVICE_NVGPU_SUPPORT)
	if (devtype == OMP_DEVICE_NVGPU) {
		cudaError_t result;
		result = cudaStreamSynchronize(st->systream.cudaStream);
		devcall_assert(result);
	}
#else
#endif
}

void omp_stream_destroy(omp_dev_stream_t * st) {
	omp_device_type_t devtype = st->dev->type;
#if defined (DEVICE_NVGPU_SUPPORT)
	if (devtype == OMP_DEVICE_NVGPU && st->systream.cudaStream != 0) {
		cudaError_t result;
		result = cudaStreamDestroy(st->systream.cudaStream);
		devcall_assert(result);
	}
#else
#endif
}

/* the event msg has limited length defined by OMP_EVENT_MSG_LENGTH macro, additional char will be cut off */
void omp_event_init(omp_event_t * ev, omp_device_t * dev, omp_event_record_method_t record_method) {
	ev->dev = dev;
	ev->record_method = record_method;
	omp_device_type_t devtype = dev->type;
	ev->count = 0;
	ev->recorded = 0;
	ev->elapsed_dev = ev->elapsed_host = 0.0;
	ev->event_name = NULL;
	ev->event_description[0] = '\0';
	if (record_method == OMP_EVENT_DEV_RECORD || record_method == OMP_EVENT_HOST_DEV_RECORD) {
#if defined (DEVICE_NVGPU_SUPPORT)
		if (devtype == OMP_DEVICE_NVGPU) {
			cudaError_t result;
			result = cudaEventCreateWithFlags(&ev->start_event_dev, cudaEventBlockingSync);
			devcall_assert(result);
			result = cudaEventCreateWithFlags(&ev->stop_event_dev, cudaEventBlockingSync);
			devcall_assert(result);
		} else
#endif
		if (devtype == OMP_DEVICE_THSIM || devtype == OMP_DEVICE_HOST) {
			/* do nothing */
		} else {
			fprintf(stderr, "other type of devices are not yet supported to init this event\n");
			abort();
		}
	}
	//omp_event_print(ev);
}

void omp_event_print(omp_event_t * ev) {
	printf("ev: %X, dev: %X, stream: %X, record method: %d, name: %s, description: %s\n", ev, ev->dev,
			ev->stream, ev->record_method, ev->event_name, ev->event_description);
}

void omp_event_record_start(omp_event_t * ev, omp_dev_stream_t * stream,  const char * event_name, const char * event_msg, ...) {
	if (stream != NULL && stream->dev != ev->dev) {
		fprintf(stderr, "stream and event are not compatible, they are from two different devices\n");
		abort();
	}
	ev->stream = stream;
	omp_event_record_method_t rm = ev->record_method;
	ev->event_name = event_name;
	va_list l;
	va_start(l, event_msg);
    vsnprintf(ev->event_description, OMP_EVENT_MSG_LENGTH, event_msg, l);
	va_end(l);

	//printf("omp_event_record_start: ev %X name: %s, dev: %X\n", ev, ev->event_name, ev->dev);
	omp_device_type_t devtype = ev->dev->type;

	if (rm == OMP_EVENT_DEV_RECORD || rm == OMP_EVENT_HOST_DEV_RECORD) {
#if defined (DEVICE_NVGPU_SUPPORT)
		if (devtype == OMP_DEVICE_NVGPU) {
			cudaError_t result;
			result = cudaStreamAddCallback(stream->systream.cudaStream, omp_stream_host_timer_callback, &ev->start_time_dev, 0);
			result = cudaEventRecord(ev->start_event_dev, stream->systream.cudaStream);
			devcall_assert(result);
		} else
#endif
		if (devtype == OMP_DEVICE_THSIM || devtype == OMP_DEVICE_HOST) {
			ev->start_time_dev = read_timer_ms();
		} else {
			fprintf(stderr, "other type of devices are not yet supported to start event recording\n");
		}
	}

	if (rm == OMP_EVENT_HOST_RECORD || rm == OMP_EVENT_HOST_DEV_RECORD) {
		ev->start_time_host = read_timer_ms();
	}
}

void omp_event_record_stop(omp_event_t * ev) {
	omp_dev_stream_t * stream = ev->stream;
	omp_event_record_method_t record_method = ev->record_method;
	omp_device_type_t devtype = ev->dev->type;
	if (record_method == OMP_EVENT_DEV_RECORD || record_method == OMP_EVENT_HOST_DEV_RECORD) {
#if defined (DEVICE_NVGPU_SUPPORT)
		if (devtype == OMP_DEVICE_NVGPU) {
			cudaError_t result;
			result = cudaStreamAddCallback(stream->systream.cudaStream, omp_stream_host_timer_callback, &ev->stop_time_dev, 0);
			result = cudaEventRecord(ev->stop_event_dev, stream->systream.cudaStream);
			devcall_assert(result);
		} else
#endif
		if (devtype == OMP_DEVICE_THSIM) {
			ev->stop_time_dev = read_timer_ms();

		} else {
			fprintf(stderr, "other type of devices are not yet supported to stop event record\n");
		}
	}

	if (record_method == OMP_EVENT_HOST_RECORD || record_method == OMP_EVENT_HOST_DEV_RECORD) {
		ev->stop_time_host = read_timer_ms();
	}
	ev->recorded = 1;
}


static double omp_event_elapsed_ms_dev(omp_event_t * ev) {
	omp_device_type_t devtype = ev->dev->type;
	float elapsed = -1.0;
	double elapsed1 = -1.0;
#if defined (DEVICE_NVGPU_SUPPORT)
	if (devtype == OMP_DEVICE_NVGPU) {
		elapsed1 = ev->stop_time_dev - ev->start_time_dev;
		cudaError_t result;
		result = cudaEventSynchronize(ev->start_event_dev);
		devcall_assert(result);
		result = cudaEventSynchronize(ev->stop_event_dev);
		devcall_assert(result);
		result = cudaEventElapsedTime(&elapsed, ev->start_event_dev, ev->stop_event_dev);
		devcall_assert(result);
		//printf("timing difference, callback: %f, event: %f\n", elapsed1, elapsed);
	} else
#endif
	if (devtype == OMP_DEVICE_THSIM) {
		elapsed = ev->stop_time_dev - ev->start_time_dev;
	} else {
		fprintf(stderr, "other type of devices are not yet supported to calculate elapsed\n");
	}
	//printf("dev event: start: %f, stop: %f, elapsed: %f (%f)\n", ev->start_time_dev, ev->stop_time_dev, elapsed, elapsed1);

	return elapsed;
}

static double omp_event_elapsed_ms_host(omp_event_t * ev) {
	double elapsed = ev->stop_time_host - ev->start_time_host;
	//printf("host event: start: %f, stop: %f, elapsed: %f\n", ev->start_time_host, ev->stop_time_host, elapsed);
	return elapsed;
}

/**
 * Computes the elapsed time between two events (in milliseconds with a resolution of around 0.5 microseconds).
 */
void omp_event_elapsed_ms(omp_event_t * ev) {
	if (!ev->recorded) return;
	omp_event_record_method_t record_method = ev->record_method;
	omp_device_type_t devtype = ev->dev->type;
	if (record_method == OMP_EVENT_DEV_RECORD || record_method == OMP_EVENT_HOST_DEV_RECORD) {
		ev->elapsed_dev = omp_event_elapsed_ms_dev(ev);
	}
	if (record_method == OMP_EVENT_HOST_RECORD || record_method == OMP_EVENT_HOST_DEV_RECORD) {
		ev->elapsed_host = omp_event_elapsed_ms_host(ev);
	}
	ev->recorded = 0;
}

void omp_event_accumulate_elapsed_ms(omp_event_t * ev) {
	if (!ev->recorded) return;
	omp_event_record_method_t record_method = ev->record_method;
	omp_device_type_t devtype = ev->dev->type;
	if (record_method == OMP_EVENT_DEV_RECORD || record_method == OMP_EVENT_HOST_DEV_RECORD) {
		ev->elapsed_dev += omp_event_elapsed_ms_dev(ev);
	}
	if (record_method == OMP_EVENT_HOST_RECORD || record_method == OMP_EVENT_HOST_DEV_RECORD) {
		ev->elapsed_host += omp_event_elapsed_ms_host(ev);
	}
	ev->count++;
	ev->recorded = 0;
}

int omp_get_max_threads_per_team(omp_device_t * dev) {
	omp_device_type_t devtype = dev->type;
#if defined (DEVICE_NVGPU_SUPPORT)
	if (devtype == OMP_DEVICE_NVGPU) {
		return 	((struct cudaDeviceProp*)dev->dev_properties)->maxThreadsPerBlock;
	} else
#endif
	if (devtype == OMP_DEVICE_THSIM) {
		return 1;
	} else {
	}
	return 0;
}

int omp_get_optimal_threads_per_team(omp_device_t * dev) {
	int max = omp_get_max_threads_per_team(dev);
	if (max == 1) return 1;
	else return max/2;
}

/**
 * so far we only do 1D, the first dimension
 */
int omp_get_max_teams_per_league(omp_device_t * dev) {
	omp_device_type_t devtype = dev->type;
#if defined (DEVICE_NVGPU_SUPPORT)
	if (devtype == OMP_DEVICE_NVGPU) {
		return 	((struct cudaDeviceProp*)dev->dev_properties)->maxGridSize[0];
	} else
#endif
	if (devtype == OMP_DEVICE_THSIM) {
		return 1;
	} else {
	}
	return 0;
}

int omp_get_optimal_teams_per_league(omp_device_t * dev, int threads_per_team, int total) {
	int teams_per_league = (total + threads_per_team - 1) / threads_per_team;
	int max_teams_per_league = omp_get_max_teams_per_league(dev);
	if (teams_per_league > max_teams_per_league) return max_teams_per_league;
	else return teams_per_league;
}

