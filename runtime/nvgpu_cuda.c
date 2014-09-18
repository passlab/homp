/*
 * NVGPU_pl.c
 *
 *  Created on: Feb 18, 2011
 *      Author: yy8
 */
#include "nvgpu_cuda.h"

short hc_nvgpu_place_init(hc_workerState * ws, place_t * pl) {
	pl->deques = NULL;
	pl->ndeques = 0;
	pl->workers = NULL;
	pl->extra.cuctxt = 0;

	CUresult status = cuInit(0);
	if ( CUDA_SUCCESS != status ) {
		LOG_FATAL(ws, "cuInit failed: %d on NVGPU_PLACE: dev: %d\n", status, pl->did);
		return status;
	}

	status = cuCtxCreate( &pl->extra.cuctxt, 0, pl->did);
	if (CUDA_SUCCESS != status) {
		LOG_FATAL(ws, "cuCtxCreate failed: %d on NVGPU_PLACE: dev: %d, CUContext: %X\n", status, pl->did, &pl->extra.cuctxt);
		return status;
	}

	// Here we must release the CUDA context from the thread context
	status = cuCtxPopCurrent(NULL);
	if (CUDA_SUCCESS != status) {
		LOG_FATAL(ws, "cuCtxPopCurrent failed %d on NVGPU_PLACE: %d, CUContext: %X\n", status, pl->did, &pl->extra.cuctxt);
		return status;
	}
	return 0;
}

short hc_nvgpu_place_cleanup(hc_workerState * ws, place_t * pl) {
	CUresult status = cuCtxPushCurrent( pl->extra.cuctxt );
	if ( CUDA_SUCCESS != status ) {
		LOG_FATAL(ws, "cuCtxPushCurrent failed on NVGPU_PLACE: dev: %d, CUContext: %X\n", pl->did, &pl->extra.cuctxt);
		return status;
	}

	status = cuCtxDestroy(pl->extra.cuctxt);
	if (CUDA_SUCCESS != status) {
		LOG_FATAL(ws, "cuCtxDestroy failed on NVGPU_PLACE: dev: %d, CUContext: %X\n", pl->did, &pl->extra.cuctxt);
		return status;
	}

	return 0;
}

/*
 * Update the CUDA kernel transactions by querying the stream
 *
 * This is not thread-safe, no need to be because we only allow one worker to do that in the runtime
 * 
 * If this operation is complete, we cleanup resources and reclaim the memory
 */
int hc_update_cuda_extra(hc_workerState * ws, hc_async_extra_t * extra) {
	CUDA_extra_t * cu_extra = (CUDA_extra_t*) extra;
	int status =  cu_extra->ex.status;
	if (status == ASYNC_EXTRA_ACTIVE) {
		hc_set_current_device(ws, extra->pl);
		cudaError_t complete = cudaStreamQuery(cu_extra->stream);
		if (complete == cudaSuccess) {
			cudaStreamDestroy(cu_extra->stream);
			hc_unset_current_device(ws, extra->pl);
			HC_FREE(ws, extra);
			return ASYNC_EXTRA_COMPLETE;
		}
	}

	return status;
}

/* the follow couple function should be called with match */
short hc_set_current_device(hc_workerState * ws, place_t * devpl) {
	CUresult status = cuCtxPushCurrent(devpl->extra.cuctxt);
	if (CUDA_SUCCESS != status) {
		LOG_FATAL(ws, "cuCtxPushCurrent failed on NVGPU_PLACE: dev: %d, CUContext: %X\n", devpl->did, &devpl->extra.cuctxt);
	}
	return status;
}

short hc_unset_current_device(hc_workerState * ws, place_t * devpl) {
	CUresult status = cuCtxPopCurrent(NULL);
	if (CUDA_SUCCESS != status) {
		LOG_FATAL(ws, "cuCtxPopCurrent failed on NVGPU_PLACE: dev: %d, CUContext: %X\n", devpl->did, &devpl->extra.cuctxt);
	}
	return status;
}

/* initialize a CUDA extra and also set the current CUContext for the GPU device
 *
 */
CUDA_kernel_extra_t * hc_init_cuda_kernel_extra(hc_workerState * ws, place_t * nvgpu_pl, int * blockDim, int *threadDim, int shared_size) {
	CUDA_kernel_extra_t * extra = HC_MALLOC(ws, sizeof(CUDA_kernel_extra_t));
	extra->cuex.ex.nnext = NULL;
	extra->cuex.ex.type = DEV_ASYNC;
	extra->cuex.ex.status = ASYNC_EXTRA_ACTIVE;
	extra->cuex.ex.update_func = hc_update_cuda_extra;
	extra->cuex.ex.pl = nvgpu_pl;
	extra->shared_size = shared_size;
	hc_set_current_device(ws, nvgpu_pl);
	cudaError_t complete = cudaStreamCreate(&extra->cuex.stream);
	extra->threadsPerBlock.x = threadDim[0];
	extra->threadsPerBlock.y = threadDim[1];
	extra->threadsPerBlock.z = threadDim[2];
	extra->blocksPerGrid.x = blockDim[0];
	extra->blocksPerGrid.y = blockDim[1];
	extra->blocksPerGrid.z = blockDim[2];
	return extra;
}

CUDA_kernel_extra_t * hc_init_cuda_kernel_extra_autodim(hc_workerState * ws, place_t * nvgpu_pl, int dim1, int dim2, int dim3, int shared_size) {
	int threadDim[3];
	int blockDim[3];
	if (dim1 > 0 && dim2 > 0 && dim3 > 0) {
		threadDim[0] = 8;
		threadDim[1] = 8;
		threadDim[2] = 4;
		blockDim[0] = (dim1 + threadDim[0] - 1) / threadDim[0];
		blockDim[1] = (dim2 + threadDim[1] - 1) / threadDim[1];
		blockDim[2] = (dim3 + threadDim[2] - 1) / threadDim[2];
	} else if (dim1 > 0 && dim2 > 0) {
		threadDim[0] = 16;
		threadDim[1] = 16;
		threadDim[2] = 1;
		blockDim[0] = (dim1 + threadDim[0] - 1) / threadDim[0];
		blockDim[1] = (dim2 + threadDim[1] - 1) / threadDim[1];
		blockDim[2] = 1;
	} else if (dim1 > 0) {
		threadDim[0] = 256;
		threadDim[1] = 1;
		threadDim[2] = 1;
		blockDim[0] = (dim1 + threadDim[0] - 1) / threadDim[0];
		blockDim[1] = 1;
		blockDim[2] = 1;
	} else {
		LOG_FATAL(ws, "Invalid dimension for autodim: dim1: %d, dim2: %d, dim3: %d\n", dim1, dim2, dim3);
		return NULL;
	}

	return hc_init_cuda_kernel_extra(ws, nvgpu_pl, blockDim, threadDim, shared_size);
}

/*
 * Programmers' APIs for memory operation between places *
 *
 */

void * HC_MALLOC_PLACE_WS(hc_workerState * ws, place_t * pl, int size) {
	if (pl->type == NVGPU_PLACE) {
		CUresult status = cuCtxPushCurrent(pl->extra.cuctxt);
		if ( CUDA_SUCCESS != status ) {
			LOG_FATAL(ws, "cuCtxPushCurrent failed on NVGPU_PLACE: dev: %d, CUContext: %X\n", pl->did, &pl->extra.cuctxt);
			return NULL;
		}
		void * ptr;
		cudaError_t error = cudaMalloc( &ptr, size);
		if ( cudaSuccess != error ) {
		    LOG_FATAL(ws, "cudaAlloc failed %d on NVGPU_PLACE: %d, CUContext: %X\n", error, pl->did, &pl->extra.cuctxt);
		    return NULL;
		}

		// Here we must release the CUDA context from the thread context
		status = cuCtxPopCurrent( NULL );
		if ( CUDA_SUCCESS != status ) {
			LOG_FATAL(ws, "cuCtxPopCurrent failed %d on NVGPU_PLACE: %d, CUContext: %X\n", status, pl->did, &pl->extra.cuctxt);
			return NULL;
		}
		return ptr;
	} else { /* do the regular malloc */
		return HC_MALLOC(ws, size);
	}
}

void * HC_MALLOC_PLACE(place_t * pl, int size) {
	hc_workerState * ws = current_ws();
	return HC_MALLOC_PLACE_WS(ws, pl, size);
}

void * HC_MALLOC_LOCAL(int size) {
	hc_workerState * ws = current_ws();
	place_t * pl = ws->frame->pl;
	return HC_MALLOC_PLACE_WS(ws, pl, size);
}

void HC_FREE_PLACE_WS(hc_workerState * ws, place_t * pl, void * ptr) {
	if (pl->type == NVGPU_PLACE) {
		CUresult status = cuCtxPushCurrent(pl->extra.cuctxt);
		if (CUDA_SUCCESS != status) {
			LOG_FATAL(ws, "cuCtxPushCurrent failed on NVGPU_PLACE: dev: %d, CUContext: %X\n", pl->did, &pl->extra.cuctxt);
		}
		cudaError_t error = cudaFree(ptr);
		if ( cudaSuccess != error ) {
			LOG_FATAL(ws, "cudaFree failed %d on NVGPU_PLACE: %d, CUContext: %X\n", error, pl->did, &pl->extra.cuctxt);
		}

		// Here we must release the CUDA context from the thread context
		status = cuCtxPopCurrent(NULL);
		if (CUDA_SUCCESS != status) {
			LOG_FATAL(ws, "cuCtxPopCurrent failed %d on NVGPU_PLACE: %d, CUContext: %X\n", status, pl->did, &pl->extra.cuctxt);
		}
	} else { /* do the regular malloc */
		HC_FREE(ws, ptr);
	}
}

void HC_FREE_PLACE(place_t * pl, void * ptr) {
	hc_workerState * ws = current_ws();
	HC_FREE_PLACE_WS(ws, pl, ptr);
}

void HC_FREE_LOCAL(void * ptr) {
	hc_workerState * ws = current_ws();
	place_t * pl = ws->frame->pl;
	HC_FREE_PLACE_WS(ws, pl, ptr);
}

/* currently only allow CPU->GPU, GPU->CPU copy, not GPU-GPU copy */
short HC_MEMCPY_P2P_WS(hc_workerState * ws, place_t * dst_pl, void * dst, place_t * src_pl, void *src, int size) {
	LOG_INFO(ws, " memcpy between two places, src: %X, src_pl: %X; dst: %X, dst_pl: %X\n", src, src_pl, dst, dst_pl);
	if (src_pl->type == NVGPU_PLACE &&  dst_pl->type == NVGPU_PLACE) {
		LOG_FATAL(ws, " Cannot do memcpy between two GPU places, src(dev: %d), dst(dev: %d)\n", src_pl->did, dst_pl->did);
		return -1;
	}
	if (dst_pl->type == NVGPU_PLACE) { /* CPU->GPU */
		CUresult status = cuCtxPushCurrent(dst_pl->extra.cuctxt);
		if (CUDA_SUCCESS != status) {
			LOG_FATAL(ws, "cuCtxPushCurrent failed on NVGPU_PLACE: dev: %d, CUContext: %X\n", dst_pl->did, &dst_pl->extra.cuctxt);
			return status;
		}

		/* do the data copy */
		cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);

		if ( cudaSuccess != error ) {
			LOG_FATAL(ws, "cudaMemcpy(Async)(H2D) failed %d on NVGPU_PLACE: %d, CUContext: %X\n", error, dst_pl->did, &dst_pl->extra.cuctxt);
			return error;
		}

		// Here we must release the CUDA context from the thread context
		status = cuCtxPopCurrent(NULL);
		if (CUDA_SUCCESS != status) {
			LOG_FATAL(ws, "cuCtxPopCurrent failed %d on NVGPU_PLACE: %d, CUContext: %X\n", status, dst_pl->did, &dst_pl->extra.cuctxt);
			return status;
		}
	} else if (src_pl->type == NVGPU_PLACE) { /* GPU->CPU */
		CUresult status = cuCtxPushCurrent(src_pl->extra.cuctxt);
		if (CUDA_SUCCESS != status) {
			LOG_FATAL(ws, "cuCtxPushCurrent failed on NVGPU_PLACE: dev: %d, CUContext: %X\n", src_pl->did, src_pl->extra.cuctxt);
			return status;
		}

		/* do the data copy */
		cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);

		if (cudaSuccess != error) {
			LOG_FATAL(ws, "cudaMemcpy(D2H) failed %d on NVGPU_PLACE: %d, CUContext: %X\n", error, dst_pl->did, &dst_pl->extra.cuctxt);
			return error;
		}


		// Here we must release the CUDA context from the thread context
		status = cuCtxPopCurrent(NULL);
		if (CUDA_SUCCESS != status) {
			LOG_FATAL(ws, "cuCtxPopCurrent failed %d on NVGPU_PLACE: %d, CUContext: %X\n", status, src_pl->did, src_pl->extra.cuctxt);
			return status;
		}
	} else { /* CPU->CPU */
		memcpy(dst, src, size);
		return 0;
	}
}

/* currently only allow CPU->GPU, GPU->CPU copy, not GPU-GPU copy,
 * For GPU using CUDA, we use cudaMemcpyAsync and stream for the implementation. 
 * If a memory copy is not completed, a book-keeping object, i.e. CUDA_memcpy_extra_t, is created and added to 
 * the enclosing finish state, and the hc runtime will query to update the status 
 */
short HC_ASYNC_MEMCPY_P2P_WS(hc_workerState * ws, place_t * dst_pl, void * dst, place_t * src_pl, void *src, int size) {
	if (src_pl->type == NVGPU_PLACE &&  dst_pl->type == NVGPU_PLACE) {
		LOG_FATAL(ws, " Cannot do memcpy between two GPU places, src(dev: %d), dst(dev: %d)\n", src_pl->did, dst_pl->did);
		return -1;
	}
	if (dst_pl->type == NVGPU_PLACE) { /* CPU->GPU */
		CUresult status = cuCtxPushCurrent(dst_pl->extra.cuctxt);
		if (CUDA_SUCCESS != status) {
			LOG_FATAL(ws, "cuCtxPushCurrent failed on NVGPU_PLACE: dev: %d, CUContext: %X\n", dst_pl->did, &dst_pl->extra.cuctxt);
			return status;
		}

		/* do the data copy */
		cudaStream_t stream;
		cudaError_t error = cudaStreamCreate(&stream);
		error = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
		if ( cudaSuccess != error ) {
			LOG_FATAL(ws, "cudaMemcpyAsyncH2D failed %d on NVGPU_PLACE: %d, CUContext: %X\n", error, dst_pl->did, &dst_pl->extra.cuctxt);
			return error;
		}
		
		error = cudaStreamQuery(stream);
		if (error != cudaSuccess) { /* add the extra to the finish state extra list */
			CUDA_memcpy_extra_t * extra = HC_MALLOC(ws, sizeof(CUDA_memcpy_extra_t));
			extra->cuex.stream = stream;
			extra->cuex.ex.type = MEMCPY_ASYNC;
			extra->cuex.ex.status = ASYNC_EXTRA_ACTIVE;
			extra->cuex.ex.update_func = hc_update_cuda_extra;
			extra->cuex.ex.pl = dst_pl;
			extra->cuex.ex.nnext = NULL;
			extra->spl = src_pl;
			extra->sptr = src;
			extra->dpl = dst_pl;
			extra->dptr = dst;
			extra->size = size;
			/* now add to the finish state extra list */
			hc_add_extra_finishState(ws, (hc_async_extra_t*)extra);
		}

		// Here we must release the CUDA context from the thread context
		status = cuCtxPopCurrent(NULL);
		if (CUDA_SUCCESS != status) {
			LOG_FATAL(ws, "cuCtxPopCurrent failed %d on NVGPU_PLACE: %d, CUContext: %X\n", status, dst_pl->did, &dst_pl->extra.cuctxt);
		}
		return status;
	} else if (src_pl->type == NVGPU_PLACE) { /* GPU->CPU */
		CUresult status = cuCtxPushCurrent(src_pl->extra.cuctxt);
		if (CUDA_SUCCESS != status) {
			LOG_FATAL(ws, "cuCtxPushCurrent failed on NVGPU_PLACE: dev: %d, CUContext: %X\n", src_pl->did, src_pl->extra.cuctxt);
			return status;
		}
		
		/* do the data copy */
		cudaStream_t stream;
		cudaError_t error = cudaStreamCreate(&stream);
		error = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
		if ( cudaSuccess != error ) {
			LOG_FATAL(ws, "cudaMemcpyAsyncH2D failed %d on NVGPU_PLACE: %d, CUContext: %X\n", error, dst_pl->did, &dst_pl->extra.cuctxt);
			return error;
		}
		
		error = cudaStreamQuery(stream);
		if (error != cudaSuccess) { /* add the extra to the finish state extra list */
			CUDA_memcpy_extra_t * extra = HC_MALLOC(ws, sizeof(CUDA_memcpy_extra_t));
			extra->cuex.stream = stream;
			extra->cuex.ex.type = MEMCPY_ASYNC;
			extra->cuex.ex.status = ASYNC_EXTRA_ACTIVE;
			extra->cuex.ex.update_func = hc_update_cuda_extra;
			extra->cuex.ex.pl = src_pl;
			extra->cuex.ex.nnext = NULL;
			extra->spl = src_pl;
			extra->sptr = src;
			extra->dpl = dst_pl;
			extra->dptr = dst;
			extra->size = size;
			/* now add to the finish state extra list */
			hc_add_extra_finishState(ws, (hc_async_extra_t*)extra);
		}

		// Here we must release the CUDA context from the thread context
		status = cuCtxPopCurrent(NULL);
		if (CUDA_SUCCESS != status) {
			LOG_FATAL(ws, "cuCtxPopCurrent failed %d on NVGPU_PLACE: %d, CUContext: %X\n", status, src_pl->did, src_pl->extra.cuctxt);
		}
		return status;
	} else { /* CPU->CPU */
		memcpy(dst, src, size);
		return 0;
	}
}

/* blocked memory copy from place to place */
short HC_MEMCPY_P2P(place_t * dst_pl, void * dst, place_t * src_pl, void *src, int size) {
	hc_workerState * ws = current_ws();
	return HC_MEMCPY_P2P_WS(ws, dst_pl, dst, src_pl, src, size);
}

/* async memory copy from place to place */
short HC_ASYNC_MEMCPY_P2P(place_t * dst_pl, void * dst, place_t * src_pl, void *src, int size) {
	hc_workerState * ws = current_ws();
	return HC_ASYNC_MEMCPY_P2P_WS(ws, dst_pl, dst, src_pl, src, size);
}

/* blocked memory copy from current place to place */
short HC_MEMCPY_TO(place_t * dst_pl, void * dst, void *src, int size) {
	hc_workerState * ws = current_ws();
	place_t * src_pl = ws->frame->pl;
	return HC_MEMCPY_P2P_WS(ws, dst_pl, dst, src_pl, src, size);
}

/* blocked memory copy from remote place to current place */
short HC_MEMCPY_FROM(void * dst, place_t * src_pl, void *src, int size) {
	hc_workerState * ws = current_ws();
	return HC_MEMCPY_P2P_WS(ws, current_place(ws), dst, src_pl, src, size);
}

/* async memory copy from current place to place */
short HC_ASYNC_MEMCPY_2P(place_t * dst_pl, void * dst, void *src, int size) {
	hc_workerState * ws = current_ws();
	place_t * src_pl = ws->frame->pl;
	return HC_ASYNC_MEMCPY_P2P_WS(ws, dst_pl, dst, src_pl, src, size);
}
