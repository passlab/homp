#ifndef CUDA_EXTRA_H_
#define CUDA_EXTRA_H_
/*
 * cuda_extra.h
 *
 *  Created on: Feb 5, 2011
 *      Author: yy8
 */

#include "hc.h"

/* the common extra for both CUDA_kernel_extra and CUDA_memcpy_extra
 */
typedef struct CUDA_extra {
	hc_async_extra_t ex; /* the platform-independent part */
	cudaStream_t stream;
} CUDA_extra_t;

/* an asynchronous CUDA kernel launching extra, see hc.h for hc_async_extra */
typedef struct CUDA_kernel_extra {
	CUDA_extra_t cuex;
	int shared_size;
	struct dim3 threadsPerBlock;
	struct dim3 blocksPerGrid;
} CUDA_kernel_extra_t;

/* the extra specifically for CUDA memcpy and using stream
 */
typedef struct CUDA_memcpy_extra {
	CUDA_extra_t cuex;
	place_t * spl; /* the source place */
        void * sptr; /* the ptr to the src memory */
	place_t * dpl; /* the dest place */
        void * dptr; /* the ptr to the dest memory */
	int size; /* the size of the memory */
} CUDA_memcpy_extra_t;

#ifdef __cplusplus
extern "C" {
#endif
extern int hc_update_cuda_extra(hc_workerState * ws, hc_async_extra_t * extra);
extern short hc_set_current_device(hc_workerState * ws, place_t * devpl);
extern short hc_unset_current_device(hc_workerState * ws, place_t * devpl);
extern CUDA_kernel_extra_t * hc_init_cuda_kernel_extra(hc_workerState * ws, place_t * nvgpu_pl, int * blockDim, int *threadDim, int shared_size);
extern CUDA_kernel_extra_t * hc_init_cuda_kernel_extra_autodim(hc_workerState * ws, place_t * nvgpu_pl, int dim1, int dim2, int dim3, int shared_size);
#ifdef __cplusplus
}
#endif

#endif /* CUDA_EXTRA_H_ */
