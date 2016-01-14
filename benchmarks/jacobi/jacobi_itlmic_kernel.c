#include <offload.h>
#include <homp.h>
#include "jacobi.h"

void jacobi_itlmic_wrapper2(omp_offloading_t *off, long n,long m,REAL *u,REAL *uold,long uold_m, int uold_0_offset, int uold_1_offset)
{
    long i, j;

#pragma offload target(mic:off->dev->sysid)) in (u: length(0) alloc_if(0) free_if(0)) \
                            in (uold: length(0) alloc_if(0) free_if(0))
#pragma omp parallel for simd
    {
//#pragma omp parallel for private(j,i) shared(m,n,uold,u,uold_0_offset,uold_1_offset)
        for (i=0; i < n; i++)
            for (j=0; j < m; j++) {
                /* since uold has halo region, here we need to adjust index to reflect the new offset */
                uold[i+uold_0_offset][j+uold_1_offset] = u[i][j];
            }
    }
}

void jacobi_itlmic_wrapper1(omp_offloading_t *off, long n,long m,REAL omega,REAL ax,REAL ay,REAL b,REAL *u,REAL *f, \
 REAL *uold, long uold_m, int uold_0_offset, int uold_1_offset, int i_start, int j_start, REAL *_dev_per_block_error, REAL resid) {
    long i, j;

#pragma offload target(mic:off->dev->sysid) in (u: length(0) alloc_if(0) free_if(0)) \
                            in (uold: length(0) alloc_if(0) free_if(0)) \
                            in (f: length(0) alloc_if(0) free_if(0))
    {
#pragma omp parallel for simd
#pragma omp parallel for private(resid,j,i) reduction(+:error)
        for (i = i_start; i < n; i++) {
            for (j = j_start; j < m; j++) {
                resid = (ax * (uold[i - 1 + uold_0_offset][j + uold_1_offset] +
                               uold[i + 1 + uold_0_offset][j + uold_1_offset]) + ay * (uold[i + uold_0_offset][j - 1 +
                                                                                                                   uold_1_offset] +
                                                                                       uold[i + uold_0_offset][j + 1 +
                                                                                                                   uold_1_offset]) +
                         b * uold[i + uold_0_offset][j + uold_1_offset] - f[i][j]) / b;

                u[i][j] = uold[i + uold_0_offset][j + uold_1_offset] - omega * resid;
                error = error + resid * resid;
            }
        }
    }
}

void jacobi_nvgpu_cuda_wrapper1(omp_offloading_t *off, long n,long m,REAL omega,REAL ax,REAL ay,REAL b,REAL *_dev_u,REAL *_dev_f, \
 REAL *_dev_uold, long uold_m, int uold_0_offset, int uold_1_offset, int start_i, int start_j, REAL *_dev_per_block_error) {
    int threads_per_team = omp_get_optimal_threads_per_team(off->dev);
    int teams_per_league = omp_get_optimal_teams_per_league(off->dev, threads_per_team, n*m);

    /* for reduction operation */
    REAL * _dev_per_block_error = (REAL*)omp_map_malloc_dev(off->dev, teams_per_league * sizeof(REAL));
    //printf("dev: %d teams per league, err block mem: %X\n", teams_per_league, _dev_per_block_error);
    REAL _host_per_block_error[teams_per_league];
    //printf("%d device: original offset: %d, mapped_offset: %d, length: %d\n", __i__, offset_n, start_n, length_n);
    /* Launch CUDA kernel ... */
    /** since here we do the same mapping, so will reuse the _threads_per_block and _num_blocks */
    jacobi_nvgpu_cuda_kernel1<<<teams_per_league, threads_per_team,(threads_per_team * sizeof(REAL)),
            off->stream->systream.cudaStream>>>(n, m,
                    omega, ax, ay, b, (REAL*)u, (REAL*)f, (REAL*)uold,uold_1_length, uold_0_offset, uold_1_offset, i_start, j_start, _dev_per_block_error);

    /* copy back the results of reduction in blocks */
    //printf("copy back reduced error: %X <-- %X\n", _host_per_block_error, _dev_per_block_error);
    omp_map_memcpy_from_async(_host_per_block_error, _dev_per_block_error, off->dev, sizeof(REAL)*teams_per_league, off->stream);
    omp_stream_sync(off->stream);

    iargs->error[off->devseqid] = xomp_beyond_block_reduction_float(_host_per_block_error, teams_per_league, XOMP_REDUCTION_PLUS);
    //cudaStreamAddCallback(__dev_stream__[__i__].systream.cudaStream, xomp_beyond_block_reduction_float_stream_callback, args, 0);
    omp_map_free_dev(off->dev, _dev_per_block_error);
}