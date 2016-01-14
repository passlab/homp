#include <homp.h>
#define REAL float

void jacobi_cpu_omp_wrapper2(omp_offloading_t *off, long n,long m,REAL *u,REAL *uold,long uold_m, int uold_0_offset, int uold_1_offset)
{
    int num_omp_threads = off->dev->num_cores;
#pragma omp parallel for private(j,i) shared(m,n,uold,u,uold_0_offset,uold_1_offset) num_threads(num_omp_threads)
    for (i=0; i < n; i++)
        for (j=0; j < m; j++) {
            /* since uold has halo region, here we need to adjust index to reflect the new offset */
            uold[i+uold_0_offset][j+uold_1_offset] = u[i][j];
        }
}


void jacobi_cpu_omp_wrapper1(omp_offloading_t *off, long n,long m,REAL omega,REAL ax,REAL ay,REAL b,REAL *_dev_u,REAL *_dev_f, \
 REAL *_dev_uold, long uold_m, int uold_0_offset, int uold_1_offset, int start_i, int start_j, REAL *_dev_per_block_error) {
    int num_omp_threads = off->dev->num_cores;
#if CORRECTNESS_CHECK
	    BEGIN_SERIALIZED_PRINTF(off->devseqid);
		printf("udev: dev: %d, %dX%d\n", off->devseqid, n, m);
		print_array_dev("udev", off->devseqid, "u",(REAL*)u, n, m);
		printf("uolddev: dev: %d, %dX%d\n", off->devseqid, uold_0_length, uold_1_length);
		print_array_dev("uolddev", off->devseqid, "uold",(REAL*)uold, uold_0_length, uold_1_length);
		printf("i_start: %d, j_start: %d, n: %d, m: %d, uold_0_offset: %d, uold_1_offset: %d\n", i_start, j_start, n, m, uold_0_offset, uold_1_offset);
		print_array_dev("f", off->devseqid, "f",(REAL*)f, map_f->map_dim[0], map_f->map_dim[1]);

		END_SERIALIZED_PRINTF();
#endif

    int i, j;
#pragma omp parallel for private(resid,j,i) reduction(+:error) num_threads(num_omp_threads)
    for (i = i_start; i < n; i++) {
        for (j = j_start; j < m; j++) {
            resid = (ax *
                     (uold[i - 1 + uold_0_offset][j + uold_1_offset] + uold[i + 1 + uold_0_offset][j + uold_1_offset]) +
                     ay *
                     (uold[i + uold_0_offset][j - 1 + uold_1_offset] + uold[i + uold_0_offset][j + 1 + uold_1_offset]) +
                     b * uold[i + uold_0_offset][j + uold_1_offset] - f[i][j]) / b;

            u[i][j] = uold[i + uold_0_offset][j + uold_1_offset] - omega * resid;
            error = error + resid * resid;
        }
    }
    iargs->error[off->devseqid] = error;
}