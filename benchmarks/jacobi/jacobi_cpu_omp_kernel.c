#include <homp.h>
#define REAL float

void jacobi_cpu_omp_wrapper2(omp_offloading_t *off, long n,long m,REAL *u,REAL *uold,long uold_n, long uold_m, int uold_0_offset, int uold_1_offset)
{
    int i, j;
    int num_omp_threads = off->dev->num_cores;
#pragma omp parallel for private(j,i) shared(m,n,uold,u,uold_0_offset,uold_1_offset, uold_m) num_threads(num_omp_threads)
    for (i=0; i < n; i++) {
        /* since uold has halo region, here we need to adjust index to reflect the new offset */
        REAL * tmp_uold = &uold[(i + uold_0_offset) * uold_m + uold_1_offset];
        REAL * tmp_u = &u[i*m];
        #pragma omp simd
        for (j = 0; j < m; j++) {
            *tmp_uold = *tmp_u;
            tmp_uold ++;
            tmp_u++;
        }
    }
}


void jacobi_cpu_omp_wrapper1(omp_offloading_t *off, long n,long m,REAL omega,REAL ax,REAL ay,REAL b,REAL *u,REAL *f, \
 REAL *uold, long uold_m, int uold_0_offset, int uold_1_offset, int i_start, int j_start, REAL *error) {
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
    REAL er = 0.0;
#pragma omp parallel for private(j,i) reduction(+:er) num_threads(num_omp_threads)
    for (i = i_start; i < n; i++) {
        REAL * tmp_uold = &uold[(i + uold_0_offset)* uold_m + uold_1_offset+j_start];
        REAL * tmp_f = &f[i*m+j_start];
        REAL * tmp_u = &u[i*m+j_start];
        #pragma omp simd
        for (j = j_start; j < m; j++) {
            REAL resid = (ax * (tmp_uold[uold_m] + tmp_uold[-uold_m]) + ay * (tmp_uold[-1] * tmp_uold[1]) + b * tmp_uold[0] - *tmp_f)/b;

            *tmp_u = *tmp_uold = omega * resid;
            er = er + resid * resid;

            tmp_uold++;
            tmp_f++;
            tmp_u++;
        }
    }

    *error = er;
}