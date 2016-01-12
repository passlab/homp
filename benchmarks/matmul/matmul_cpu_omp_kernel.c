#include "matvec.h"
#include <homp.h>
#define REAL float

void matmul_cpu_omp_wrapper(omp_offloading_t *off, long i, long j,long k,float *A,float *B,float *C)
{
    long ii, jj, kk;
    //	omp_set_num_threads(off->dev->num_cores);
    //printf("%d cores on host\n", off->dev->num_cores);
//#pragma omp parallel for shared(A, B, C, i,j,k) private(ii, jj, kk)
    for (ii = 0; ii < i; ii++) {
        for (jj = 0; jj < j; jj++) {
            REAL sum = 0.0;
            for (kk = 0; kk < k; kk++) {
                sum += A[ii * k + kk] * B[kk * j + jj];
            }
            C[ii * j + jj] = sum;
        }
    }
}