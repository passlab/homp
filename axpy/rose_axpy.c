// Experimental test input for Accelerator directives
// Liao 1/15/2013
/* the following headers crashed ROSE on coil because of stddef.h not found
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
*/
#include "axpy.h"
#define VEC_LEN 1024000 //use a fixed number for now
/* zero out the entire vector */
void axpy(REAL* x, REAL* y,  long n, REAL a) {
  long i;
  for (i = 0; i < n; ++i) {
    y[i] += a * x[i];
//	printf("x[%d]: %f, y[%d]: %f\n", i, x[i], i, y[i]);
  }
}

void zero(REAL *A, long n)
{
   long i;
  for (i = 0; i < n; i++) {
    A[i] = 0.0;
  }
}
/* initialize a vector with random floating point numbers */

void init(REAL *A, long n)
{
   long i;
  for (i = 0; i < n; i++) {
    A[i] = ((REAL )(drand48()));
  }
}

REAL check(REAL *A,REAL *B, long n)
{
   long i;
  REAL sum = 0.0;
  for (i = 0; i < n; i++) {
//	printf("A[%d]: %f, B[%d]: %f\n", i, A[i], i, B[i]);
    sum += fabs(A[i] - B[i]);
  }
  return sum;
}

int main(int argc,char *argv[])
{
  int status = 0;
  omp_init_devices();
  long n;
  REAL *y;
  REAL *y_ompacc;
  REAL *x;
  REAL a = 123.456;
//  n = 1024*1024*1024; // too large for tux268
  n = 500000;
  if (argc >= 2) 
    n = atoi(argv[1]);
  printf("n = %d\n", n);  
  y = ((REAL *)(malloc((n * sizeof(REAL )))));
  y_ompacc = ((REAL *)(malloc((n * sizeof(REAL )))));
  x = ((REAL *)(malloc((n * sizeof(REAL )))));
  srand48(1 << 12);
  init(x,n);
  init(y,n);
  memcpy(y_ompacc,y,(n * sizeof(REAL )));
  REAL omp_time = read_timer_ms();
  //axpy(x,y,n,a);
  omp_time = (read_timer_ms() - omp_time);
  REAL ompacc_time = axpy_ompacc_mdev_v2(x,y_ompacc,n,a);
  omp_fini_devices();

  printf("axpy(%d): checksum: %g; time(ms):\tSerial\t\tOMPACC(%d devices)\n",n,check(y,y_ompacc,n),omp_get_num_active_devices());
  printf("\t\t\t\t\t%4f\t%4f\n",omp_time,ompacc_time);
  free(y);
  free(y_ompacc);
  free(x);
  return 0;
}
