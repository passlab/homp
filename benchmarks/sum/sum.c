#include <assert.h>
#include "sum.h"

#define VEC_LEN 1024000 //use a fixed number for now

/* zero out the entire vector */
REAL sum(long n, REAL *x, REAL *y) {
    long i;
    REAL result = 0.0;
    for (i = 0; i < n; ++i) {
        result += y[i] * x[i];
	//printf("host: x[%d]: %f, y[%d]: %f\n", i, x[i], i, y[i]);
    }
    return result;
}

void zero(REAL *A, long n) {
    long i;
    for (i = 0; i < n; i++) {
        A[i] = 0.0;
    }
}

/* initialize a vector with random floating point numbers */

void init(REAL *A, long n) {
    long i;
    for (i = 0; i < n; i++) {
        A[i] = ((REAL) (drand48()));
    }
}

REAL check(REAL *A, REAL *B, long n) {
    long i;
    REAL sum = 0.0;
    for (i = 0; i < n; i++) {
//	printf("A[%d]: %f, B[%d]: %f\n", i, A[i], i, B[i]);
	REAL diff = fabs(A[i] - B[i]);
//	if (diff > 1.0) printf("diff at %d: %f\n", i, diff);
        sum += fabs(A[i] - B[i]);
    }
    return sum;
}

extern int sum_mdev_v;

int main(int argc, char *argv[]) {
    omp_init_devices();
    long n;
    REAL *y;
    REAL *x;
    REAL mdev_sum;
    n = 500000;
    printf("usage: sum [n], default n: %d\n", n);
    if (argc >= 2)
        n = atoi(argv[1]);
    y = ((REAL *) (omp_unified_malloc((n * sizeof(REAL)))));
    x = ((REAL *) (omp_unified_malloc((n * sizeof(REAL)))));
    srand48(1 << 12);
    init(x, n);
    init(y, n);
    REAL omp_time = read_timer_ms();
    // reference serial execution for error checking  
    REAL serial_sum = sum(n, x, y);
    omp_time = (read_timer_ms() - omp_time);

    int i;
    /* run only on NVGPU */
    int num_active_devs = omp_get_num_active_devices();
    int targets[num_active_devs];
    int num_targets;
    double ompacc_time;
#if 0
    /* one HOSTCPU */
    num_targets = omp_get_devices(OMP_DEVICE_HOSTCPU, targets, 1);
    ompacc_time = sum_ompacc_mdev(num_targets, targets, n, x, y, &mdev_sum);
  
    /* one NVGPU */
    num_targets = omp_get_devices(OMP_DEVICE_NVGPU, targets, 1);
    ompacc_time = sum_ompacc_mdev(num_targets, targets, n, x, y, &mdev_sum);

    /* two NVGPU */
    num_targets = omp_get_devices(OMP_DEVICE_NVGPU, targets, 2);
    ompacc_time = sum_ompacc_mdev(num_targets, targets, n, x, y, &mdev_sum);
    
    /* four NVGPU */
    num_targets = omp_get_devices(OMP_DEVICE_NVGPU, targets, 4);
    ompacc_time = sum_ompacc_mdev(num_targets, targets, n, x, y, &mdev_sum);

    /* one ITLMIC */
    num_targets = omp_get_devices(OMP_DEVICE_ITLMIC, targets, 1);
    ompacc_time = sum_ompacc_mdev(num_targets, targets, n, x, y, &mdev_sum);

    /* two ITLMIC */
    num_targets = omp_get_devices(OMP_DEVICE_ITLMIC, targets, 2);
    ompacc_time = sum_ompacc_mdev(num_targets, targets, n, x, y, &mdev_sum);

    /* one HOSTCPU and one NVGPU */
    num_targets = omp_get_devices(OMP_DEVICE_HOSTCPU, targets, 1);
    num_targets += omp_get_devices(OMP_DEVICE_NVGPU, targets+num_targets, 1);
    ompacc_time = sum_ompacc_mdev(num_targets, targets, n, x, y, &mdev_sum);

    /* one HOSTCPU and one ITLMIC */
    num_targets = omp_get_devices(OMP_DEVICE_HOSTCPU, targets, 1);
    num_targets += omp_get_devices(OMP_DEVICE_ITLMIC, targets+num_targets, 1);
    ompacc_time = sum_ompacc_mdev(num_targets, targets, n, x, y, &mdev_sum);

    /* one NVGPU and one ITLMIC */
    num_targets = omp_get_devices(OMP_DEVICE_NVGPU, targets, 1);
    num_targets += omp_get_devices(OMP_DEVICE_ITLMIC, targets+num_targets, 1);
    ompacc_time = sum_ompacc_mdev(num_targets, targets, n, x, y, &mdev_sum);

    /* one HOSTCPU and two NVGPU */
    num_targets = omp_get_devices(OMP_DEVICE_HOSTCPU, targets, 1);
    num_targets += omp_get_devices(OMP_DEVICE_NVGPU, targets+num_targets, 2);
    ompacc_time = sum_ompacc_mdev(num_targets, targets, n, x, y, &mdev_sum);

    /* one HOSTCPU and four NVGPU */
    num_targets = omp_get_devices(OMP_DEVICE_HOSTCPU, targets, 1);
    num_targets += omp_get_devices(OMP_DEVICE_NVGPU, targets+num_targets, 4);
    ompacc_time = sum_ompacc_mdev(num_targets, targets, n, x, y, &mdev_sum);

    /* one HOSTCPU and two ITLMIC */
    num_targets = omp_get_devices(OMP_DEVICE_HOSTCPU, targets, 1);
    num_targets += omp_get_devices(OMP_DEVICE_ITLMIC, targets+num_targets, 2);
    ompacc_time = sum_ompacc_mdev(num_targets, targets, n, x, y, &mdev_sum);

    /* two NVGPU and two ITLMIC */
    //num_targets = omp_get_devices(OMP_DEVICE_NVGPU, targets, 2);
    //num_targets += omp_get_devices(OMP_DEVICE_ITLMIC, targets+num_targets, 2);
    //ompacc_time = sum_ompacc_mdev(num_targets, targets, n, x, y, &mdev_sum);

    /* four NVGPU and two ITLMIC */
    num_targets = omp_get_devices(OMP_DEVICE_NVGPU, targets, 4);
    num_targets += omp_get_devices(OMP_DEVICE_ITLMIC, targets+num_targets, 2);
    ompacc_time = sum_ompacc_mdev(num_targets, targets, n, x, y, &mdev_sum);

    /* one CPU, two NVGPU and two ITLMIC */
    //num_targets = omp_get_devices(OMP_DEVICE_HOSTCPU, targets, 1);
    //num_targets += omp_get_devices(OMP_DEVICE_NVGPU, targets+num_targets, 2);
    //num_targets += omp_get_devices(OMP_DEVICE_ITLMIC, targets+num_targets, 2);
    //ompacc_time = sum_ompacc_mdev(num_targets, targets, n, x, y, &mdev_sum);

    /* one CPU, four NVGPU and two ITLMIC */
    num_targets = omp_get_devices(OMP_DEVICE_HOSTCPU, targets, 1);
    num_targets += omp_get_devices(OMP_DEVICE_NVGPU, targets+num_targets, 4);
    num_targets += omp_get_devices(OMP_DEVICE_ITLMIC, targets+num_targets, 2);
    ompacc_time = sum_ompacc_mdev(num_targets, targets, n, x, y, &mdev_sum);

#endif
    /* run on all devices */
    num_targets = num_active_devs;
    for (i=0;i<num_active_devs;i++) targets[i] = i;
    ompacc_time = sum_ompacc_mdev(num_targets, targets, n, x, y, &mdev_sum);

    omp_fini_devices();
    REAL cksm = serial_sum - mdev_sum;
    //printf("serial sum: %f, mdev sum: %f\n", serial_sum, mdev_sum);
    printf("sum(%d): checksum: %f; time(ms):\tSerial\t\tOMPACC(%d devices)\n", n, cksm, omp_get_num_active_devices());
    printf("\t\t\t\t\t\t%4f\t%4f\n", omp_time, ompacc_time);
    free(y);
    omp_unified_free(x);
    // I got 1.093e-09
    //assert (cksm< 1.0e-07);
    printf("usage: sum [n], default n: %d\n", n);
    return 0;
}
