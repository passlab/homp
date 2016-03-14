#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include "homp.h"
#include "bm2d.h"

/* 2D/3D stencil computation, take a maxwin sized coefficient matrix, and apply stencil computation to a matrix
 * The stencil could be cross-based, which only uses neighbors from one dimension stride (A[i-1][j][k], or square-based
 * which use neighbors from multiple dimension stride (A[i-2][j-1][k]).
 */

#define DEFAULT_DIMSIZE 256

/* use the macro (SQUARE_STENCIL) from compiler to build two versions of the stencil
 * 1. CROSS-based stencil, default, coefficient is an array of 4*maxwin+1, [0] is the center value, and then row and column values
 * 2. SQUARE-based stencil, coefficient is a square matrix with one dimension of (2*maxwin+1)
 */

void print_array(char * title, char * name, REAL * A, long n, long m) {
	printf("%s:\n", title);
	long i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            printf("%s[%d][%d]:%f\n", name, i, j, A[i * m + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void init_array(long N, REAL *A, REAL lower, REAL upper) {
	long i;

	for (i = 0; i < N; i++) {
		A[i] = (REAL)(lower + ((REAL)rand() / (REAL)RAND_MAX) * (upper - lower));
	}
}

REAL check_accdiff(const REAL *output, const REAL *reference, const long dimx, const long dimy, const int maxwin, REAL tolerance){
	int ix, iy;
	REAL acc_diff = 0.0;
	int count = 0;
	for (ix = -maxwin ; ix < dimx + maxwin ; ix++) {
		for (iy = -maxwin ; iy < dimy + maxwin ; iy++) {
			if (ix >= 0 && ix < dimx && iy >= 0 && iy < dimy) {
				// Determine the absolute difference
				REAL difference = fabs(*reference - *output);
				acc_diff += difference;

				REAL error;
				// Determine the relative error
				if (*reference != 0)
					error = difference / *reference;
				else
					error = difference;

				// Check the error is within the tolerance
				//printf("Data at point (%d,%d)\t%f instead of %f\n", ix, iy, *output, *reference);
				if (error > tolerance) {
	//				if (count++<16)	printf("Data error at point (%d,%d)\t%f instead of %f\n", ix, iy, *output, *reference);
				}
			}
			++output;
			++reference;
		}
	}
	return acc_diff;
}

void bm2d_seq(long n, long m, REAL *u, int maxwin, REAL *filter, int num_its);
void bm2d_omp(long n, long m, REAL *u, int maxwin, REAL *coeff, int num_its);

int main(int argc, char * argv[]) {
	long n = DEFAULT_DIMSIZE;
	long m = DEFAULT_DIMSIZE;
	int maxwin = 10;
	int num_its = 5000;

    fprintf(stderr,"Usage: jacobi [<n> <m> <maxwin> <num_its>]\n");
    fprintf(stderr, "\tn - grid dimension in x direction, default: %d\n", n);
    fprintf(stderr, "\tm - grid dimension in y direction, default: n if provided or %d\n", m);
    fprintf(stderr, "\tmaxwin - Max search window size in radius, default: %d\n", maxwin);
    fprintf(stderr, "\tnum_its  - # iterations for iterative solver, default: %d\n", num_its);

    if (argc == 2)      { sscanf(argv[1], "%d", &n); m = n; }
    else if (argc == 3) { sscanf(argv[1], "%d", &n); sscanf(argv[2], "%d", &m); }
    else if (argc == 4) { sscanf(argv[1], "%d", &n); sscanf(argv[2], "%d", &m); sscanf(argv[3], "%d", &maxwin); }
    else if (argc == 5) { sscanf(argv[1], "%d", &n); sscanf(argv[2], "%d", &m); sscanf(argv[3], "%d", &maxwin); sscanf(argv[4], "%d", &num_its); }
    else {
    	/* the rest of arg ignored */
    }

	if (num_its%2==0) num_its++; /* make it odd so uold/u exchange easier */

	long u_dimX = n+maxwin+maxwin;
	long u_dimY = m+maxwin+maxwin;
	long u_volumn = u_dimX*u_dimY;
	int coeff_volumn;
	coeff_volumn = (2*maxwin+1)*(2*maxwin+1); /* this is for square. Even the cross stencil that use only 4*maxwin +1, we will use the same square coeff simply */
//	coeff_volumn = 4*maxwin+1;
    REAL * u = (REAL *)malloc(sizeof(REAL)* u_volumn);
	REAL * u_omp = (REAL *)malloc(sizeof(REAL)* u_volumn);
	REAL * u_omp_mdev = (REAL *)omp_unified_malloc(sizeof(REAL)* u_volumn);
	REAL * u_omp_mdev_iterate = (REAL *)omp_unified_malloc(sizeof(REAL)* u_volumn);
	REAL *coeff = (REAL *) omp_unified_malloc(sizeof(REAL)*coeff_volumn);

	srand(0);
	init_array(u_volumn, u, 0.0, 1.0);
	init_array(coeff_volumn, coeff, 0.0, 1.0);
	memcpy(u_omp, u, sizeof(REAL)*u_volumn);
	memcpy(u_omp_mdev, u, sizeof(REAL)*u_volumn);
	memcpy(u_omp_mdev_iterate, u, sizeof(REAL)*u_volumn);
	//print_array("coeff", "coeff", coeff, 2*maxwin+1, 2*maxwin+1);
	//print_array("original", "u", u, u_dimX, u_dimY);

	printf("serial execution\n");
	REAL base_elapsed = read_timer_ms();
//	bm2d_seq(n, m, u, maxwin, coeff, num_its);
	base_elapsed = read_timer_ms() - base_elapsed;
	//print_array("after stencil", "us", u, u_dimX, u_dimY);

	printf("OMP execution\n");
	REAL omp_elapsed = read_timer_ms();
	int i;
	int num_runs = 1;
//	for (i=0;i<num_runs;i++) bm2d_omp(n, m, u_omp, maxwin, coeff, num_its);
	omp_elapsed = (read_timer_ms() - omp_elapsed)/num_runs;

	omp_init_devices();
//	printf("OMP mdev execution\n");

	printf("OMP mdev iterate execution\n");
	REAL mdev_elapsed = 0.0;

	int num_active_devs = omp_get_num_active_devices();
	int targets[num_active_devs];
	int num_targets = 1;
	double (*bm2d_omp_mdev_function)(int ndevs, int *targets, long n, long m, REAL *u, int maxwin, REAL *coeff,
								 int num_its);
	if (LOOP_DIST_POLICY == OMP_DIST_POLICY_BLOCK || LOOP_DIST_POLICY == OMP_DIST_POLICY_MODEL_1_AUTO || LOOP_DIST_POLICY == OMP_DIST_POLICY_MODEL_2_AUTO) {
		bm2d_omp_mdev_function = bm2d_omp_mdev_iterate;
	} else {
		bm2d_omp_mdev_function = bm2d_omp_mdev;
	}
#if 0
	/* one HOSTCPU */
	num_targets = omp_get_devices(OMP_DEVICE_HOSTCPU, targets, 1);
	mdev_elapsed = bm2d_omp_mdev_function(num_targets, targets, n, m, u_omp_mdev_iterate, maxwin, coeff, num_its);

	/* one NVGPU */
	num_targets = omp_get_devices(OMP_DEVICE_NVGPU, targets, 1);
	mdev_elapsed = bm2d_omp_mdev_function(num_targets, targets, n, m, u_omp_mdev_iterate, maxwin, coeff, num_its);

	/* two NVGPU */
	num_targets = omp_get_devices(OMP_DEVICE_NVGPU, targets, 2);
	mdev_elapsed = bm2d_omp_mdev_function(num_targets, targets, n, m, u_omp_mdev_iterate, maxwin, coeff, num_its);

	/* four NVGPU */
	num_targets = omp_get_devices(OMP_DEVICE_NVGPU, targets, 4);
	mdev_elapsed = bm2d_omp_mdev_function(num_targets, targets, n, m, u_omp_mdev_iterate, maxwin, coeff, num_its);

	/* one ITLMIC */
	num_targets = omp_get_devices(OMP_DEVICE_ITLMIC, targets, 1);
	mdev_elapsed = bm2d_omp_mdev_function(num_targets, targets, n, m, u_omp_mdev_iterate, maxwin, coeff, num_its);

	/* two ITLMIC */
	num_targets = omp_get_devices(OMP_DEVICE_ITLMIC, targets, 2);
	mdev_elapsed = bm2d_omp_mdev_function(num_targets, targets, n, m, u_omp_mdev_iterate, maxwin, coeff, num_its);

	/* one HOSTCPU and one NVGPU */
	num_targets = omp_get_devices(OMP_DEVICE_HOSTCPU, targets, 1);
	num_targets += omp_get_devices(OMP_DEVICE_NVGPU, targets+num_targets, 1);
	mdev_elapsed = bm2d_omp_mdev_function(num_targets, targets, n, m, u_omp_mdev_iterate, maxwin, coeff, num_its);

	/* one HOSTCPU and one ITLMIC */
	num_targets = omp_get_devices(OMP_DEVICE_HOSTCPU, targets, 1);
	num_targets += omp_get_devices(OMP_DEVICE_ITLMIC, targets+num_targets, 1);
	mdev_elapsed = bm2d_omp_mdev_function(num_targets, targets, n, m, u_omp_mdev_iterate, maxwin, coeff, num_its);

	/* one NVGPU and one ITLMIC */
	num_targets = omp_get_devices(OMP_DEVICE_NVGPU, targets, 1);
	num_targets += omp_get_devices(OMP_DEVICE_ITLMIC, targets+num_targets, 1);
	mdev_elapsed = bm2d_omp_mdev_function(num_targets, targets, n, m, u_omp_mdev_iterate, maxwin, coeff, num_its);

	/* one HOSTCPU and two NVGPU */
	num_targets = omp_get_devices(OMP_DEVICE_HOSTCPU, targets, 1);
	num_targets += omp_get_devices(OMP_DEVICE_NVGPU, targets+num_targets, 2);
	mdev_elapsed = bm2d_omp_mdev_function(num_targets, targets, n, m, u_omp_mdev_iterate, maxwin, coeff, num_its);

	/* one HOSTCPU and two ITLMIC */
	num_targets = omp_get_devices(OMP_DEVICE_HOSTCPU, targets, 1);
	num_targets += omp_get_devices(OMP_DEVICE_ITLMIC, targets+num_targets, 2);
	mdev_elapsed = bm2d_omp_mdev_function(num_targets, targets, n, m, u_omp_mdev_iterate, maxwin, coeff, num_its);

	/* two NVGPU and two ITLMIC */
	num_targets = omp_get_devices(OMP_DEVICE_NVGPU, targets, 2);
	num_targets += omp_get_devices(OMP_DEVICE_ITLMIC, targets+num_targets, 2);
	mdev_elapsed = bm2d_omp_mdev_function(num_targets, targets, n, m, u_omp_mdev_iterate, maxwin, coeff, num_its);

	/* four NVGPU and two ITLMIC */
	num_targets = omp_get_devices(OMP_DEVICE_NVGPU, targets, 4);
	num_targets += omp_get_devices(OMP_DEVICE_ITLMIC, targets+num_targets, 2);
	mdev_elapsed = bm2d_omp_mdev_function(num_targets, targets, n, m, u_omp_mdev_iterate, maxwin, coeff, num_its);

	/* one CPU, two NVGPU and two ITLMIC */
	num_targets = omp_get_devices(OMP_DEVICE_HOSTCPU, targets, 1);
	num_targets += omp_get_devices(OMP_DEVICE_NVGPU, targets+num_targets, 2);
	num_targets += omp_get_devices(OMP_DEVICE_ITLMIC, targets+num_targets, 2);
	mdev_elapsed = bm2d_omp_mdev_function(num_targets, targets, n, m, u_omp_mdev_iterate, maxwin, coeff, num_its);
	/* one CPU, four NVGPU and two ITLMIC */
	num_targets = omp_get_devices(OMP_DEVICE_HOSTCPU, targets, 1);
	num_targets += omp_get_devices(OMP_DEVICE_NVGPU, targets+num_targets, 4);
	num_targets += omp_get_devices(OMP_DEVICE_ITLMIC, targets+num_targets, 2);
	mdev_elapsed = bm2d_omp_mdev_function(num_targets, targets, n, m, u_omp_mdev_iterate, maxwin, coeff, num_its);
#endif

    /* run on all devices */
    num_targets = num_active_devs;
    for (i=0;i<num_active_devs;i++) targets[i] = i;
	mdev_elapsed = bm2d_omp_mdev_function(num_targets, targets, n, m, u_omp_mdev_iterate, maxwin, coeff, num_its);

	long flops = n*m*maxwin;
#ifdef SQUARE_SETNCIL
	flops *= 8;
#else
	flops *= 16;
#endif

	printf("======================================================================================================\n");
	printf("\tStencil 2D: %dx%d, stencil maxwin: %d, #iteratins: %d\n", n, m, maxwin, num_its);
	printf("------------------------------------------------------------------------------------------------------\n");
	printf("Performance:\t\tRuntime (ms)\t MFLOPS \t\tError (compared to base)\n");
	printf("------------------------------------------------------------------------------------------------------\n");
	printf("base:\t\t%4f\t%4f \t\t%g\n", base_elapsed, flops / (1.0e-3 * base_elapsed), 0.0); //check_accdiff(u, u, u_dimX, u_dimY, maxwin, 1.0e-7));
	printf("omp: \t\t%4f\t%4f \t\t%g\n", omp_elapsed, flops / (1.0e-3 * omp_elapsed), check_accdiff(u, u_omp, n, m, maxwin, 0.00001f));
//	printf("omp_mdev: \t%4f\t%4f \t\t%g\n", mdev_elapsed, flops / (1.0e-3 * mdev_elapsed), check_accdiff(u, u_omp_mdev, n, m, maxwin, 0.00001f));
	printf("omp_mdev_it: \t%4f\t%4f \t\t%g\n", mdev_elapsed, flops / (1.0e-3 * mdev_elapsed), check_accdiff(u, u_omp_mdev_iterate, n, m, maxwin, 0.00001f));

	free(u);
	free(u_omp);
	omp_unified_free(u_omp_mdev);
	omp_unified_free(u_omp_mdev_iterate);
	omp_unified_free(coeff);
	omp_fini_devices();

	return 0;
}

void bm2d_seq_normal(long n, long m, REAL *u, int maxwin, REAL *coeff, int num_its) {
	long it; /* iteration */
	long u_dimX = n + 2 * maxwin;
	long u_dimY = m + 2 * maxwin;
	int coeff_dimX = 2*maxwin+1;
	REAL *uold = (REAL*)malloc(sizeof(REAL)*u_dimX * u_dimY);
	memcpy(uold, u, sizeof(REAL)*u_dimX*u_dimY);
	coeff = coeff + coeff_dimX * maxwin + maxwin; /* let coeff point to the center element */
	REAL * uold_save = uold;
	REAL * u_save = u;

	for (it = 0; it < num_its; it++) {
		int ix, iy, ir;

		for (ix = 0; ix < n; ix++) {
			for (iy = 0; iy < m; iy++) {
				int radius = drand48() * maxwin;
				if (radius < 0) continue;
				int count = 4*radius+1;
#ifdef SQUARE_SETNCIL
				count = coeff_dimX * coeff_dimX;
#endif
				int offset = (ix+radius)*u_dimY+radius+iy;
				REAL * temp_u = &u[offset];
				REAL * temp_uold = &uold[offset];
				REAL result = temp_uold[0] * coeff[0];
				/* 2/4 way loop unrolling */
				for (ir = 1; ir <= radius; ir++) {
					result += coeff[ir] * temp_uold[ir];           		//horizontal right
					result += coeff[-ir]* temp_uold[-ir];                  // horizontal left
					result += coeff[-ir*coeff_dimX] * temp_uold[-ir * u_dimY]; //vertical up
					result += coeff[ir*coeff_dimX] * temp_uold[ir * u_dimY]; // vertical bottom
#ifdef SQUARE_SETNCIL
					result += coeff[-ir*coeff_dimX-ir] * temp_uold[-ir * u_dimY-ir] // left upper corner
					result += coeff[-ir*coeff_dimX+ir] * temp_uold[-ir * u_dimY+ir] // right upper corner
					result += coeff[ir*coeff_dimX-ir] * temp_uold[ir * u_dimY]-ir] // left bottom corner
					result += coeff[ir*coeff_dimX+ir] * temp_uold[ir * u_dimY]+ir] // right bottom corner
#endif
				}
				*temp_u = result/count;
			}
		}
		REAL * tmp = uold;
		uold = u;
		u = tmp;
//		if (it % 500 == 0)
//			printf("Finished %d iteration\n", it);
	} /*  End iteration loop */
	free(uold_save);
}

#if 0
void bm2d_omp_mdev(long n, long m, REAL *u, int maxwin, REAL *coeff, int num_its) {
	long it; /* iteration */
	long u_dimX = n + 2 * maxwin;
	long u_dimY = m + 2 * maxwin;
	int coeff_dimX = 2 * maxwin + 1;
	coeff = coeff + (2 * maxwin + 1) * maxwin + maxwin; /* let coeff point to the center element */
	int count = 4*maxwin+1;
#ifdef SQUARE_SETNCIL
	count = coeff_dimX * coeff_dimX;
#endif

	/* uold should be simpliy allocated on the dev and then copy data from u, here we simplified the initialization */
	REAL *uold = (REAL *) malloc(sizeof(REAL) * u_dimX * u_dimY);
	memcpy(uold, u, sizeof(REAL)*u_dimX * u_dimY);
#pragma omp target data device(*) map(to:n, m, u_dimX, u_dimY, maxwin, coeff_center, coeff[coeff_dimX][coeff_dimX]) \
  map(tofrom:u[u_dimX][u_dimY] dist_data(BLOCK,DUPLICATE) halo(maxwin,)) map(to:uold[u_dimX][u_dimY] dist_data(BLOCK,DUPLICATE) halo(maxwin,))
#pragma omp parallel shared(n, m, maxwin, coeff, num_its, u_dimX, u_dimY, coeff_dimX) private(it) firstprivate(u, uold) //num_threads(/* num of devices + number of cores */)
	{
		int ix, iy, ir;

/*
#pragma omp target device(*) dist_iteration(BLOCK)
#pragma omp for
		for (ix = 0; ix < u_dimX; ix++) {
			for (iy = 0; iy < u_dimY; iy++) {
				uold[ix * u_dimY + iy] = u[ix * u_dimY + iy];
			}
		}
*/
		for (it = 0; it < num_its; it++) {
#pragma omp target device(*) dist_iteration(BLOCK)
#pragma omp for
			for (ix = 0; ix < n; ix++) {
				REAL *temp_u = &u[(ix + maxwin) * u_dimY+maxwin];
				REAL *temp_uold = &uold[(ix + maxwin) * u_dimY+maxwin];
				for (iy = 0; iy < m; iy++) {
					REAL result = temp_uold[0] * coeff[0];
					/* 2/4 way loop unrolling */
					for (ir = 1; ir <= maxwin; ir++) {
						result += coeff[ir] * temp_uold[ir];                //horizontal right
						result += coeff[-ir] * temp_uold[-ir];                  // horizontal left
						result += coeff[-ir * coeff_dimX] * temp_uold[-ir * u_dimY]; //vertical up
						result += coeff[ir * coeff_dimX] * temp_uold[ir * u_dimY]; // vertical bottom
#ifdef SQUARE_SETNCIL
						result += coeff[-ir*coeff_dimX-ir] * temp_uold[-ir * u_dimY-ir] // left upper corner
						result += coeff[-ir*coeff_dimX+ir] * temp_uold[-ir * u_dimY+ir] // right upper corner
						result += coeff[ir*coeff_dimX-ir] * temp_uold[ir * u_dimY]-ir] // left bottom corner
						result += coeff[ir*coeff_dimX+ir] * temp_uold[ir * u_dimY]+ir] // right bottom corner
#endif
					}
					*temp_u = result/count;
					temp_u++;
					temp_uold++;
				}
			}
#pragma omp halo_exchange(u);
			REAL *tmp = uold;
			uold = u;
			u = tmp;
//		if (it % 500 == 0)
//			printf("Finished %d iteration by thread %d of %d\n", it, omp_get_thread_num(), omp_get_num_threads());
		} /*  End iteration loop */
	}
	free(uold);
}
#endif
