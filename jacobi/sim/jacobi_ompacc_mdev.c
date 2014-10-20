#include <stdio.h>
#include <math.h>
#include <omp.h>
// Add timing support
#include <sys/time.h>
#include "homp.h"

/************************************************************
 * program to solve a finite difference
 * discretization of Helmholtz equation :
 * (d2/dx2)u + (d2/dy2)u - alpha u = f
 * using Jacobi iterative method.
 *
 * Modified: Sanjiv Shah,       Kuck and Associates, Inc. (KAI), 1998
 * Author:   Joseph Robicheaux, Kuck and Associates, Inc. (KAI), 1998
 *
 * This c version program is translated by
 * Chunhua Liao, University of Houston, Jan, 2005
 *
 * Directives are used in this code to achieve parallelism.
 * All do loops are parallelized with default 'static' scheduling.
 *
 * Input :  n - grid dimension in x direction
 *          m - grid dimension in y direction
 *          alpha - Helmholtz constant (always greater than 0.0)
 *          tol   - error tolerance for iterative solver
 *          relax - Successice over relaxation parameter
 *          mits  - Maximum iterations for iterative solver
 *
 * On output
 *       : u(n,m) - Dependent variable (solutions)
 *       : f(n,m) - Right hand side function
 *************************************************************/

#define REAL double
// flexible between REAL and double
int dist = 1; /* 1, 2, or 3; 1: row major; 2: column major; 3: row-column */
#define DEFAULT_MSIZE 256

void print_array(char * title, char * name, REAL * A, long n, long m) {
	printf("%s:\n", title);
	long i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            printf("%s[%d][%d]:%f  ", name, i, j, A[i * m + j]);
        }
        printf("\n");
    }
    printf("\n");
}

/*      subroutine initialize (n,m,alpha,dx,dy,u,f)
 ******************************************************
 * Initializes data
 * Assumes exact solution is u(x,y) = (1-x^2)*(1-y^2)
 *
 ******************************************************/
void initialize(long n, long m, REAL alpha, REAL *dx, REAL * dy, REAL u[n][m], REAL f[n][m]) {
	int i;
	int j;
	int xx;
	int yy;
//double PI=3.1415926;
	*dx = (2.0 / (n - 1));
	*dy = (2.0 / (m - 1));
	/* Initialize initial condition and RHS */
//#pragma omp parallel for private(xx,yy,j,i)
	for (i = 0; i < n; i++)
		for (j = 0; j < m; j++) {
			xx = ((int) (-1.0 + (*dx * (i - 1))));
			yy = ((int) (-1.0 + (*dy * (j - 1))));
			u[i][j] = 0.0;
			f[i][j] = (((((-1.0 * alpha) * (1.0 - (xx * xx)))
					* (1.0 - (yy * yy))) - (2.0 * (1.0 - (xx * xx))))
					- (2.0 * (1.0 - (yy * yy))));
		}
}

/*  subroutine error_check (n,m,alpha,dx,dy,u,f)
 implicit none
 ************************************************************
 * Checks error between numerical and exact solution
 *
 ************************************************************/
void error_check(long n, long m, REAL alpha, REAL dx, REAL dy, REAL u[n][m], REAL f[n][m]) {
	int i;
	int j;
	REAL xx;
	REAL yy;
	REAL temp;
	REAL error;
	error = 0.0;
//#pragma omp parallel for private(xx,yy,temp,j,i) reduction(+:error)
	for (i = 0; i < n; i++)
		for (j = 0; j < m; j++) {
			xx = (-1.0 + (dx * (i - 1)));
			yy = (-1.0 + (dy * (j - 1)));
			temp = (u[i][j] - ((1.0 - (xx * xx)) * (1.0 - (yy * yy))));
			error = (error + (temp * temp));
		}
	error = (sqrt(error) / (n * m));
	printf("Solution Error: %2.6g\n", error);
}
void jacobi_seq(long n, long m, REAL dx, REAL dy, REAL alpha, REAL relax, REAL u[n][m], REAL f[n][m], REAL tol, int mits);
void jacobi_omp_mdev(long n, long m, REAL dx, REAL dy, REAL alpha, REAL relax, REAL u[n][m], REAL f[n][m], REAL tol, int mits);

int main(int argc, char * argv[]) {
	long n = DEFAULT_MSIZE;
	long m = DEFAULT_MSIZE;
	REAL alpha = 0.0543;
	REAL tol = 0.0000000001;
	REAL relax = 1.0;
	int mits = 2;

    fprintf(stderr,"Usage: jacobi [<n> <m> <alpha> <tol> <relax> <mits>]\n");
    fprintf(stderr, "\tn - grid dimension in x direction, default: %d\n", n);
    fprintf(stderr, "\tm - grid dimension in y direction, default: n if provided or %d\n", m);
    fprintf(stderr, "\talpha - Helmholtz constant (always greater than 0.0), default: %g\n", alpha);
    fprintf(stderr, "\ttol   - error tolerance for iterative solver, default: %g\n", tol);
    fprintf(stderr, "\trelax - Successice over relaxation parameter, default: %g\n", relax);
    fprintf(stderr, "\tmits  - Maximum iterations for iterative solver, default: %d\n", mits);

    if (argc == 2)      { sscanf(argv[1], "%d", &n); m = n; }
    else if (argc == 3) { sscanf(argv[1], "%d", &n); sscanf(argv[2], "%d", &m); }
    else if (argc == 4) { sscanf(argv[1], "%d", &n); sscanf(argv[2], "%d", &m); sscanf(argv[3], "%g", &alpha); }
    else if (argc == 5) { sscanf(argv[1], "%d", &n); sscanf(argv[2], "%d", &m); sscanf(argv[3], "%g", &alpha); sscanf(argv[4], "%g", &tol); }
    else if (argc == 6) { sscanf(argv[1], "%d", &n); sscanf(argv[2], "%d", &m); sscanf(argv[3], "%g", &alpha); sscanf(argv[4], "%g", &tol); sscanf(argv[5], "%g", &relax); }
    else if (argc == 7) { sscanf(argv[1], "%d", &n); sscanf(argv[2], "%d", &m); sscanf(argv[3], "%g", &alpha); sscanf(argv[4], "%g", &tol); sscanf(argv[5], "%g", &relax); sscanf(argv[6], "%d", &mits); }
    else {
    	/* the rest of arg ignored */
    }
    printf("jacobi %d %d %g %g %g %d\n", n, m, alpha, tol, relax, mits);
    printf("------------------------------------------------------------------------------------------------------\n");

    /** init the array */
    //REAL u[n][m];
    //REAL f[n][m];
	omp_init_devices();

    REAL * u = (REAL *)malloc(sizeof(REAL)*n*m);
    REAL * f = (REAL *)malloc(sizeof(REAL)*n*m);

    REAL *udev = (REAL *)malloc(sizeof(REAL)*n*m);
    REAL *fdev = (REAL *)malloc(sizeof(REAL)*n*m);

    REAL dx; /* grid spacing in x direction */
    REAL dy; /* grid spacing in y direction */

    initialize(n, m, alpha, &dx, &dy, u, f);

    memcpy(udev, u, n*m*sizeof(REAL));
    memcpy(fdev, f, n*m*sizeof(REAL));
	print_array("Before Run", "u",(REAL*)u, n, m);
	print_array("Before Run", "u",(REAL*)udev, n, m);
    int i, j;
    for (i=0; i<n*m; i++) {
    	udev[i] = u[i];
    	fdev[i] = f[i];
    }
	REAL elapsed = read_timer_ms();
	//jacobi_seq(n, m, dx, dy, alpha, relax, u, f, tol, mits);
	elapsed = read_timer_ms() - elapsed;
	printf("seq elasped time(ms): %12.6g\n", elapsed);
	double mflops = (0.001*mits*(n-2)*(m-2)*13) / elapsed;
	printf("MFLOPS: %12.6g\n", mflops);

	elapsed = read_timer_ms();


	jacobi_omp_mdev(n, m, dx, dy, alpha, relax, udev, fdev, tol, mits);
	elapsed = read_timer_ms() - elapsed;
	printf("mdev elasped time(ms): %12.6g\n", elapsed);
	mflops = (0.001*mits*(n-2)*(m-2)*13) / elapsed;
	printf("MFLOPS: %12.6g\n", mflops);

	//print_array("Sequential Run", "u",(REAL*)u, n, m);
	//print_array("Mdev Run", "udev", (REAL*)udev, n, m);

	error_check(n, m, alpha, dx, dy, u, f);
	free(u); free(f);
	free(udev);free(fdev);

	return 0;
}

/*      subroutine jacobi (n,m,dx,dy,alpha,omega,u,f,tol,mits)
 ******************************************************************
 * Subroutine HelmholtzJ
 * Solves poisson equation on rectangular grid assuming :
 * (1) Uniform discretization in each direction, and
 * (2) Dirichlect boundary conditions
 *
 * Jacobi method is used in this routine
 *
 * Input : n,m   Number of grid points in the X/Y directions
 *         dx,dy Grid spacing in the X/Y directions
 *         alpha Helmholtz eqn. coefficient
 *         omega Relaxation factor
 *         f(n,m) Right hand side function
 *         u(n,m) Dependent variable/Solution
 *         tol    Tolerance for iterative solver
 *         mits  Maximum number of iterations
 *
 * Output : u(n,m) - Solution
 *****************************************************************/
void jacobi_seq(long n, long m, REAL dx, REAL dy, REAL alpha, REAL omega, REAL u[n][m], REAL f[n][m], REAL tol, int mits) {
	int i, j, k;
	REAL error;
	REAL ax;
	REAL ay;
	REAL b;
	REAL resid;
	REAL uold[n][m];
	/*
	 * Initialize coefficients */
	/* X-direction coef */
	ax = (1.0 / (dx * dx));
	/* Y-direction coef */
	ay = (1.0 / (dy * dy));
	/* Central coeff */
	b = (((-2.0 / (dx * dx)) - (2.0 / (dy * dy))) - alpha);
	error = (10.0 * tol);
	k = 1;
	while ((k <= mits) && (error > tol)) {
		error = 0.0;

		/* Copy new solution into old */
		for (i = 0; i < n; i++)
			for (j = 0; j < m; j++)
				uold[i][j] = u[i][j];

		for (i = 1; i < (n - 1); i++)
			for (j = 1; j < (m - 1); j++) {
				resid = (ax * (uold[i - 1][j] + uold[i + 1][j]) + ay * (uold[i][j - 1] + uold[i][j + 1]) + b * uold[i][j] - f[i][j]) / b;

				u[i][j] = uold[i][j] - omega * resid;
				error = error + resid * resid;
			}
		/* Error check */
		if (k % 500 == 0)
		printf("Finished %d iteration with error: %g\n", k, error);
		error = sqrt(error) / (n * m);

		k = k + 1;
	} /*  End iteration loop */
	printf("Total Number of Iterations: %d\n", k);
	printf("Residual: %.15g\n", error);
}

#if defined (DEVICE_NVGPU_SUPPORT)
#define LOOP_COLLAPSE 1
#if !LOOP_COLLAPSE
__global__ void OUT__2__10550__(int n,int m,REAL *_dev_u,REAL *_dev_uold)
{
  int _p_j;
  int _dev_i ;
  long _dev_lower, _dev_upper;
  XOMP_accelerator_loop_default (0, n-1, 1, &_dev_lower, &_dev_upper);
  for (_dev_i = _dev_lower ; _dev_i <= _dev_upper; _dev_i++) {
    for (_p_j = 0; _p_j < m; _p_j++)
      _dev_uold[_dev_i * MSIZE + _p_j] = _dev_u[_dev_i * MSIZE + _p_j];
  }
}

__global__ void OUT__1__10550__(int start_n, int n,int m,REAL omega,REAL ax,REAL ay,REAL b,REAL *_dev_per_block_error,REAL *_dev_u,REAL *_dev_f,REAL *_dev_uold)
{
  int _p_j;
  REAL _p_error;
  _p_error = 0;
  REAL _p_resid;
  long _dev_lower, _dev_upper;
  XOMP_accelerator_loop_default (start_n, n-1, 1, &_dev_lower, &_dev_upper);
  int _dev_i;
  for (_dev_i = _dev_lower; _dev_i<= _dev_upper; _dev_i ++) {
    for (_p_j = 1; _p_j < (m - 1); _p_j++) {
      _p_resid = (((((ax * (_dev_uold[(_dev_i - 1) * MSIZE + _p_j] + _dev_uold[(_dev_i + 1) * MSIZE + _p_j])) + (ay * (_dev_uold[_dev_i * MSIZE + (_p_j - 1)] + _dev_uold[_dev_i * MSIZE + (_p_j + 1)]))) + (b * _dev_uold[_dev_i * MSIZE + _p_j])) - _dev_f[_dev_i * MSIZE + _p_j]) / b);
      _dev_u[_dev_i * MSIZE + _p_j] = (_dev_uold[_dev_i * MSIZE + _p_j] - (omega * _p_resid));
      _p_error = (_p_error + (_p_resid * _p_resid));
    }
  }
  xomp_inner_block_reduction_REAL(_p_error,_dev_per_block_error,6);
}
#else
__global__ void OUT__2__10550__(int n,int m,REAL *_dev_u,REAL *_dev_uold)
{
  int _p_j;
  int ij;
  int _dev_lower, _dev_upper;

  int _dev_i ;

 // variables for adjusted loop info considering both original chunk size and step(strip)
 int _dev_loop_chunk_size;
 int _dev_loop_sched_index;
 int _dev_loop_stride;

// 1-D thread block:
int _dev_thread_num = gridDim.x * blockDim.x;
int _dev_thread_id = blockDim.x * blockIdx.x + threadIdx.x;

int orig_start =0;
int orig_end = n*m-1; // inclusive upper bound
int orig_step = 1;
int orig_chunk_size = 1;

 XOMP_static_sched_init (orig_start, orig_end, orig_step, orig_chunk_size, _dev_thread_num, _dev_thread_id, \
                         & _dev_loop_chunk_size , & _dev_loop_sched_index, & _dev_loop_stride);

 //XOMP_accelerator_loop_default (1, (n-1)*(m-1)-1, 1, &_dev_lower, &_dev_upper);
 while (XOMP_static_sched_next (&_dev_loop_sched_index, orig_end, orig_step,_dev_loop_stride, _dev_loop_chunk_size, _dev_thread_num, _dev_thread_id, & _dev_lower
, & _dev_upper))
 {
   for (ij = _dev_lower ; ij <= _dev_upper; ij ++) {
     //  for (_dev_i = _dev_lower ; _dev_i <= _dev_upper; _dev_i++) {
     //    for (_p_j = 0; _p_j < m; _p_j++)
     _dev_i = ij/m;
     _p_j = ij%m;
     _dev_uold[_dev_i * MSIZE + _p_j] = _dev_u[_dev_i * MSIZE + _p_j];
   }
  }
 }

__global__ void OUT__1__10550__(int start_n, int n,int m,REAL omega,REAL ax,REAL ay,REAL b,REAL *_dev_per_block_error,REAL *_dev_u,REAL *_dev_f,REAL *_dev_uold)
{
  int _dev_i;
  int ij;
  int _p_j;
  int _dev_lower, _dev_upper;

  REAL _p_error;
  _p_error = 0;
  REAL _p_resid;

  // variables for adjusted loop info considering both original chunk size and step(strip)
  int _dev_loop_chunk_size;
  int _dev_loop_sched_index;
  int _dev_loop_stride;

  // 1-D thread block:
  int _dev_thread_num = gridDim.x * blockDim.x;
  int _dev_thread_id = blockDim.x * blockIdx.x + threadIdx.x;

  //TODO: adjust bound to be inclusive later
  int orig_start =start_n;
  int orig_end = (n)*m-1;
  int orig_step = 1;
  int orig_chunk_size = 1;

  XOMP_static_sched_init (orig_start, orig_end, orig_step, orig_chunk_size, _dev_thread_num, _dev_thread_id, \
      & _dev_loop_chunk_size , & _dev_loop_sched_index, & _dev_loop_stride);

  //XOMP_accelerator_loop_default (1, (n-1)*(m-1)-1, 1, &_dev_lower, &_dev_upper);
  while (XOMP_static_sched_next (&_dev_loop_sched_index, orig_end,orig_step, _dev_loop_stride, _dev_loop_chunk_size, _dev_thread_num, _dev_thread_id, & _dev_lower, & _dev_upper))
  {
    for (ij = _dev_lower; ij <= _dev_upper; ij ++) {
      _dev_i = ij/(m-1);
      _p_j = ij%(m-1);

      if (_dev_i>=start_n && _dev_i< (n) && _p_j>=1 && _p_j< (m-1)) // must preserve the original boudary conditions here!!
      {
	_p_resid = (((((ax * (_dev_uold[(_dev_i - 1) * MSIZE + _p_j] + _dev_uold[(_dev_i + 1) * MSIZE + _p_j])) + (ay * (_dev_uold[_dev_i * MSIZE + (_p_j - 1)] + _dev_uold[_dev_i * MSIZE + (_p_j + 1)]))) + (b * _dev_uold[_dev_i * MSIZE + _p_j])) - _dev_f[_dev_i * MSIZE + _p_j]) / b);
	_dev_u[_dev_i * MSIZE + _p_j] = (_dev_uold[_dev_i * MSIZE + _p_j] - (omega * _p_resid));
	_p_error = (_p_error + (_p_resid * _p_resid));
      }
    }
  }

  xomp_inner_block_reduction_REAL(_p_error,_dev_per_block_error,6);
}
#endif /* LOOP_CLAPSE */
#endif /* NVGPU support */

struct OUT__2__10550__args {
	long n;
	long m;
	REAL * u;
	REAL * uold;
};

void OUT__2__10550__launcher(omp_offloading_t * off, void *args) {
    struct OUT__2__10550__args * iargs = (struct OUT__2__10550__args*) args;
    long n = iargs->n;
    long m = iargs->m;

    omp_offloading_info_t * off_info = off->off_info;
//    printf("IN LAUNCHER: dev: %d, off: %X, off_info: %X\n", off->devseqid, off, off_info);

    omp_data_map_t * map_u = omp_map_get_map(off, iargs->u, -1); /* 1 means the map u */
    omp_data_map_t * map_uold = omp_map_get_map(off, iargs->uold, -1); /* 2 means the map uold */

    REAL * u_p = (REAL *)map_u->map_dev_ptr;
    REAL (*u)[m] = (REAL(*)[m])u_p;

    /* we need to adjust index offset for those who has halo region because of we use attached halo region memory management */
    REAL * uold_p = (REAL *)map_uold->map_dev_ptr;
    int uold_0_offset;
    int uold_1_offset;

    if (omp_data_map_get_halo_left_devseqid(map_uold, 0) >= 0) {
    	uold_0_offset = map_uold->info->halo_info[0].left;
    } else uold_0_offset = 0;
    if (omp_data_map_get_halo_left_devseqid(map_uold, 1) >= 0) {
        uold_1_offset = map_uold->info->halo_info[1].left;
    } else uold_1_offset = 0;

    int uold_0_length = map_uold->map_dim[0];
    int uold_1_length = map_uold->map_dim[1];

    REAL (*uold)[uold_1_length] = (REAL(*)[uold_1_length])uold_p; /** cast a pointer to a 2-D array */

    printf("kernel launcher: u: %X, uold: %X\n", u, uold);
#if CORRECTNESS_CHECK
    print_array("u in device: ", "udev", u, n, m);
    print_array("uold in device: ", "uolddev", uold, n, m);
#endif

	long start;
	if (dist == 1) {
		omp_loop_map_range(map_u, 0, -1, -1, &start, &n);
	} else if (dist == 2) {
		omp_loop_map_range(map_u, 1, -1, -1, &start, &m);
	} else /* vx == 3) */ {
		omp_loop_map_range(map_u, 0, -1, -1, &start, &n);
		omp_loop_map_range(map_u, 1, -1, -1, &start, &m);
	}
//	printf("dist: %d, dev: %d, n: %d, m: %d\n", dist, off->devseqid, n,m);

	omp_device_type_t devtype = off_info->targets[off->devseqid]->type;
#if defined (DEVICE_NVGPU_SUPPORT)
	if (devtype == OMP_DEVICE_NVGPU) {
		int _threads_per_block_ = xomp_get_maxThreadsPerBlock();
		int _num_blocks_ = xomp_get_max1DBlock(length_n*m);
		OUT__2__10550__<<<_num_blocks_, _threads_per_block_, 0,off->stream.systream.cudaStream>>>(n, m,u,uold);
	} else
#endif
	if (devtype == OMP_DEVICE_THSIM) {
		int i, j;
		for (i = 0; i < n; i++)
			for (j = 0; j < m; j++) {
				/* since uold has halo region, here we need to adjust index to reflect the new offset */
				uold[i+uold_0_offset][j+uold_1_offset] = u[i][j];
			}
	} else {
		fprintf(stderr, "device type is not supported for this call\n");
	}

	printf("udev: %d\n", off->devseqid);
	print_array("udev", "u",(REAL*)u, n, m);

	//printf("OUT__2__10550__launcher done\n");
}

struct OUT__1__10550__args {
	long n;
	long m;
	REAL ax;
	REAL ay;
	REAL b;
	REAL omega;
	REAL resid;
	REAL *error;

	REAL * u;
	REAL * uold;
	REAL * f;
};

void OUT__1__10550__launcher(omp_offloading_t * off, void *args) {
    struct OUT__1__10550__args * iargs = (struct OUT__1__10550__args*) args;
    long n = iargs->n;
    long m = iargs->m;
    REAL ax = iargs->ax;
    REAL ay = iargs->ay;
    REAL b = iargs->b;
    REAL omega = iargs->omega;
	REAL error = iargs->error[off->devseqid];
	REAL resid = iargs->resid;

    omp_offloading_info_t * off_info = off->off_info;
//    printf("off: %X, off_info: %X, devseqid: %d\n", off, off_info, off->devseqid);
    omp_data_map_t * map_f = omp_map_get_map(off, iargs->f, -1); /* 0 is for the map f, here we use -1 so it will search the offloading stack */
    omp_data_map_t * map_u = omp_map_get_map(off, iargs->u, -1); /* 1 is for the map u */
    omp_data_map_t * map_uold = omp_map_get_map(off, iargs->uold, -1); /* 2 is for the map uld */

    REAL * f_p = (REAL *)map_f->map_dev_ptr;
    REAL * u_p = (REAL *)map_u->map_dev_ptr;
    REAL (*f)[m] = (REAL(*)[m])f_p; /* cast pointer to array */
    REAL (*u)[m] = (REAL(*)[m])u_p;

    /* we need to adjust index offset for those who has halo region because of we use attached halo region memory management */
    REAL * uold_p = (REAL *)map_uold->map_dev_ptr;
    int uold_0_offset;
    int uold_1_offset;

    if (omp_data_map_get_halo_left_devseqid(map_uold, 0) >= 0) {
    	uold_0_offset = map_uold->info->halo_info[0].left;
    } else uold_0_offset = 0;
    if (omp_data_map_get_halo_left_devseqid(map_uold, 1) >= 0) {
        uold_1_offset = map_uold->info->halo_info[1].left;
    } else uold_1_offset = 0;

    int uold_0_length = map_uold->map_dim[0];
    int uold_1_length = map_uold->map_dim[1];

    REAL (*uold)[uold_1_length] = (REAL(*)[uold_1_length])uold_p; /** cast a pointer to a 2-D array */

#if CORRECTNESS_CHECK
    printf("kernel launcher: u: %X, uold: %X\n", u, uold);
    print_array("u in device: ", "udev", u, n, m);
    print_array("uold in device: ", "uolddev", uold, n, m);
#endif

	long start;
	if (dist == 1) {
		omp_loop_map_range(map_u, 0, -1, -1, &start, &n);
	} else if (dist == 2) {
		omp_loop_map_range(map_u, 1, -1, -1, &start, &m);
	} else /* vx == 3) */ {
		omp_loop_map_range(map_u, 0, -1, -1, &start, &n);
		omp_loop_map_range(map_u, 1, -1, -1, &start, &m);
	}
//	printf("dist: %d, dev: %d, n: %d, m: %d\n", dist, off->devseqid, n,m);

	omp_device_type_t devtype = off_info->targets[off->devseqid]->type;
#if defined (DEVICE_NVGPU_SUPPORT)
	if (devtype == OMP_DEVICE_NVGPU) {
	/* for reduction operation */
	long start_n, length_n;
	omp_loop_map_range(__dev_map_u__, 0, -1, -1, &start_n, &length_n);
	int _threads_per_block_ = xomp_get_maxThreadsPerBlock();
	int _num_blocks_ = xomp_get_max1DBlock(length_n*m);
	cudaMalloc(&_dev_per_block_error[__i__], _num_blocks_ * sizeof(REAL));
	_host_per_block_error[__i__] = (REAL*)(malloc(_num_blocks_*sizeof(REAL)));
	reduction_callback_args[__i__] = (omp_reduction_REAL_t*)malloc(sizeof(omp_reduction_REAL_t));

	int _threads_per_block_ = xomp_get_maxThreadsPerBlock();
	int _num_blocks_ = xomp_get_max1DBlock(length_n*m);

	omp_reduction_REAL_t * args = reduction_callback_args[__i__];
	args->input = _host_per_block_error[__i__];
	args->num = _num_blocks_;
	args->opers = 6;
	//printf("%d device: original offset: %d, mapped_offset: %d, length: %d\n", __i__, offset_n, start_n, length_n);

	/* Launch CUDA kernel ... */
	/** since here we do the same mapping, so will reuse the _threads_per_block and _num_blocks */
	OUT__1__10550__<<<_num_blocks_, _threads_per_block_,(_threads_per_block_ * sizeof(REAL)),
			__dev_stream__[__i__].systream.cudaStream>>>(start_n, length_n, m,
			omega, ax, ay, b, _dev_per_block_error[__i__],
			(REAL*)__dev_map_u__->map_dev_ptr, (REAL*)__dev_map_f__->map_dev_ptr,(REAL*)__dev_map_uold__->map_dev_ptr);
	/* copy back the results of reduction in blocks */
	cudaMemcpyAsync(_host_per_block_error[__i__], _dev_per_block_error[__i__], sizeof(REAL)*_num_blocks_, cudaMemcpyDeviceToHost, __dev_stream__[__i__].systream.cudaStream);
	cudaStreamAddCallback(__dev_stream__[__i__].systream.cudaStream, xomp_beyond_block_reduction_REAL_stream_callback, args, 0);
	/* xomp_beyond_block_reduction_REAL(_dev_per_block_error, _num_blocks_, 6); */
	//xomp_freeDevice(_dev_per_block_error);

	} else
#endif
	if (devtype == OMP_DEVICE_THSIM) {
		int i, j;
		REAL resid;
		REAL error;
	    if (omp_data_map_get_halo_left_devseqid(map_uold, 0) >= 0) {
	    	i = 0;
	    } else i = 1;

	    if (omp_data_map_get_halo_right_devseqid(map_uold, 0) >= 0) {
	    } else n = n - 1;

	    if (omp_data_map_get_halo_left_devseqid(map_uold, 1) >= 0) {
	    	j = 0;
	    } else j = 1;
	    if (omp_data_map_get_halo_right_devseqid(map_uold, 1) >= 0) {
	    } else m = m - 1;

		for (; i <n; i++)
			for (; j <m; j++) {
				resid = (ax * (uold[i - 1 + uold_0_offset][j + uold_1_offset] + uold[i + 1 + uold_0_offset][j+uold_1_offset]) + ay * (uold[i+uold_0_offset][j - 1+uold_1_offset] + uold[i+uold_0_offset][j + 1+uold_1_offset]) + b * uold[i+uold_0_offset][j+uold_1_offset] - f[i][j]) / b;

				u[i][j] = uold[i+uold_0_offset][j+uold_1_offset] - omega * resid;
				error = error + resid * resid;
			}
		iargs->error[off->devseqid] = error;
	} else {
		fprintf(stderr, "device type is not supported for this call\n");
	}
}

void jacobi_omp_mdev(long n, long m, REAL dx, REAL dy, REAL alpha, REAL omega, REAL u[n][m], REAL f[n][m], REAL tol, int mits) {
	int i, j, k;
	REAL error;
	REAL ax;
	REAL ay;
	REAL b;
	REAL uold[n][m];
	/*
	 * Initialize coefficients */
	/* X-direction coef */
	ax = (1.0 / (dx * dx));
	/* Y-direction coef */
	ay = (1.0 / (dy * dy));
	/* Central coeff */
	b = (((-2.0 / (dx * dx)) - (2.0 / (dy * dy))) - alpha);
	error = (10.0 * tol);
	k = 1;
#if 0
#pragma omp target data device(*) map(to:n, m, omega, ax, ay, b, f{0}[0:m]>>{:}) map(tofrom:u{0:n}[0:m]>>{:}) map(alloc:uold{0:n|1}[0:m]>>{:})
  while ((k<=mits)&&(error>tol))
  {
    error = 0.0;

    /* Copy new solution into old */
//#pragma omp parallel
//    {
#pragma omp target device(*)//map(in:n, m, u[0:n][0:m]) map(out:uold[0:n][0:m])
#pragma omp parallel for private(j,i) dist_iteration match_range u[]{}
/* #pragma omp parallel for private(j,i) dist_iteration >>(*) */
      for(i=0;i<n;i++)
        for(j=0;j<m;j++)
          uold[i][j] = u[i][j];
#pragma omp halo exchange uold[:]{:}

#pragma omp target device(*)//map(in:n, m, omega, ax, ay, b, f[0:n][0:m], uold[0:n][0:m]) map(out:u[0:n][0:m])
#pragma omp parallel for private(resid,j,i) reduction(+:error) dist_iteration match_range u[]{}// nowait
      for (i=1;i<(n-1);i++)
        for (j=1;j<(m-1);j++)
        {
          resid = (ax*(uold[i-1][j] + uold[i+1][j])\
              + ay*(uold[i][j-1] + uold[i][j+1])+ b * uold[i][j] - f[i][j])/b;

          u[i][j] = uold[i][j] - omega * resid;
          error = error + resid*resid ;
        }

//    }
    /*  omp end parallel */

    /* Error check */

    if (k%500==0)
      printf("Finished %d iteration with error =%g\n",k, error);
    error = sqrt(error)/(n*m);

    k = k + 1;
  }          /*  End iteration loop */

#endif
//  double ompacc_time = read_timer_ms();
  	/* get number of target devices specified by the programmers */
  	int __num_target_devices__ = omp_get_num_active_devices(); /*XXX: = runtime or compiler generated code */

  	omp_device_t *__target_devices__[__num_target_devices__ ];
  	/**TODO: compiler generated code or runtime call to init the __target_devices__ array */
  	int __i__;
  	for (__i__ = 0; __i__ < __num_target_devices__; __i__++) {
  		__target_devices__[__i__] = &omp_devices[__i__]; /* currently this is simple a copy of the pointer */
  	}

  	/**TODO: compiler generated code or runtime call to init the topology */
  	omp_grid_topology_t __top__;
  	int __top_ndims__;
  	/**************************************** dist-specific *****************************************/
  	if (dist == 1 || dist == 2) __top_ndims__ = 1;
  	else /* dist == 3 */__top_ndims__ = 2;
  	/************************************************************************************************/

  	int __top_dims__[__top_ndims__ ];
  	int __top_periodic__[__top_ndims__ ];
  	int __id_map__[__num_target_devices__ ];
  	omp_grid_topology_init_simple(&__top__, __target_devices__, __num_target_devices__, __top_ndims__, __top_dims__,__top_periodic__, __id_map__);

  	int __num_mapped_array__ = 3; /* XXX: need compiler output */
  	omp_data_map_info_t __data_map_infos__[__num_mapped_array__ ];

  	/* f map info */
  	omp_data_map_info_t * __info__ = &__data_map_infos__[0];
  	long f_dims[2];f_dims[0] = n;f_dims[1] = m;
  	omp_data_map_t f_maps[__num_target_devices__];
  	omp_data_map_dist_t f_dist[2];
  	omp_data_map_init_info(__info__, &__top__, f, 2, f_dims, sizeof(REAL), f_maps, OMP_DATA_MAP_TO, f_dist);

  	/* u map info */
  	__info__ = &__data_map_infos__[1];
  	long u_dims[2];u_dims[0] = n;u_dims[1] = m;
  	omp_data_map_t u_maps[__num_target_devices__];
  	omp_data_map_dist_t u_dist[2];
  	//omp_data_map_halo_region_info_t u_halo[2];
  	omp_data_map_init_info(__info__, &__top__, u, 2, u_dims, sizeof(REAL), u_maps, OMP_DATA_MAP_TOFROM, u_dist);

  	/* uold map info */
  	__info__ = &__data_map_infos__[2];
  	long uold_dims[2];uold_dims[0] = n; uold_dims[1] = m;
  	omp_data_map_t uold_maps[__num_target_devices__];
  	omp_data_map_dist_t uold_dist[2];
  	omp_data_map_halo_region_info_t uold_halo[2];
  	omp_data_map_init_info_with_halo(__info__, &__top__, uold, 2, uold_dims, sizeof(REAL),uold_maps,OMP_DATA_MAP_ALLOC, uold_dist, uold_halo);

  	/**************************************** dist-specific *****************************************/
  	if (dist == 1) {
  		omp_data_map_init_dist(&f_dist[0], 0, n, OMP_DATA_MAP_DIST_EVEN, 0);
  		omp_data_map_init_dist(&f_dist[1], 0, m, OMP_DATA_MAP_DIST_FULL, 0);

  		omp_data_map_init_dist(&u_dist[0], 0, n, OMP_DATA_MAP_DIST_EVEN, 0);
  		omp_data_map_init_dist(&u_dist[1], 0, m, OMP_DATA_MAP_DIST_FULL, 0);
  		//omp_map_add_halo_region(&__data_map_infos__[1], 0, 1, 1, 0);

  		omp_data_map_init_dist(&uold_dist[0], 0, n, OMP_DATA_MAP_DIST_EVEN, 0);
  		omp_data_map_init_dist(&uold_dist[1], 0, m, OMP_DATA_MAP_DIST_FULL, 0);
  		omp_map_add_halo_region(&__data_map_infos__[2], 0, 1, 1, 0);
  	} else if (dist == 2) {
  		omp_data_map_init_dist(&f_dist[0], 0, n, OMP_DATA_MAP_DIST_FULL, 0);
  		omp_data_map_init_dist(&f_dist[1], 0, m, OMP_DATA_MAP_DIST_EVEN, 0);

  		omp_data_map_init_dist(&u_dist[0], 0, n, OMP_DATA_MAP_DIST_FULL, 0);
  		omp_data_map_init_dist(&u_dist[1], 0, m, OMP_DATA_MAP_DIST_EVEN, 0);
  		//omp_map_add_halo_region(&__data_map_infos__[1], 1, 1, 1, 0);

  		omp_data_map_init_dist(&uold_dist[0], 0, n, OMP_DATA_MAP_DIST_FULL, 0);
  		omp_data_map_init_dist(&uold_dist[1], 0, m, OMP_DATA_MAP_DIST_EVEN, 0);
  		omp_map_add_halo_region(&__data_map_infos__[2], 1, 1, 1, 0);
  	} else /* dist == 3 */{
  		omp_data_map_init_dist(&f_dist[0], 0, n, OMP_DATA_MAP_DIST_EVEN, 0);
  		omp_data_map_init_dist(&f_dist[1], 0, m, OMP_DATA_MAP_DIST_EVEN, 1);

  		omp_data_map_init_dist(&u_dist[0], 0, n, OMP_DATA_MAP_DIST_EVEN, 0);
  		omp_data_map_init_dist(&u_dist[1], 0, m, OMP_DATA_MAP_DIST_EVEN, 1);
  		//omp_map_add_halo_region(&__data_map_infos__[1], 0, 1, 1, 0);
  		//omp_map_add_halo_region(&__data_map_infos__[1], 1, 1, 1, 0);

  		omp_data_map_init_dist(&uold_dist[0], 0, n, OMP_DATA_MAP_DIST_EVEN, 0);
  		omp_data_map_init_dist(&uold_dist[1], 0, m, OMP_DATA_MAP_DIST_EVEN, 1);
  		omp_map_add_halo_region(&__data_map_infos__[2], 0, 1, 1, 0);
  		omp_map_add_halo_region(&__data_map_infos__[2], 1, 1, 1, 0);
  	}
  	/************************************************************************************************/

  	omp_offloading_info_t __offloading_info__;
  	__offloading_info__.offloadings = (omp_offloading_t *) alloca(sizeof(omp_offloading_t) * __num_target_devices__);
  	/* we use universal args and launcher because axpy can do it */
  	omp_offloading_init_info(&__offloading_info__, &__top__, __target_devices__, OMP_OFFLOADING_DATA, __num_mapped_array__, __data_map_infos__, NULL, NULL);

	/*********** NOW notifying helper thread to work on this offload ******************/
#if DEBUG_MSG
	printf("=========================================== offloading to %d targets ==========================================\n", __num_target_devices__);
#endif
	/* here we do not need sync start */
	omp_offloading_start(__target_devices__,__num_target_devices__, &__offloading_info__);
	printf("----- data copyin .... \n");

	while ((k <= mits)/* && (error > tol)*/) {
		error = 0.0;
		/* Copy new solution into old */
	  	omp_offloading_info_t __off_info_1__;
	  	omp_offloading_t __offs_1__[__num_target_devices__];
	  	__off_info_1__.offloadings = __offs_1__;
	  	/* we use universal args and launcher because axpy can do it */
	  	struct OUT__2__10550__args args_1; args_1.n = n; args_1.m = m;args_1.u = (REAL*)u; args_1.uold = (REAL*)uold;
	  	omp_offloading_init_info(&__off_info_1__, &__top__, __target_devices__, OMP_OFFLOADING_CODE, -1, NULL, OUT__2__10550__launcher, &args_1);
	  	omp_offloading_start(__target_devices__, __num_target_devices__, &__off_info_1__);

		/** halo exchange */
		//printf("----- u <-> uold halo exchange, k: %d, off_info: %X\n", k, &__off_info_1__);

	  	while(1);
	  	omp_data_map_exchange_info_t u_uold_xchange;
	  	omp_data_map_halo_exchange_t x_halos[1];
	  	x_halos[0].map_info = &__data_map_infos__[0]; x_halos[1].x_direction = OMP_DATA_MAP_EXCHANGE_FROM_LEFT_RIGHT; /* uold */
	  	if (dist == 1) {
	  		x_halos[0].x_dim = 0;
	  	}
	  	else if (dist == 2) {
	  		x_halos[0].x_dim = 1;
	  	}
	  	else {
	  		x_halos[0].x_dim = -1; /* means all the dimension */
	  	}

	  	u_uold_xchange.x_halos = x_halos;
	  	u_uold_xchange.num_maps = 1;
	  	omp_data_map_exchange_start(__target_devices__, __num_target_devices__, &u_uold_xchange);

		/* jacobi */
	  	omp_offloading_info_t __off_info_2__;
	  	omp_offloading_t __offs_2__[__num_target_devices__];
	  	__off_info_2__.offloadings = __offs_2__;	  	/* we use universal args and launcher because axpy can do it */
	  	struct OUT__1__10550__args args_2;
	  	args_2.n = n; args_2.m = m; args_2.ax = ax; args_2.ay = ay; args_2.b = b; args_2.omega = omega;args_2.u = (REAL*)u; args_2.uold = (REAL*)uold; args_2.f = (REAL*) f;

	  	REAL __reduction_error__[__num_target_devices__]; args_2.error = __reduction_error__;
	  	int __i__;
		for (__i__ = 0; __i__ < __num_target_devices__;__i__++) {
			__reduction_error__[__i__] = error;
		}

	  	omp_offloading_init_info(&__off_info_2__, &__top__, __target_devices__, OMP_OFFLOADING_CODE, -1, NULL, OUT__1__10550__launcher, &args_2);
	  	omp_offloading_start(__target_devices__, __num_target_devices__, &__off_info_2__);
		for (__i__ = 0; __i__ < __num_target_devices__;__i__++) {
			error += __reduction_error__[__i__];
		}

		/* Error check */
		if ((k % 500) == 0) printf("Finished %d iteration with error =%g\n", k, error);
		error = (sqrt(error) / (n * m));
		k = (k + 1);
		/*  End iteration loop */
	}
	/* copy back u from each device and free others */
	omp_offloading_finish_copyfrom(__target_devices__,__num_target_devices__, &__offloading_info__);

	/*
	ompacc_time = read_timer_ms() - ompacc_time;
	double cpu_total = ompacc_time;
	printf("Total Number of Iterations:%d\n", k);
	printf("Residual:%E\n", error);
*/
#if 0
    /* for profiling */
	REAL f_map_to_elapsed[__num_target_devices__]; /* event 0 */
	REAL u_map_to_elapsed[__num_target_devices__]; /* event 1 */
	REAL kernel_u2uold_elapsed[__num_target_devices__]; /* event 2 */
	REAL halo_exchange_elapsed[__num_target_devices__]; /* event 3 */
	REAL kernel_jacobi_elapsed[__num_target_devices__]; /* event 4, also including the reduction */
	REAL u_map_from_elapsed[__num_target_devices__]; /* event 5 */

	printf("==============================================================================================================================================================================\n");
	printf("=========================== GPU Results (%d GPUs) for jacobi: u[][](tofrom), f[][](to), uold[][](alloc) size: [%d][%d], time in ms (s/1000) ===============================\n", __num_target_devices__, n, m);
	REAL f_map_to_accumulated = 0.0;
	REAL u_map_to_accumulated = 0.0;
	REAL kernel_u2uold_accumulated = 0.0;
	REAL halo_exchange_accumulated = 0.0;
	REAL kernel_jacobi_accumulated = 0.0;
	REAL u_map_from_accumulated = 0.0;
	REAL streamCreate_accumulated = 0.0;
	for (__i__ = 0; __i__ < __num_target_devices__; __i__++) {
		f_map_to_elapsed[__i__] = omp_stream_event_elapsed_ms(&__dev_stream__[__i__], 0);
		u_map_to_elapsed[__i__] = omp_stream_event_elapsed_ms(&__dev_stream__[__i__], 1);
		kernel_u2uold_elapsed[__i__] = __dev_stream__[__i__].elapsed[2]; /* event 2 */
		halo_exchange_elapsed[__i__] = __dev_stream__[__i__].elapsed[3]; /* event 3 */
		kernel_jacobi_elapsed[__i__] = __dev_stream__[__i__].elapsed[4]; /* event 4, also including the reduction */
		u_map_from_elapsed[__i__] = omp_stream_event_elapsed_ms(&__dev_stream__[__i__], 5);
		REAL total = f_map_to_elapsed[__i__] + u_map_to_elapsed[__i__] + kernel_u2uold_elapsed[__i__] + halo_exchange_elapsed[__i__] + kernel_jacobi_elapsed[__i__]  + u_map_from_elapsed[__i__];
		printf("device: %d, total: %4f\n", __i__, total);
		printf("\t\tstreamCreate overhead: %4f\n", streamCreate_elapsed[__i__]);
		printf("\t\tbreakdown: f map_to: %4f; u map_to: %4f; u2uold kernel: %4f; halo_exchange: %4f; kernel_jacobi: %4f, u map_from: %g\n", f_map_to_elapsed[__i__], u_map_to_elapsed[__i__], kernel_u2uold_elapsed[__i__], halo_exchange_elapsed[__i__], kernel_jacobi_elapsed[__i__], u_map_from_elapsed[__i__]);
		printf("\t\tbreakdown: f map_to (u and f): %4f; u2uold kernel: %4f; halo_exchange: %4f; kernel_jacobi: %4f, map_from (u): %g\n", f_map_to_elapsed[__i__] + u_map_to_elapsed[__i__], kernel_u2uold_elapsed[__i__], halo_exchange_elapsed[__i__], kernel_jacobi_elapsed[__i__], u_map_from_elapsed[__i__]);
		f_map_to_accumulated += f_map_to_elapsed[__i__];
		u_map_to_accumulated += u_map_to_elapsed[__i__];
		kernel_u2uold_accumulated += kernel_u2uold_elapsed[__i__];
		halo_exchange_accumulated += halo_exchange_elapsed[__i__];
		kernel_jacobi_accumulated += kernel_jacobi_elapsed[__i__];
		u_map_from_accumulated += u_map_from_elapsed[__i__];
		streamCreate_accumulated += streamCreate_elapsed[__i__];
	}
	REAL gpu_total = f_map_to_accumulated + u_map_to_accumulated + kernel_u2uold_accumulated + halo_exchange_accumulated + kernel_jacobi_accumulated + u_map_from_accumulated;
	printf("ACCUMULATED GPU time (%d GPUs): %4f\n", __num_target_devices__ , gpu_total);
	printf("\t\tstreamCreate overhead: %4f\n",streamCreate_accumulated);
	printf("\t\tbreakdown: f map_to: %4f; u map_to: %4f; u2uold kernel: %4f; halo_exchange: %4f; kernel_jacobi: %4f, u map_from: %g\n", f_map_to_accumulated , u_map_to_accumulated , kernel_u2uold_accumulated , halo_exchange_accumulated , kernel_jacobi_accumulated , u_map_from_accumulated);
	printf("\t\tbreakdown: f map_to (u and f): %4f; u2uold kernel: %4f; halo_exchange: %4f; kernel_jacobi: %4f, map_from (u): %g\n", f_map_to_accumulated + u_map_to_accumulated, kernel_u2uold_accumulated , halo_exchange_accumulated , kernel_jacobi_accumulated , u_map_from_accumulated);

	printf("AVERAGE GPU time (per GPU): %4f\n", gpu_total/__num_target_devices__);
	printf("\t\tbreakdown: f map_to: %4f; u map_to: %4f; u2uold kernel: %4f; halo_exchange: %4f; kernel_jacobi: %4f, u map_from: %g\n", f_map_to_accumulated/__num_target_devices__, u_map_to_accumulated/__num_target_devices__, kernel_u2uold_accumulated/__num_target_devices__, halo_exchange_accumulated/__num_target_devices__, kernel_jacobi_accumulated/__num_target_devices__, u_map_from_accumulated);
	printf("\t\tbreakdown: f map_to (u and f): %4f; u2uold kernel: %4f; halo_exchange: %4f; kernel_jacobi: %4f, map_from (u): %g\n", f_map_to_accumulated/__num_target_devices__ + u_map_to_accumulated/__num_target_devices__, kernel_u2uold_accumulated/__num_target_devices__, halo_exchange_accumulated/__num_target_devices__, kernel_jacobi_accumulated/__num_target_devices__, u_map_from_accumulated);

	printf("----------------------------------------------------------------\n");
	printf("Total time measured from CPU: %4f\n", cpu_total);
	printf("Total time measured without streamCreate %4f\n", (cpu_total-streamCreate_accumulated));
	printf("AVERAGE total (CPU cost+GPU) per GPU: %4f\n", cpu_total/__num_target_devices__);
	printf("Total CPU cost: %4f\n", cpu_total - gpu_total);
	printf("AVERAGE CPU cost per GPU: %4f\n", (cpu_total-gpu_total)/__num_target_devices__);
	printf("==========================================================================================================================================\n");
#endif
}
