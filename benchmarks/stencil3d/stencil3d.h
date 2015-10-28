#ifndef OFFOMP_STENCIL3D_H
#define OFFOMP_STENCIL3D_H
#ifdef __cplusplus
extern "C" {
#endif
// flexible between REAL and double
#define REAL float

/* this should go to a compiler generated header file and to be included in the souce files (launcher, off pragma call, etc)*/
struct stencil3d_off_args {
    long n; long m; long k; REAL *u; int radius; REAL *coeff; int num_its;

    long u_dimX;
    long u_dimY;
    long u_dimZ;
    int coeff_dimX;
    REAL * coeff_center;
    REAL * uold;
};

extern int dist_dim;
extern int dist_policy;

extern void stencil3d_omp_mdev_off_launcher(omp_offloading_t *off, void *args);
extern void stencil3d_omp_mdev_iterate_off_launcher(omp_offloading_t *off, void *args);

#ifdef __cplusplus
 }
#endif
#endif //OFFOMP_STENCIL3D_H
