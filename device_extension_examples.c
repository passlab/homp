#include <stdio.h>
#include <stdlib.h>

/* device clause should support the following examples and 
 * detail grammar is specified in https://github.com/passlab/passlab.github.io/wiki/HOMP-Extensions-to-OpenMP
 
 device(0:*): all devices
 device(0, 2, 3, 5)
 device(0:2, 4:2) which is the list of 0,1,4,5.
 device(0:*:OMP_DEVICE_NVGPU): ALL GPU devices and consider OMP_DEVICE_NVGPU as a enum/integer so just pass it as 
 expression.  
 */

void axpy(int N, float *Y, float *X, float a) {
    int i;

/* official */
#pragma omp target device(0)
#pragma omp parallel for
    for (i = 0; i < N; ++i)
        Y[i] += a * X[i];
}


void axpy_ext(int N, float *Y, float *X, float a) {
    int i;
/* the rest are extensions */
#pragma omp target device(0:*)
#pragma omp parallel for
    for (i = 0; i < N; ++i)
        Y[i] += a * X[i];

#pragma omp target device(0,2,3,5)
#pragma omp parallel for
    for (i = 0; i < N; ++i)
        Y[i] += a * X[i];

#pragma omp target device(0:2, 4:2)
#pragma omp parallel for
    for (i = 0; i < N; ++i)
        Y[i] += a * X[i];

#pragma omp target device(0:8:OMP_DEVICE_NVGPU)
#pragma omp parallel for
    for (i = 0; i < N; ++i)
        Y[i] += a * X[i];

#pragma omp target device(0:*:OMP_DEVICE_NVGPU)
#pragma omp parallel for
    for (i = 0; i < N; ++i)
        Y[i] += a * X[i];
}

