#include<cuda.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <stdio.h>




double
read_timer() {
    struct timespec ts;
#if defined(CLOCK_MONOTONIC_PRECISE)
  /* BSD. --------------------------------------------- */
  const clockid_t id = CLOCK_MONOTONIC_PRECISE;
#elif defined(CLOCK_MONOTONIC_RAW)
  /* Linux. ------------------------------------------- */
  const clockid_t id = CLOCK_MONOTONIC_RAW;
#elif defined(CLOCK_HIGHRES)
  /* Solaris. ----------------------------------------- */
  const clockid_t id = CLOCK_HIGHRES;
#elif defined(CLOCK_MONOTONIC)
  /* AIX, BSD, Linux, POSIX, Solaris. ----------------- */
  const clockid_t id = CLOCK_MONOTONIC;
#elif defined(CLOCK_REALTIME)
  /* AIX, BSD, HP-UX, Linux, POSIX. ------------------- */
  const clockid_t id = CLOCK_REALTIME;
#else
    const clockid_t id = (clockid_t) - 1;    /* Unknown. */
#endif

    if (id != (clockid_t) - 1 && clock_gettime(id, &ts) != -1)
        return (double) ts.tv_sec + (double) ts.tv_nsec / 1000000000.0;
}

#define __global__
__device__ double d_flops;

__global__ void Add_Mul() {
    // need to initialise differently otherwise compiler might optimise away

    double add = M_PI;
    double mul = 1.0 + 1e-8;
    int ops = 1000000;

    double sum1 = 0.1, sum2 = -0.1, sum3 = 0.2, sum4 = -0.2, sum5 = 0.0;
    double mul1 = 1.0, mul2 = 1.1, mul3 = 1.2, mul4 = 1.3, mul5 = 1.4;
    int loops = ops / 10;    // we have 10 floating point ops inside the loop
    double expected = 5.0 * add * loops + (sum1 + sum2 + sum3 + sum4 + sum5)
                      + pow(mul, loops) * (mul1 + mul2 + mul3 + mul4 + mul5);

    int i;
    for (i = 0; i < loops; i++) {
        mul1 *= mul;
        mul2 *= mul;
        mul3 *= mul;
        mul4 *= mul;
        mul5 *= mul;
        sum1 += add;
        sum2 += add;
        sum3 += add;
        sum4 += add;
        sum5 += add;
    }

    d_flops =
            sum1 + sum2 + sum3 + sum4 + sum5 + mul1 + mul2 + mul3 + mul4 + mul5 -
            expected;
}


int
main(int argc, char *argv[]) {

    //int n = 1000000;
    double timer = read_timer();
    Add_Mul << < 1024, 32 >> > ();
    timer = read_timer() - timer;
    typeof(d_flops) flops;  // Indicates that kernel is returning a single value

    cudaMemcpy(&flops, "d_flops", (sizeof(flops)), cudaMemcpyDeviceToHost);
    flops = (1 * 1024 * 32) / timer /
            1e3; // Value was over flowing, so I self-optimized it in order to make it stay within the range.
//1024*32 is added because this will be 1024 threads in each of the 32 blocks working in parallel to compute the result. Hence GFlops is related to how many threads are made to work.


    printf("GPU flops is %f \n", flops);
}
