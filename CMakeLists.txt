#cmake_minimum_required(VERSION 3.1)
cmake_minimum_required(VERSION 2.8)
project(homp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(RUNTIME_SOURCE_FILES
    runtime/dev_xthread.c
    runtime/homp.c
    runtime/homp.h
    runtime/homp_dev.c
    util/dictionary.c
    util/dictionary.h
    util/iniparser.c
    util/iniparser.h)

set(AXPY_SOURCE_FILES
    benchmarks/axpy/axpy.c
    benchmarks/axpy/axpy.h
    benchmarks/axpy/axpy_ompacc.c
    benchmarks/axpy/axpy_cpu_omp_kernel.c
    benchmarks/axpy/axpy_nvgpu_cuda_kernel.cu
    benchmarks/axpy/axpy_itlmic_kernel.c)

set(SUM_SOURCE_FILES
        benchmarks/sum/sum.c
        benchmarks/sum/sum.h
        benchmarks/sum/sum_ompacc.c
        benchmarks/sum/sum_cpu_omp_kernel.c
        benchmarks/sum/sum_nvgpu_cuda_kernel.cu
        benchmarks/sum/sum_itlmic_kernel.c)

set(MATVEC_SOURCE_FILES
    benchmarks/matvec/matvec.c
    benchmarks/matvec/matvec.h
    benchmarks/matvec/matvec_ompacc.c
        benchmarks/matvec/matvec_cpu_omp_kernel.c
        benchmarks/matvec/matvec_itlmic_kernel.c
        benchmarks/matvec/matvec_nvgpu_cuda_kernel.cu
        )

set(MATMUL_SOURCE_FILES
        benchmarks/matmul/matmul_mdev.c
        benchmarks/matmul/matmul.h
        benchmarks/matmul/matmul_cpu_omp_kernel.c
        benchmarks/matmul/matmul_itlmic_kernel.c
        benchmarks/matmul/matmul_nvgpu_cuda_kernel.cu
        )

set(RADIX_SOURCE_FILES
    benchmarks/radixsort/radix.c)

set(STENCIL2D_SOURCE_FILES
        benchmarks/stencil2d/stencil2d_mdev.c
        benchmarks/stencil2d/stencil2d_mdev_iteration.c
        benchmarks/stencil2d/stencil2d.h
        benchmarks/stencil2d/stencil2d.c
        benchmarks/stencil2d/stencil2d_cpu_omp_kernel.c
        benchmarks/stencil2d/stencil2d_itlmic_kernel.c
        benchmarks/stencil2d/stencil2d_nvgpu_cuda_kernel.cu
        )

set(BM2D_SOURCE_FILES
        benchmarks/bm2d/bm2d_mdev.c
        benchmarks/bm2d/bm2d_mdev_iteration.c
        benchmarks/bm2d/bm2d.h
        benchmarks/bm2d/bm2d.c
        benchmarks/bm2d/bm2d_cpu_omp_kernel.c
        benchmarks/bm2d/bm2d_itlmic_kernel.c
        benchmarks/bm2d/bm2d_nvgpu_cuda_kernel.cu
        )

set(STENCIL3D_SOURCE_FILES
        benchmarks/stencil3d/stencil3d_mdev.c
        benchmarks/stencil3d/stencil3d_mdev_iteration.c
        benchmarks/stencil3d/stencil3d.h
        benchmarks/stencil3d/stencil3d.c
        benchmarks/stencil3d/stencil3d_cpu_omp_kernel.c
        benchmarks/stencil3d/stencil3d_itlmic_kernel.c
        benchmarks/stencil3d/stencil3d_nvgpu_cuda_kernel.cu
        )

set(JACOBI_SOURCE_FILES
        benchmarks/jacobi/jacobi_ompacc_mdev.c
        benchmarks/jacobi/jacobi.h
        benchmarks/jacobi/jacobi_cpu_omp_kernel.c
        benchmarks/jacobi/jacobi_itlmic_kernel.c
        benchmarks/jacobi/jacobi_nvgpu_cuda_kernel.cu
        )

include_directories("runtime")

add_executable(axpy-cpu ${RUNTIME_SOURCE_FILES} ${AXPY_SOURCE_FILES})
add_executable(sum-cpu ${RUNTIME_SOURCE_FILES} ${SUM_SOURCE_FILES})
add_executable(matvec-cpu ${RUNTIME_SOURCE_FILES} ${MATVEC_SOURCE_FILES})
add_executable(matmul-cpu ${RUNTIME_SOURCE_FILES} ${MATMUL_SOURCE_FILES})
add_executable(radix-cpu ${RUNTIME_SOURCE_FILES} ${RADIX_SOURCE_FILES})
add_executable(stencil2d-cpu ${RUNTIME_SOURCE_FILES} ${STENCIL2D_SOURCE_FILES})
add_executable(jacobi-cpu ${RUNTIME_SOURCE_FILES} ${JACOBI_SOURCE_FILES})
target_link_libraries(axpy-cpu pthread m)
target_link_libraries(matvec-cpu pthread m)
target_link_libraries(matmul-cpu pthread m)
target_link_libraries(radix-cpu pthread m)
target_link_libraries(stencil2d-cpu pthread m)
target_link_libraries(jacobi-cpu pthread m)

