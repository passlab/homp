#cmake_minimum_required(VERSION 3.1)
cmake_minimum_required(VERSION 2.8)
project(offomp)

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
    benchmarks/axpy/axpy_ompacc.c)

set(MATMUL_SOURCE_FILES
    benchmarks/matmul/matmul_mdev.c)

set(JACOBI_SOURCE_FILES
    benchmarks/jacobi/jacobi_ompacc_mdev.c)

include_directories("runtime")

add_executable(axpy-thsim ${RUNTIME_SOURCE_FILES} ${AXPY_SOURCE_FILES})
add_executable(matmul-thsim ${RUNTIME_SOURCE_FILES} ${MATMUL_SOURCE_FILES})
add_executable(jacobi-thsim ${RUNTIME_SOURCE_FILES} ${JACOBI_SOURCE_FILES})
target_link_libraries(axpy-thsim pthread m)
target_link_libraries(matmul-thsim pthread m)
target_link_libraries(jacobi-thsim pthread m)

