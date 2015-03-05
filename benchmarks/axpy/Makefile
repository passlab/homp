TEST_INCLUDES = -I../../runtime -I.
TEST_LINK = -lm -lpthread

#NVGPU_CUDA_PATH=/opt/NVIDIA/cuda/
NVGPU_CUDA_PATH=/usr/local/cuda-5.5/
NVGPU_CUDA_PATH=/APPS/cuda/include

all: axpy-thsim axpy-nvgpu

axpy-thsim:
	gcc $(TEST_INCLUDES) -g -DOMP_BREAKDOWN_TIMING -DPROFILE_PLOT=1 -fopenmp ../../runtime/homp.c ../../runtime/homp_dev.c ../../runtime/dev_xthread.c rose_axpy_ompacc.c rose_axpy.c -c
#	gcc $(TEST_INCLUDES) -g ../../runtime/homp.c ../../runtime/homp_dev.c ../../runtime/dev_xthread.c rose_axpy_ompacc.c rose_axpy.c -c
	gcc $(TEST_INCLUDES) -g *.o -o $@ -L/usr/lib/gcc/x86_64-linux-gnu/4.8.2/ -lgomp ${TEST_LINK}

axpy-nvgpu:
	nvcc $(TEST_INCLUDES) -g -I${NVGPU_CUDA_PATH}/include -Xcompiler -fopenmp -DDEVICE_NVGPU_SUPPORT=1 -DOMP_BREAKDOWN_TIMING -DPROFILE_PLOT=1 ../../runtime/homp.c ../../runtime/homp_dev.c ../../runtime/dev_xthread.c rose_axpy_ompacc.cu rose_axpy.c -c
	nvcc $(TEST_INCLUDES) -g *.o -o $@ -L/usr/lib/gcc/x86_64-redhat-linux/4.4.6 -lgomp ${TEST_LINK}

clean:
	rm -rf *.o axpy-thsim axpy-nvgpu

