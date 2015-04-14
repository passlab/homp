TEST_INCLUDES = -I../../runtime -I../../util -I.
TEST_LINK = -lm -lrt -lpthread

#NVGPU_CUDA_PATH=/opt/NVIDIA/cuda/
NVGPU_CUDA_PATH=/usr/local/cuda-5.5/
NVGPU_CUDA_PATH=/APPS/cuda/include

all: matvec-thsim matvec-nvgpu

matvec-thsim:
	gcc $(TEST_INCLUDES) -g -DOMP_BREAKDOWN_TIMING -DPROFILE_PLOT=1 -fopenmp ../../runtime/homp.c ../../runtime/homp_dev.c ../../runtime/dev_xthread.c ../../runtime/dev_xthread.c ../../util/iniparser.c ../../util/dictionary.c matvec_ompacc.c matvec.c -c -w

	gcc $(TEST_INCLUDES) -g *.o -o $@ -L/usr/lib/gcc/x86_64-linux-gnu/4.8.2/ -lgomp ${TEST_LINK}

matvec-nvgpu:
	nvcc $(TEST_INCLUDES) -g -I${NVGPU_CUDA_PATH}/include -Xcompiler -fopenmp -DDEVICE_NVGPU_SUPPORT=1 -DOMP_BREAKDOWN_TIMING -DPROFILE_PLOT=1 ../../runtime/homp.c ../../runtime/homp_dev.c ../../runtime/dev_xthread.c ../../util/iniparser.c ../../util/dictionary.c  matvec_ompacc.cu matvec.c -c -w
	nvcc $(TEST_INCLUDES) -g *.o -o $@ -L/usr/lib/gcc/x86_64-redhat-linux/4.4.6 -lgomp ${TEST_LINK} -w

clean:
	rm -rf *.o matvec-thsim matvec-nvgpu

