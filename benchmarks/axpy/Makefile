CUDA_PATH=/opt/NVIDIA/cuda/

default: axpy-nvgpu-itlmic
all: axpy-cpu axpy-nvgpu axpy-nvgpu-itlmic

BASE_CFLAGS = -I../../runtime -I../../util -I.
CFLAGS = ${BASE_CFLAGS}
# To enable breakdown timing and profiling, to disable, comment the following line
BASE_CFLAGS += -g -DOMP_BREAKDOWN_TIMING -DPROFILE_PLOT=1

# CFLAGS for NVGPU_CUDA support
NVCC_CFLAGS += -DDEVICE_NVGPU_CUDA_SUPPORT=1 -Xcompiler -fopenmp -I${CUDA_PATH}/include
# To enable NVGPU_CUDA virtual shared memory support
# NVCC_CFLAGS += -DDEVICE_NVGPU_CUDA_VSHAREDM=1

# To enable ITLMIC support
ICC_CFLAGS += -DDEVICE_ITLMIC_SUPPORT=1

LDFLAGS = -L/usr/lib/gcc/x86_64-linux-gnu/4.8.2 -lgomp -lm -lrt -lpthread
#gpu.sec
LDFLAGS = -L/usr/lib/gcc/x86_64-redhat-linux/4.4.6 -lgomp -lm -lrt -lpthread

RUNTIME_SOURCES = ../../runtime/homp.c ../../runtime/dev_xthread.c ../../util/iniparser.c ../../util/dictionary.c

axpy-nvgpu: cleanobj
	nvcc $(BASE_CFLAGS) $(NVCC_CFLAGS) ${RUNTIME_SOURCES} axpy_ompacc.c axpy.c -c
	nvcc $(BASE_CFLAGS) $(NVCC_CFLAGS) ../../runtime/homp_dev.c -c
	nvcc $(BASE_CFLAGS) $(NVCC_CFLAGS) axpy_cpu_omp_kernel.c -c
	nvcc $(BASE_CFLAGS) $(NVCC_CFLAGS) axpy_nvgpu_cuda_kernel.cu -c
	nvcc $(LDFLAGS) -g *.o -o $@

axpy-nvgpu-itlmic: cleanobj
	nvcc $(BASE_CFLAGS) $(NVCC_CFLAGS) $(ICC_CFLAGS) ${RUNTIME_SOURCES} axpy_ompacc.c axpy.c -c
	nvcc $(BASE_CFLAGS) $(NVCC_CFLAGS) $(ICC_CFLAGS) ../../runtime/homp_dev.c -c
	nvcc $(BASE_CFLAGS) $(NVCC_CFLAGS) $(ICC_CFLAGS) axpy_cpu_omp_kernel.c -c
#	icc $(BASE_CFLAGS) $(NVCC_CFLAGS) $(ICC_CFLAGS) -openmp axpy_cpu_omp_kernel.c -c
	nvcc $(BASE_CFLAGS) $(NVCC_CFLAGS) $(ICC_CFLAGS) axpy_nvgpu_cuda_kernel.cu -c
	icc $(BASE_CFLAGS) $(NVCC_CFLAGS) $(ICC_CFLAGS)  -openmp axpy_itlmic_kernel.c -c
	nvcc $(LDFLAGS) -g *.o -o $@

axpy-cpu: cleanobj
	gcc $(BASE_CFLAGS) ${RUNTIME_SOURCES} axpy_ompacc.c axpy.c -c
	gcc $(BASE_CFLAGS) ../../runtime/homp_dev.c -c
	gcc $(BASE_CFLAGS) axpy_cpu_omp_kernel.c -c
	gcc *.o -o $@ $(LDFLAGS)

cleanobj:
	rm -rf *.o

clean:
	rm -rf *.o axpy-* *.plot core

