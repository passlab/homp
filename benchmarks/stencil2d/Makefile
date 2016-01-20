CUDA_PATH=/opt/nvidia/cuda

default: stencil2d-nvgpu-itlmic
all: stencil2d-cpu stencil2d-nvgpu stencil2d-nvgpu-itlmic

BASE_CFLAGS = -I../../runtime -I../../util -I.
CFLAGS = ${BASE_CFLAGS}
# To enable breakdown timing and profiling, to disable, comment the following line
BASE_CFLAGS += -g -DOMP_BREAKDOWN_TIMING 
# BASE_CFLAGS += DPROFILE_PLOT=1

# CFLAGS for NVGPU_CUDA support
NVCC_CFLAGS += -DDEVICE_NVGPU_CUDA_SUPPORT=1 -I${CUDA_PATH}/include
# To enable NVGPU_CUDA virtual shared memory support
# NVCC_CFLAGS += -DDEVICE_NVGPU_CUDA_VSHAREDM=1

# To enable ITLMIC support
ICC_CFLAGS += -DDEVICE_ITLMIC_SUPPORT=1
# -L/opt/intel/lib/intel64/ -loffload

#fornax
LDFLAGS = -L/usr/lib/gcc/x86_64-redhat-linux/4.4.7 -lgomp -lm -lrt -lpthread

RUNTIME_SOURCES = ../../runtime/homp.c ../../runtime/dev_xthread.c ../../util/iniparser.c ../../util/dictionary.c

stencil2d-nvgpu: cleanobj
	nvcc $(BASE_CFLAGS) $(NVCC_CFLAGS) ${RUNTIME_SOURCES} stencil2d.c stencil2d_mdev.c stencil2d_mdev_iteration.c -c
	nvcc $(BASE_CFLAGS) $(NVCC_CFLAGS) ../../runtime/homp_dev.c -c
	nvcc $(BASE_CFLAGS) $(NVCC_CFLAGS) -Xcompiler -fopenmp stencil2d_cpu_omp_kernel.c -c
	nvcc $(BASE_CFLAGS) $(NVCC_CFLAGS) stencil2d_nvgpu_cuda_kernel.cu -c
	nvcc ${LDFLAGS} -g *.o -o $@ -L${CUDA_PATH}/lib64 -lcudart -lpthread -lcublas

stencil2d-nvgpu-itlmic: cleanobj
	nvcc $(BASE_CFLAGS) $(NVCC_CFLAGS) $(ICC_CFLAGS) ${RUNTIME_SOURCES} stencil2d.c stencil2d_mdev.c stencil2d_mdev_iteration.c -c
	icc $(BASE_CFLAGS) $(NVCC_CFLAGS) $(ICC_CFLAGS) ../../runtime/homp_dev.c -c
#	nvcc $(BASE_CFLAGS) $(NVCC_CFLAGS) $(ICC_CFLAGS) -Xcompiler -fopenmp stencil2d_cpu_omp_kernel.c -c
	icc $(BASE_CFLAGS) $(ICC_CFLAGS) -openmp stencil2d_cpu_omp_kernel.c -c
	nvcc $(BASE_CFLAGS) $(NVCC_CFLAGS) $(ICC_CFLAGS) stencil2d_nvgpu_cuda_kernel.cu -c
	icc $(BASE_CFLAGS) $(NVCC_CFLAGS) $(ICC_CFLAGS)  -openmp stencil2d_itlmic_kernel.c -c
	icc -openmp -g *.o -o $@ -L${CUDA_PATH}/lib64 -lcudart -lpthread -lcublas

stencil2d-cpu: cleanobj
	gcc $(BASE_CFLAGS) ${RUNTIME_SOURCES}  stencil2d.c stencil2d_mdev.c stencil2d_mdev_iteration.c -c
	gcc $(BASE_CFLAGS) ../../runtime/homp_dev.c -c
	gcc $(BASE_CFLAGS) stencil2d_cpu_omp_kernel.c -c
	gcc *.o -o $@ $(LDFLAGS)

cleanobj:
	rm -rf *.o

clean:
	rm -rf *.o stencil2d-* *.plot core core.*
