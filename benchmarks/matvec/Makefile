CUDA_PATH=/opt/nvidia/cuda

default: matvec-nvgpu-itlmic
all: matvec-cpu matvec-nvgpu matvec-nvgpu-itlmic

BASE_CFLAGS = -I../../runtime -I../../util -I.
CFLAGS = ${BASE_CFLAGS}
# To enable breakdown timing and profiling, to disable, comment the following line
BASE_CFLAGS += -g -DOMP_BREAKDOWN_TIMING -DPROFILE_PLOT=1

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

matvec-nvgpu: cleanobj
	nvcc $(BASE_CFLAGS) $(NVCC_CFLAGS) ${RUNTIME_SOURCES} matvec_ompacc.c matvec.c -c
	nvcc $(BASE_CFLAGS) $(NVCC_CFLAGS) ../../runtime/homp_dev.c -c
#	nvcc $(BASE_CFLAGS) $(NVCC_CFLAGS) -Xcompiler -fopenmp matvec_cpu_omp_kernel.c -c
	icc $(BASE_CFLAGS) $(ICC_CFLAGS) -openmp matvec_cpu_omp_kernel.c -c
	nvcc $(BASE_CFLAGS) $(NVCC_CFLAGS) matvec_nvgpu_cuda_kernel.cu -c
	icc -openmp -g *.o -o $@ -L${CUDA_PATH}/lib64 -lcudart -lpthread

matvec-nvgpu-itlmic: cleanobj
	nvcc $(BASE_CFLAGS) $(NVCC_CFLAGS) $(ICC_CFLAGS) ${RUNTIME_SOURCES} matvec_ompacc.c matvec.c -c
	nvcc $(BASE_CFLAGS) $(NVCC_CFLAGS) $(ICC_CFLAGS) ../../runtime/homp_dev.c -c
#	nvcc $(BASE_CFLAGS) $(NVCC_CFLAGS) $(ICC_CFLAGS) -Xcompiler -fopenmp matvec_cpu_omp_kernel.c -c
	icc $(BASE_CFLAGS) $(ICC_CFLAGS) -openmp matvec_cpu_omp_kernel.c -c
	nvcc $(BASE_CFLAGS) $(NVCC_CFLAGS) $(ICC_CFLAGS) matvec_nvgpu_cuda_kernel.cu -c
	icc $(BASE_CFLAGS) $(NVCC_CFLAGS) $(ICC_CFLAGS)  -openmp matvec_itlmic_kernel.c -c
	icc -openmp -g *.o -o $@ -L${CUDA_PATH}/lib64 -lcudart -lpthread

matvec-cpu: cleanobj
	gcc $(BASE_CFLAGS) ${RUNTIME_SOURCES} matvec_ompacc.c matvec.c -c
	gcc $(BASE_CFLAGS) ../../runtime/homp_dev.c -c
	gcc $(BASE_CFLAGS) matvec_cpu_omp_kernel.c -c
	gcc *.o -o $@ $(LDFLAGS)

cleanobj:
	rm -rf *.o

clean:
	rm -rf *.o matvec-* *.plot core core.*