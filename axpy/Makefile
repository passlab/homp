TEST_INCLUDES = -I../runtime -I.
TEST_LINK = -lm -lpthread

NVGPU_CUDA_PATH=/opt/NVIDIA/cuda/

axpy-thsim:
	gcc $(TEST_INCLUDES) -g -DOMP_BREAKDOWN_TIMING ../runtime/homp.c ../runtime/homp_dev.c ../runtime/dev_xthread.c rose_axpy_ompacc.c rose_axpy.c -c
#	gcc $(TEST_INCLUDES) -g ../runtime/homp.c ../runtime/homp_dev.c ../runtime/dev_xthread.c rose_axpy_ompacc.c rose_axpy.c -c
	gcc $(TEST_INCLUDES) -g *.o -o $@ ${TEST_LINK}

axpy-nvgpu:
	nvcc $(TEST_INCLUDES) -I${NVGPU_CUDA_PATH}/include -DDEVICE_NVGPU_SUPPORT=1 ../runtime/homp.c ../runtime/homp_dev.c rose_axpy_ompacc.cu rose_axpy.c -c
	nvcc $(TEST_INCLUDES) -g *.o -o $@ ${TEST_LINK}

clean:
	rm -rf *.o axpy-thsim axpy-nvgpu

