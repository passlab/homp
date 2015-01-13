TEST_INCLUDES = -I../runtime -I.
TEST_LINK = -lpthread -lm

NVGPU_CUDA_PATH=/opt/NVIDIA/cuda/

jacobi-thsim:	
	gcc $(TEST_INCLUDES) -DOMP_BREAKDOWN_TIMING -g -O0 ../runtime/homp.c ../runtime/homp_dev.c ../runtime/dev_xthread.c  jacobi_ompacc_mdev.c -c
#	gcc $(TEST_INCLUDES) -g -O0 ../runtime/homp.c ../runtime/homp_dev.c ../runtime/dev_xthread.c  jacobi_ompacc_mdev.c -c
	gcc $(TEST_INCLUDES) -g *.o -o $@ ${TEST_LINK}

jacobi-nvgpu:
	nvcc $(TEST_INCLUDES) -g -I${NVGPU_CUDA_PATH}/include -Xcompiler -fopenmp -DDEVICE_NVGPU_SUPPORT=1 -DOMP_BREAKDOWN_TIMING ../runtime/homp.c ../runtime/homp_dev.c ../runtime/dev_xthread.c jacobi_ompacc_mdev.cu  -c
	nvcc $(TEST_INCLUDES) -g *.o -o $@ -L/usr/lib/gcc/x86_64-redhat-linux/4.4.6 -lgomp ${TEST_LINK}

clean:
	rm -rf *.o jacobi-*
