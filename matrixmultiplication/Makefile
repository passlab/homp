TEST_INCLUDES = -I../runtime -I.
TEST_LINK = -lpthread -lm

NVGPU_CUDA_PATH=/opt/NVIDIA/cuda/

matmul-thsim:	
#	gcc $(TEST_INCLUDES) -DOMP_BREAKDOWN_TIMING -g ../runtime/homp.c ../runtime/homp_dev.c ../runtime/dev_xthread.c matmul_mdev.c -c
	gcc $(TEST_INCLUDES) -g  -DOMP_BREAKDOWN_TIMING=1 ../runtime/homp.c ../runtime/homp_dev.c ../runtime/dev_xthread.c matmul_mdev.c -c
	gcc $(TEST_INCLUDES) -g *.o -o $@ ${TEST_LINK}

matmul-nvgpu:
	nvcc $(TEST_INCLUDES) -I${NVGPU_CUDA_PATH}/include -DDEVICE_NVGPU_SUPPORT=1 -DOMP_BREAKDOWN_TIMING=1 ../runtime/homp.c ../runtime/homp_dev.c ../runtime/dev_xthread.c matmul_mdev.cu  -c
	nvcc $(TEST_INCLUDES) -g *.o -o $@ ${TEST_LINK}

clean:
	rm -rf *.o matmul-*
