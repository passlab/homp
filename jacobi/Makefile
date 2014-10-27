TEST_INCLUDES = -I../runtime -I.
TEST_LINK = -lpthread -lm

NVGPU_CUDA_PATH=/opt/NVIDIA/cuda/

jacobi-thsim:	
#	gcc $(TEST_INCLUDES) -DDEBUG_MSG -g -O0 ../../runtime/homp.c ../../runtime/homp_dev.c jacobi_ompacc_mdev.c -c
	gcc $(TEST_INCLUDES) -g -O0 ../runtime/homp.c ../runtime/homp_dev.c jacobi_ompacc_mdev.c -c
	gcc $(TEST_INCLUDES) -g *.o -o $@ ${TEST_LINK}

jacobi-nvgpu:
	nvcc $(TEST_INCLUDES) -g -I${NVGPU_CUDA_PATH}/include -DDEVICE_NVGPU_SUPPORT=1 ../runtime/homp.c ../runtime/homp_dev.c jacobi_ompacc_mdev.cu  -c
	nvcc $(TEST_INCLUDES) -g *.o -o $@ ${TEST_LINK}

clean:
	rm -rf *.o jacobi-*
