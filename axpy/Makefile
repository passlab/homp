TEST_INCLUDES = -I../runtime -I.
TEST_LINK = -lm -lpthread

axpy:
#	${ROSE_OMPTEST}/roseompacc ${TEST_INCLUDES} ${TEST_FLAGS} -rose:skipfinalCompileStep axpy_ompacc.c
#	${ROSE_OMPTEST}/roseomp ${TEST_INCLUDES} ${TEST_FLAGS} -rose:skipfinalCompileStep axpy_omp.c
#	${ROSE_OMPTEST}/roseomp ${TEST_INCLUDES} ${TEST_FLAGS} -rose:skipfinalCompileStep axpy.c
#	nvcc $(TEST_INCLUDES) ${TEST_LINK}  -DUSE_ROSE_GOMP_OPENMP_LIBRARY ../runtime/homp.c $(ROSE_XOMPSRC)/xomp_cuda_lib.cu rose_axpy_omp.c rose_axpy_ompacc.cu rose_axpy.c timer.c -c
#	nvcc $(TEST_INCLUDES) ${TEST_LINK} -arch sm_20 -DUSE_ROSE_GOMP_OPENMP_LIBRARY ../runtime/homp.c ../runtime/homp_dev.c rose_axpy_omp.c rose_axpy_ompacc.cu rose_axpy.c -c
#	nvcc $(TEST_INCLUDES) -DDEVICE_NVGPU_SUPPORT ../runtime/homp.c ../runtime/homp_dev.c rose_axpy_omp.c rose_axpy_ompacc.cu rose_axpy.c -c
#	nvcc $(TEST_INCLUDES) ${TEST_LINK}  *.o -o $@
	gcc $(TEST_INCLUDES) -DDEBUG_MSG -g ../runtime/homp.c ../runtime/homp_dev.c rose_axpy_ompacc.c rose_axpy.c -c
	gcc $(TEST_INCLUDES) -g *.o -o $@ ${TEST_LINK}

clean:
	rm -rf *.o axpy

