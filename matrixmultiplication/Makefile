# the translator should use: https://bitbucket.org/ChunhuaLiao/edg4x-rose-omp-accelerator
# just build librose.so under buildtree/src, and then build the translator roseompacc under tests/roseTests/ompLoweringTests/
# The translator source is tests/roseTests/ompLoweringTests/roseompacc.C
# The command lines (two steps for now) to use roseompacc is exemplified in
# rose-omp-accelerator/tests/roseTests/ompLoweringTests/Makefile.am
ROSE_DIR=/data/yy8/ROSE/
top_srcdir=/data/yy8/ROSE/edg4x-rose
top_builddir=/data/yy8/ROSE/edg4x-rose-build
GOMP_PATH=/opt/local/gcc/4.4.6/lib64/
XOMP_RUNTIME_PATH=../../runtime
ROSE_OMPTEST=${ROSE_DIR}/edg4x-rose-build/tests/roseTests/ompLoweringTests

TEST_INCLUDES = -I$(top_srcdir)/src/frontend/SageIII -I$(top_srcdir)/src/midend/programTransformation/ompLowering -I.
TEST_FLAGS=-rose:openmp:lowering -g --edg:no_warnings -DUSE_ROSE_GOMP_OPENMP_LIBRARY
TEST_LINK = -L$(top_builddir)/src/midend/.libs -lxomp $(GOMP_PATH)/libgomp.a -lpthread -lm

matmul:
	${ROSE_OMPTEST}/roseompacc ${TEST_INCLUDES} ${TEST_FLAGS} -rose:skipfinalCompileStep matmul.c
#	${ROSE_OMPTEST}/roseomp ${TEST_INCLUDES} ${TEST_FLAGS} -rose:skipfinalCompileStep matmul_omp.c
#	${ROSE_OMPTEST}/roseomp ${TEST_INCLUDES} ${TEST_FLAGS} -rose:skipfinalCompileStep matmul.c
#	nvcc $(TEST_INCLUDES) ${TEST_LINK}  -DUSE_ROSE_GOMP_OPENMP_LIBRARY ../xomp_cuda_lib.cu rose_matmul_omp.c rose_matmul_ompacc.cu rose_matmul.c timer.c -c
	nvcc $(TEST_INCLUDES) ${TEST_LINK}  rose_matmul.cu ${XOMP_RUNTIME_PATH}/xomp_cuda_lib.cu -o $@

matmul_time:
	nvcc $(TEST_INCLUDES) ${TEST_LINK}  rose_matmul_time.cu ${XOMP_RUNTIME_PATH}/xomp_cuda_lib.cu -o $@

openacc:
	pgcc -DOPENACC=1 -ta=nvidia -Minfo=accel -O3 -o matmul_acc_pgi matmul.c
#	pgcc -DOPENACC=1 -ta=nvidia,nofma,cc20 -Minfo=accel -O3 -o matmul_acc_pgi matmul.c
	hmpp gcc -fopenmp -DOPENACC=1 matmul.c -o matmul_acc_hmpp -O3

clean:
	rm -rf *.o matmul

