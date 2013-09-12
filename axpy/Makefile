# the translator should use: https://bitbucket.org/ChunhuaLiao/rose-omp-accelerator
# just build librose.so under buildtree/src, and then build the translator roseompacc under tests/roseTests/ompLoweringTests/
# The translator source is rose-omp-accelerator/tests/roseTests/ompLoweringTests/roseompacc.C
# The command lines (two steps for now) to use roseompacc is exemplified in
# rose-omp-accelerator/tests/roseTests/ompLoweringTests/Makefile.am
ROSE_DIR=/data/yy8//ROSE/
top_srcdir=/data/yy8//ROSE/rose-omp-accelerator
top_builddir=/data/yy8//ROSE/rose-omp-accelerator-build
GOMP_PATH=/opt/local/gcc/4.4.7/lib64/
ROSE_OMPTEST=${top_builddir}/tests/roseTests/ompLoweringTests

TEST_INCLUDES = -I$(top_srcdir)/src/frontend/SageIII -I$(top_srcdir)/src/midend/programTransformation/ompLowering -I.
TEST_FLAGS=-rose:openmp:lowering -g --edg:no_warnings -DUSE_ROSE_GOMP_OPENMP_LIBRARY
TEST_LINK = -L$(top_builddir)/src/midend/.libs -lxomp $(GOMP_PATH)/libgomp.a -lpthread -lm

axpy:
	${ROSE_OMPTEST}/roseompacc ${TEST_INCLUDES} ${TEST_FLAGS} -rose:skipfinalCompileStep axpy_ompacc.c
	${ROSE_OMPTEST}/roseomp ${TEST_INCLUDES} ${TEST_FLAGS} -rose:skipfinalCompileStep axpy_omp.c
	${ROSE_OMPTEST}/roseomp ${TEST_INCLUDES} ${TEST_FLAGS} -rose:skipfinalCompileStep axpy.c
	nvcc $(TEST_INCLUDES) ${TEST_LINK}  -DUSE_ROSE_GOMP_OPENMP_LIBRARY ../xomp_cuda_lib.cu rose_axpy_omp.c rose_axpy_ompacc.cu rose_axpy.c timer.c -c
	nvcc $(TEST_INCLUDES) ${TEST_LINK}  *.o -o $@

clean:
	rm -rf *.o axpy

