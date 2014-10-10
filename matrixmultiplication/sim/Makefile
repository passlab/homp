GOMP_PATH=/usr/lib/gcc/x86_64-linux-gnu/4.4

TEST_INCLUDES = -I../../runtime -I.
TEST_LINK = -lpthread -lm

matmul:	
	gcc-4.4 $(TEST_INCLUDES) -DDEBUG_MSG -g ../../runtime/homp.c ../../runtime/homp_dev.c rose_matmul_mdev.c -c
	gcc-4.4 $(TEST_INCLUDES) -g *.o -o $@ ${TEST_LINK}

clean:
	rm -rf *.o matmul

