matmul:	
	gcc $(TEST_INCLUDES) -DDEBUG_MSG -g ../../runtime/homp.c ../../runtime/homp_dev.c rose_matmul_mdev.c -c
	gcc $(TEST_INCLUDES) -g *.o -o $@ ${TEST_LINK}

clean:
	rm -rf *.o matmul

