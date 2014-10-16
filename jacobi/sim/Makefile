TEST_INCLUDES = -I../../runtime -I.
TEST_LINK = -lpthread -lm

jacobi:	
#	gcc $(TEST_INCLUDES) -DDEBUG_MSG -g ../../runtime/homp.c ../../runtime/homp_dev.c jacobi_ompacc_mdev.c -c
	gcc $(TEST_INCLUDES) -g -O0 ../../runtime/homp.c ../../runtime/homp_dev.c jacobi_ompacc_mdev.c -c
	gcc $(TEST_INCLUDES) -g *.o -o $@ ${TEST_LINK}

clean:
	rm -rf *.o jacobi

