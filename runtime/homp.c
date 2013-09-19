/*
 * homp.c
 *
 *  Created on: Sep 16, 2013
 *      Author: yy8
 */


#include "omp4.0.h"
#include "homp.h"
omp_device_t * omp_devices;
int omp_num_device;

void omp_init_devices() {
	omp_num_device = omp_get_num_devices();
	omp_devices = malloc(sizeof(omp_device_t) * omp_num_device);
	int i;
	for (i=0; i<omp_num_device; i++)
	{
		omp_devices[i].id = i;
		omp_devices[i].type = OMP_DEVICE_NVGPU;
		omp_devices[i].status = 1;
		omp_devices[i].next = &omp_devices[i+1];
	}
	if (omp_num_device) {
		default_device_var = 0;
		omp_devices[omp_num_device-1].next = NULL;
	}
}


int linearize2D(int X, int Y, int i, int j) {
	return i*Y+j;
}

void cartize2D(int X, int Y, int num, int *i, int *j) {
	*i = num/Y;
	*j = num%Y;
}

int linearize3D(int X, int Y, int Z, int i, int j, int k) {
	return 0;
}

void cartize3D(int X, int Y, int Z, int num, int *i, int *j, int *k) {
}

void map3D(int X, int Y, int Z, int startX, int subX, int startY, int subY, int startZ, int subZ, int i, int j, int k, int *subi, int *subj, int *subk) {
}

void rev_map3D(int X, int Y, int Z, int startX, int subX, int startY, int subY, int startZ, int subZ, int *i, int *j, int *k, int subi, int subj, int subk) {
}

void * marshal2DArrayRegion(void * start_address, int ele_size, int X, int Y, int startX, int lengthX, int startY, int lenghtY) {
	return NULL;

}

void * marshalArrayRegion(void * start_address, int ele_size, int X, int Y, int Z, int startX, int lenghtX, int startY, int lengthY, int startZ, int lenghtZ) {
	return NULL;
}



