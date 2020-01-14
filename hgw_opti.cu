#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <vector>
#include <string>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>


void cuda_vHGW_opti(size_t* data_host, int height, int width, int p) {
	size_t* data_read;
	size_t* data_write;

	cudaMalloc(&data_reqd, sizeof(size_t) * height * width);
	cudaMalloc(&data_write, sizeof(size_t) * height * width);
	cudaMemcpy(data_reqd, data_host, sizeof(size_t) * width * height, cudaMemcpyHostToDevice);
}


int main() {
	size_t* data;

	int height = 10;
	int width  = 10;
	int p = 3;

	data = (size_t*)malloc(sizeof(size_t) * height*width);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			data[j + i * width] = (i * width) +j;
		}
	}


	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			printf("%lu ", data[j+i*width]);
		}
		printf("\n");
	}

	cuda_vHGW_opti(data, height, width, p);
	return 0;

}