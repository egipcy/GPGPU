#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <vector>
#include <string>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>


void cuda_vHGW_opti(size_t** data_host, int height, int width, int p) {
	size_t** data;

	cudaMalloc(&data, sizeof(size_t*) * height);

	for (int i = 0; i < height; i++) {
		cudaMalloc(&data[i], sizeof(size_t) * width);
		cudaMemcpy(data[i], data_host[i], sizeof(size_t) * width, cudaMemcpyHostToDevice);
	}




}


int main() {
	size_t** data;

	int height = 10;
	int width  = 10;
	int p = 3;

	data = (size_t**)malloc(sizeof(size_t*) * height);

	for (int i = 0; i < height; i++) {
		data[i] = (size_t*)malloc(sizeof(size_t) * width);
		for (int j = 0; j < width; j++) {
			data[i][j] = (i * width) +j;
		}
	}


	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			printf("%lu ", data[i][j]);
		}
		printf("\n");
	}

	cuda_vHGW_opti(data, height, width, p);
	return 0;

}