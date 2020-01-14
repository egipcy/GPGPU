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

	cudaMalloc(&data, sizeof(size_t*) * height, cudaMemcpyHostToDevice);

	for (int i = 0; i < height; i++) {
		cudaMalloc(&data[i], sizeof(size_t) * width, cudaMemcpyHostToDevice);
		cudaMemcpy(data[i], data_host[i], sizeof(size_t) * width, cudaMemcpyHostToDevice);
	}




}


void main() {
	size_t** data;

	int height = 10;
	int widht  = 10;
	int p = 3;

	data = (size_t**)malloc(sizeof(size_t*) * height);

	for (int i = 0; i < height; i++) {
		data[i] = (size_t*)malloc(sizeof(size_t) * widht)
		for (int j = 0; j < widht; j++) {
			data[i][j] = (i * width) +j;
		}
	}


	for (int i = 0; i < height; i++) {
		for (int j = 0; j < widht; j++) {
			printf("%lu ", data[i][j]);
		}
		print("\n");
	}

	cuda_vHGW_opti(data, height, widht, p);

}