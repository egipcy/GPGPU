#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <vector>
#include <string>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void print_cuda(size_t* data, int height, int width) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	printf("%i, %i\n", x, y);
	if (x >= width || y >= height) {
		return;
	}


}

void cuda_vHGW_opti(size_t* data_host, int height, int width, int p) {
	size_t* data_read;
	size_t* data_write;

	cudaMalloc(&data_read, sizeof(size_t) * height * width);
	cudaMalloc(&data_write, sizeof(size_t) * height * width);
	cudaMemcpy(data_read, data_host, sizeof(size_t) * width * height, cudaMemcpyHostToDevice);

	int bsize = 32;
	int w = std::ceil((float)width / bsize);
	int h = std::ceil((float)height / bsize);

	dim3 dimBlock(bsize, bsize);
	dim3 dimGrid(w, h);

	printf("BEFORE\n");
	print_cuda<<<dimGrid, dimBlock>>>(data_read, height, width);
	cudaDeviceSynchronize();
	printf("AFTER\n");
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