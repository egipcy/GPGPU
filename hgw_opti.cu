#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <vector>
#include <string>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define max(a,b) a>b?a:b


__global__ void print_cuda(size_t* data, int height, int width) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}
	printf("%i, %i --> %lu\n", x, y, data[x+y*width]);
}


__global__ void compute_vHGW(size_t* data_read, size_t* data_write, int height, int width, size_t* g, size_t* h, size_t k) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	auto m = width;
	auto psa = (k - (m - 1) % k) - 1;

	if (index >= width)
		return;

	size_t* curr_line = data_read+index*width;
	size_t* g_line = g+index*width;
	size_t* h_line = h+index*width;
	size_t* v_line = data_write+index*width;


	// Compute G
	for (int x = 0; x < m; x++) {
      g_line[x] = (x % k) == 0 ? curr_line[x] : max(g_line[x - 1], curr_line[x]);
	}


  	// Compute H
  	h_line[m - 1] = curr_line[m - 1];
    for (int y = 1; y < m; y++) {
      size_t x = m - 1 - y;
      h_line[x] = (x + 1) % k == 0 ? curr_line[x] : max(h_line[x + 1], curr_line[x]);
    }


    // Compute new line 
    for (size_t x = 0; x < m; x++)
    {
      if (2*x < k)
        v_line[x] = g_line[x + k/2];
      else if (x + k/2 >= m)
        v_line[x] = x + k/2 < m + psa ? max(g_line[m - 1], h_line[x - k/2]) : h_line[x - k/2];
      else
        v_line[x] = max(g_line[x + k/2], h_line[x - k/2]);
    }

}

void cuda_vHGW(size_t* data_host, int height, int width, size_t k) {
	size_t* data_read;
	size_t* data_write;
	size_t* h;
	size_t* g;

	// Allocate device memory 
	cudaMalloc(&data_read, sizeof(size_t) * height * width);
	cudaMalloc(&data_write, sizeof(size_t) * height * width);
	cudaMalloc(&g, sizeof(size_t) * height * width);
	cudaMalloc(&h, sizeof(size_t) * height * width);

	// Transfer data from host to device memory
	cudaMemcpy(data_read, data_host, sizeof(size_t) * width * height, cudaMemcpyHostToDevice);

	//int bsize = 1;
	//int ww = std::ceil((float)width / bsize);
	//int hh = std::ceil((float)height / bsize);

	// Executing kernel 
	//dim3 dimBlock(bsize, bsize);
	//dim3 dimGrid(w, h);

	printf("BEFORE\n");
	compute_vHGW<<<height, 1>>>(data_read, data_write, height, width, g, h, k);
	cudaDeviceSynchronize();
	printf("AFTER\n");

	// Transfer data back to host memory
	cudaMemcpy(data_host, data_write, sizeof(size_t) * width * height, cudaMemcpyDeviceToHost);

	// Deallocate device memory
    cudaFree(data_read);
    cudaFree(data_write);
    cudaFree(h);
    cudaFree(g);
}


int main() {
	size_t* data;

	int height = 10;
	int width  = 10;

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

	size_t k = 3;

	cuda_vHGW(data, height, width, k);
	return 0;

}