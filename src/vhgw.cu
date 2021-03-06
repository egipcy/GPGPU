#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <vector>
#include <string>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>


__device__ size_t compare(size_t a, size_t b, bool is_dilatation) {

  if (is_dilatation)
    return a > b ? a : b;

  return a < b ? a : b;
}



__global__ void print_cuda(size_t* data, int height, int width) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}
	printf("%i, %i --> %lu\n", x, y, data[x+y*width]);
}


__global__ void compute_vHGW(size_t* data_read, size_t* data_write, int height, int width, size_t* g, size_t* h, size_t k, bool is_dilatation) {
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
      g_line[x] = (x % k) == 0 ? curr_line[x] : compare(g_line[x - 1], curr_line[x], is_dilatation);
	}

	h_line[m - 1] = curr_line[m - 1];
    for (size_t y = 1; y < m; y++)
    {
      size_t x = m - 1 - y;
      h_line[x] = (x + 1) % k == 0 ? curr_line[x] : compare(h_line[x + 1], v_line[x], is_dilatation);
    }

    // Compute new line 
    for (size_t x = 0; x < m; x++)
    {
    	auto div2 = k / 2;
    	if (x  < div2)
    		v_line[x] = g_line[x + div2];
    	else if (x + div2 >= m)
    		v_line[x] = x + div2 < m + psa ? compare(g_line[m - 1], h_line[x - (div2)], is_dilatation) : h_line[x - (div2)];
    	else
        	v_line[x] = compare(g_line[x + div2], h_line[x - div2], is_dilatation);
    }

}

void cuda_vHGW(size_t* data_host, int height, int width, size_t k, bool	is_dilatation) {
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

	int bsize = 1;

	// Executing kernel 

	compute_vHGW<<<height, bsize>>>(data_read, data_write, height, width, g, h, k, is_dilatation);
	cudaDeviceSynchronize();

	// Transfer data back to host memory
	cudaMemcpy(data_host, data_write, sizeof(size_t) * width * height, cudaMemcpyDeviceToHost);

	// Deallocate device memory
    cudaFree(data_read);
    cudaFree(data_write);
    cudaFree(h);
    cudaFree(g);
}
