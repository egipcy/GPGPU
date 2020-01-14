#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void compute_g(size_t* g, size_t* v, size_t k, int n) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < n) {
    g[tid] = (tid % k) == 0 ? v[x] : std::max(g[tid - 1], v[tid]);
  }

}

__global__ void compute_h(float* h, size_t* v, size_t k, int n) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  h[n - 1] = v[n - 1];
  if (tid < n) {
    size_t i = n - 1 - tid;
    h[i] = (i + 1) % k == 0 ? v[i] : std::max(h[i + 1], v[i]);
  }
}

__global__ void compute_v(size_t* v, size_t* g, size_t* h, size_t k, auto psa, int n) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < n) {
    if (2*tid < k)
      v[tid] = g[tid + k/2];
    else if (tid + k/2 >= n)
      v[tid] = tid + k/2 < n + psa ? std::max(g[n - 1], h[tid - k/2]) : h[tid - k/2];
    else
      v[tid] = std::max(g[tid + k/2], h[tid - k/2]);
  }
}

std::vector<std::vector<size_t>> vHGW(std::vector<std::vector<size_t>> matrix, size_t k)
{
  // http://www.cmm.mines-paristech.fr/~beucher/publi/HGWimproved.pdf - Algorithm 1

  for (auto& v: matrix)
  {
    auto m = v.size();

    auto psa = (k - (m - 1) % k) - 1;

    std::vector<size_t> g(m);
    std::vector<size_t> h(m);

    // Allocate device memory 
    cudaMalloc((void**)&d_g, sizeof(size_t) * m);
    cudaMalloc((void**)&d_h, sizeof(size_t) * m);
    cudaMalloc((void**)&d_v, sizeof(size_t) * m);

    // Transfer data from host to device memory
    cudaMemcpy(d_g, g, sizeof(size_t) * m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, h, sizeof(size_t) * m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, sizeof(size_t) * m, cudaMemcpyHostToDevice);

    // Executing kernel 
    int block_size = BLOCK_SIZE;
    int grid_size = ((m + block_size) / block_size);

    compute_g<<<grid_size,block_size>>>(d_g, d_v, k, m);
    compute_h<<<grid_size,block_size>>>(d_h, d_v, k, m);
    compute_v<<<grid_size,block_size>>>(d_v, d_g, d_h, k, psa, m);

    // Transfer data back to host memory
    cudaMemcpy(v, d_v, sizeof(size_t) * m, cudaMemcpyDeviceToHost);

    // Deallocate device memory
    cudaFree(d_g);
    cudaFree(d_h);
    cudaFree(d_v);
  }

  return matrix;
}