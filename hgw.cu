#include "hgw.hh"

#define BLOCK_SIZE 256

__global__ void compute_g(size_t* g, size_t* v, size_t k, int n, size_t(*extremum)(const size_t&, const size_t&)) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < n) {
    g[tid] = (tid % k) == 0 ? v[tid] : extremum(g[tid - 1], v[tid]);
  }

}

__global__ void compute_h(size_t* h, size_t* v, size_t k, int n, size_t(*extremum)(const size_t&, const size_t&)) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  h[n - 1] = v[n - 1];
  if (tid < n) {
    size_t i = n - 1 - tid;
    h[i] = (i + 1) % k == 0 ? v[i] : extremum(h[i + 1], v[i]);
  }
}

__global__ void compute_v(size_t* v, size_t* g, size_t* h, size_t k, size_t psa, int n, size_t(*extremum)(const size_t&, const size_t&)) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < n) {
    if (2*tid < k)
      v[tid] = g[tid + k/2];
    else if (tid + k/2 >= n)
      v[tid] = tid + k/2 < n + psa ? extremum(g[n - 1], h[tid - k/2]) : h[tid - k/2];
    else
      v[tid] = std::max(g[tid + k/2], h[tid - k/2]);
  }
}

void cuda_vHGW(std::vector<std::vector<size_t*>>& matrix, size_t k, size_t(*extremum)(const size_t&, const size_t&))
{
  // http://www.cmm.mines-paristech.fr/~beucher/publi/HGWimproved.pdf - Algorithm 1

  for (auto& v: matrix)
  {
    auto m = v.size();
    auto psa = (k - (m - 1) % k) - 1;

    size_t *d_g, *d_h, *d_v; 

    std::vector<size_t> g(m);
    std::vector<size_t> h(m);

    // Allocate device memory 
    cudaMalloc((void**)&d_g, sizeof(size_t) * m);
    cudaMalloc((void**)&d_h, sizeof(size_t) * m);
    cudaMalloc((void**)&d_v, sizeof(size_t) * m);

    // Transfer data from host to device memory
    cudaMemcpy(d_g, &(g[0]), sizeof(size_t) * m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h, &(h[0]), sizeof(size_t) * m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, &(v[0]), sizeof(size_t) * m, cudaMemcpyHostToDevice);

    // Executing kernel 
    int block_size = BLOCK_SIZE;
    int grid_size = ((m + block_size) / block_size);

    compute_g<<<grid_size,block_size>>>(d_g, d_v, k, m, extremum);
    compute_h<<<grid_size,block_size>>>(d_h, d_v, k, m, extremum);
    compute_v<<<grid_size,block_size>>>(d_v, d_g, d_h, k, psa, m, extremum);

    // Transfer data back to host memory
    cudaMemcpy(&(v[0]), d_v, sizeof(size_t) * m, cudaMemcpyDeviceToHost);

    // Deallocate device memory
    cudaFree(d_g);
    cudaFree(d_h);
    cudaFree(d_v);
  }
}

void vHGW(std::vector<std::vector<size_t*>>& matrix, size_t k, size_t(*extremum)(const size_t&, const size_t&))
{
  // http://www.cmm.mines-paristech.fr/~beucher/publi/HGWimproved.pdf - Algorithm 1

  for (auto& v: matrix)
  {
    auto m = v.size();

    auto psa = (k - (m - 1) % k) - 1;

    std::vector<size_t> g(m);
    std::vector<size_t> h(m);

    for (size_t x = 0; x < m; x++)
      g[x] = (x % k) == 0 ? *(v[x]) : extremum(g[x - 1], *(v[x]));

    h[m - 1] = *(v[m - 1]);
    for (size_t y = 1; y < m; y++)
    {
      size_t x = m - 1 - y;
      h[x] = (x + 1) % k == 0 ? *(v[x]) : extremum(h[x + 1], *(v[x]));
    }

    for (size_t x = 0; x < m; x++)
    {
      if (2*x < k)
        *(v[x]) = g[x + k/2];
      else if (x + k/2 >= m)
        *(v[x]) = x + k/2 < m + psa ? extremum(g[m - 1], h[x - k/2]) : h[x - k/2];
      else
        *(v[x]) = extremum(g[x + k/2], h[x - k/2]);
    }
  }
}