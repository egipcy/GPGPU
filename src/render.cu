#include "render.hpp"
#include <spdlog/spdlog.h>
#include <cassert>
#include <stdlib.h>

[[gnu::noinline]]
void _abortError(const char* msg, const char* fname, int line)
{
  cudaError_t err = cudaGetLastError();
  spdlog::error("{} ({}, line: {})", msg, fname, line);
  spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
  std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

__global__ void applyDilEro(size_t* buffer_read, size_t* buffer_write, int p, int* kernel, int width, int height, bool max_comparator) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;


  if (x >= width || y >= height)
    return;

  int x_left = x - p/2;
  int x_right = x + p/2;
  int p_start = 0;
  int p_end = p;

  if (x_left < 0) {
    p_start -= x_left;
  }
  if (x_right >= width) {
    p_end += width - x_right - 1;
  }

  // Selecting the first value where kernel is 1.
  // We can't take just the first value and start comparing, as this value can be skipped if kernel value is 0 for this pixel.
  int first_one_index = p_start;
  //We assume that at least one element in kernel is equal to 1.
  while (kernel[first_one_index] != 1 && first_one_index < p_end) {
    first_one_index++;
  }
  
  size_t* line = buffer_read+y*width;
  size_t best = line[x_left+first_one_index];
  if (max_comparator) {
    for (int i = p_start; i <p_end; i++) {
      if (kernel[i] && line[x_left+i] > best)
        best = line[x_left+i];
    }
  } else {
    for (int i = p_start; i <p_end; i++) {
      if (kernel[i] && line[x_left+i] < best)
        best = line[x_left+i];
    }
  }
  buffer_write[x+y*width] = best;
}

__global__ void applyDilEro2D(size_t* buffer_read, size_t* buffer_write, int p, int q, int* kernel, int width, int height, bool max_comparator) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;


  if (x >= width || y >= height)
    return;

  int x_left = x - p/2;
  int x_right = x + p/2;
  int y_top = y - q/2;
  int y_bottom = y + q/2;
  int p_start = 0;
  int p_end = p;
  int q_start = 0;
  int q_end = q;

  if (x_left < 0) {
    p_start -= x_left;
  }
  if (x_right >= width) {
    p_end += width - x_right - 1;
  }
  if (y_top < 0) {
    q_start -= y_top;
  }
  if (y_bottom >= height) {
    q_end += height - y_bottom - 1;
  }

  size_t best = buffer_read[x_left+p_start+(y_top+q_start)*width];

  if (max_comparator) {
    for (int yy = q_start; yy < q_end; yy++) {
      for (int xx = p_start; xx < p_end; xx++) {
        if (kernel[xx + yy*q] && buffer_read[x_left + xx + (y_top+yy)*width] > best)
          best = buffer_read[x_left + xx + (y_top+yy)*width];
      }
    }
  } else {
    for (int yy = q_start; yy < q_end; yy++) {
      for (int xx = p_start; xx < p_end; xx++) {
        if (kernel[xx + yy*q] && buffer_read[x_left + xx + (y_top+yy)*width] < best)
          best = buffer_read[x_left + xx + (y_top+yy)*width];
      }
    }
  }
  buffer_write[x+y*width] = best;
}


void apply_1D(size_t* hostBuffer, int *kernel_host, int p, int width, int height, bool max_comparator)
{
  /**
  * Applies dilatation (max_comparator == True) or erosion (max_comparator == False) over hostBuffer.
  * Then copies the dilated (erroded) image into hostBuffer.
  *
  * @param hostBuffer: the input image
  * @param kernel_host: p size array
  * @param p: size of array
  * @param max_comparator: if true(false) the functions replace the current pixel by the biggest(smallest) pixel 
      in p/2 range.
  */
  cudaError_t rc = cudaSuccess;

  // Allocate device memory
  size_t*  buffer_read;
  size_t*  buffer_write;
  
  //Copy kernel
  int* kernel;
  rc = cudaMalloc((void**)&kernel,p*sizeof(int));
  rc = cudaMalloc(&buffer_read, width*sizeof(size_t)*height);
  rc = cudaMalloc(&buffer_write, width*sizeof(size_t)*height);
  if (rc)
    abortError("Fail buffer allocation");
  
  rc = cudaMemcpy((void*)kernel, (void*)kernel_host, p*sizeof(int), cudaMemcpyHostToDevice);
  if (rc)
    abortError("Unable to copy Kernel");
  rc = cudaMemcpy(buffer_read, hostBuffer, width*sizeof(size_t)*height, cudaMemcpyHostToDevice);
  if (rc)
    abortError("Unable to copy image to buffer");
    

  int bsize = 32;
  int w     = std::ceil((float)width / bsize);
  int h     = std::ceil((float)height / bsize);

  spdlog::debug("running kernel of size ({},{})", w, h);

  dim3 dimBlock(bsize, bsize);
  dim3 dimGrid(w, h);

  applyDilEro<<<dimGrid, dimBlock>>>(buffer_read, buffer_write, p, kernel, width, height, max_comparator);

  if (cudaPeekAtLastError())
    abortError("Computation Error");

  // Copy back to main memory
  rc = cudaMemcpy(hostBuffer, buffer_write, width*sizeof(size_t)*height, cudaMemcpyDeviceToHost);
  if (rc)
    abortError("Unable to copy buffer back to memory");

  // Free
  rc = cudaFree(buffer_read);
  rc = cudaFree(buffer_write);
  rc = cudaFree(kernel);
  if (rc)
    abortError("Unable to free memory");
}

void apply_2D(size_t* hostBuffer, int *kernel_host, int p, int q, int width, int height, bool max_comparator)
{
  /**
  * Applies dilatation (max_comparator == True) or erosion (max_comparator == False) over hostBuffer.
  * Then copies the dilated (erroded) image into hostBuffer.
  *
  * @param hostBuffer: the input image
  * @param kernel_host: pq size array
  * @param p, q: size of kernel
  * @param max_comparator: if true(false) the functions replace the current pixel by the biggest(smallest) pixel 
      in p/2 range.
  */
  cudaError_t rc = cudaSuccess;

  // Allocate device memory
  size_t*  buffer_read;
  size_t*  buffer_write;
  
  //Copy kernel
  int* kernel;
  rc = cudaMalloc((void**)&kernel,p*q*sizeof(int));
  rc = cudaMalloc(&buffer_read, width*sizeof(size_t)*height);
  rc = cudaMalloc(&buffer_write, width*sizeof(size_t)*height);
  if (rc)
    abortError("Fail buffer allocation");
  
  rc = cudaMemcpy((void*)kernel, (void*)kernel_host, p*q*sizeof(int), cudaMemcpyHostToDevice);
  if (rc)
    abortError("Unable to copy Kernel");
  rc = cudaMemcpy(buffer_read, hostBuffer, width*sizeof(size_t)*height, cudaMemcpyHostToDevice);
  if (rc)
    abortError("Unable to copy image to buffer");
    

  int bsize = 32;
  int w     = std::ceil((float)width / bsize);
  int h     = std::ceil((float)height / bsize);

  spdlog::debug("running kernel of size ({},{})", w, h);

  dim3 dimBlock(bsize, bsize);
  dim3 dimGrid(w, h);

  applyDilEro2D<<<dimGrid, dimBlock>>>(buffer_read, buffer_write, p, q, kernel, width, height, max_comparator);

  if (cudaPeekAtLastError())
    abortError("Computation Error");

  // Copy back to main memory
  rc = cudaMemcpy(hostBuffer, buffer_write, width*sizeof(size_t)*height, cudaMemcpyDeviceToHost);
  if (rc)
    abortError("Unable to copy buffer back to memory");

  // Free
  rc = cudaFree(buffer_read);
  rc = cudaFree(buffer_write);
  rc = cudaFree(kernel);
  if (rc)
    abortError("Unable to free memory");
}