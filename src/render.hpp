#pragma once
#include <cstddef>
#include <memory>




/// \param buffer The RGBA24 image buffer
/// \param width Image width
/// \param height Image height
/// \param stride Number of bytes between two lines
/// \param n_iterations Number of iterations maximal to decide if a point
///                     belongs to the mandelbrot set.
extern "C"
void render_cpu(char* buffer, int width, int height, std::ptrdiff_t stride, int n_iterations = 100);



void apply_1D(size_t* hostBuffer, int *kernel_host, int p, int width, int height, bool max_comparator);
void apply_2D(size_t* hostBuffer, int *kernel_host, int p, int q, int width, int height, bool max_comparator);
