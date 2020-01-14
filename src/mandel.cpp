#include <cstddef>
#include <memory>
#include <stdlib.h>

#include <png.h>

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>
#include "render.hpp"


void write_png(const std::byte* buffer,
               int width,
               int height,
               int stride,
               const char* filename)
{
  png_structp png_ptr =
    png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

  if (!png_ptr)
    return;

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
  {
    png_destroy_write_struct(&png_ptr, nullptr);
    return;
  }

  FILE* fp = fopen(filename, "wb");
  png_init_io(png_ptr, fp);

  png_set_IHDR(png_ptr, info_ptr,
               width,
               height,
               8,
               PNG_COLOR_TYPE_RGB_ALPHA,
               PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);

  png_write_info(png_ptr, info_ptr);
  for (int i = 0; i < height; ++i)
  {
    png_write_row(png_ptr, reinterpret_cast<png_const_bytep>(buffer));
    buffer += stride;
  }

  png_write_end(png_ptr, info_ptr);
  png_destroy_write_struct(&png_ptr, nullptr);
  fclose(fp);
}


// Usage: ./mandel
int main(int argc, char** argv)
{
  (void) argc;
  (void) argv;

  std::string filename = "output.png";
  std::string mode = "GPU";
  int width = 10;
  int height = 10;
  int niter = 100;

  CLI::App app{"mandel"};
  app.add_option("-o", filename, "Output image");
  app.add_option("niter", niter, "number of iteration");
  app.add_option("width", width, "width of the output image");
  app.add_option("height", height, "height of the output image");
  app.add_set("-m", mode, {"GPU", "CPU"}, "Either 'GPU' or 'CPU'");

  CLI11_PARSE(app, argc, argv);

  size_t* buffer = (size_t*)calloc(height * width, height * width*sizeof(size_t));

  // Rendering
  spdlog::info("Runnging {} mode with (w={},h={}).", mode, width, height);
  if (mode == "CPU")
  {
    //render_cpu(reinterpret_cast<char*>(buffer.get()), width, height, stride, niter);
  }
  else if (mode == "GPU")
  {
    int p = 3;
    int *kernel_host = (int*)calloc(p, p*sizeof(int));
    for (int i=0; i<p; i++) {
      kernel_host[i] = 1;
    }
    bool max_comparator = 1;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
          buffer[y*width+x] = y*width+x;
      }
    }

    apply_1D(buffer, kernel_host, p, width, height, max_comparator);
    printf("Output buffer:\n");
    for (int y=0; y < height; y++) {
      for (int x=0; x < width; x++) {
        printf("%lu ", buffer[x+y*width]);
      }
      printf("\n");
    }

    free(kernel_host);
   /* 2D example
    int p = 3;
    int q = 3;
    int *kernel_host = (int*)calloc(p*q, p*q*sizeof(int));
    for (int j=0; j<q;j++)
      for (int i=0; i<p; i++) {
      kernel_host[i+j*q] = 1;
    }
    bool max_comparator = 1;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
          buffer[y*width+x] = y*width+x;
      }
    }

    apply_2D(buffer, kernel_host, p, q, width, height, max_comparator);
    printf("Output buffer:\n");
    for (int y=0; y < height; y++) {
      for (int x=0; x < width; x++) {
        printf("%lu ", buffer[x+y*width]);
      }
      printf("\n");
    }

    free(kernel_host);
    */
  }

  // Save
  //write_png(buffer.get(), width, height, stride, filename.c_str());
  spdlog::info("Output saved in {}.", filename);
}

