#include "render.hpp"
#include <vector>
#include <benchmark/benchmark.h>

static void BM_Rendering_CPU_image_load_write(benchmark::State& state)
{
  for (auto _ : state)
  {
    auto image = PGM("house.pgm");
    image.write("house.pgm");
  }
}

static void BM_Rendering_CPU_kernel(benchmark::State& state)
{
  size_t i = 0;
  for (auto _ : state)
  {
    size_t k = state.range(i++);
    
    std::vector<std::vector<bool>> kernel = std::vector<std::vector<bool>>(k);
    for (size_t j = 0; j < kernel.size(); j++)
      kernel[j] = std::vector<bool>(k, true);
  }
}

static void BM_Rendering_CPU_dilate_naive(benchmark::State& state)
{
  size_t i = 0;
  for (auto _ : state)
  {
    size_t k = state.range(i++);
    
    std::vector<std::vector<bool>> kernel = std::vector<std::vector<bool>>(k);
    for (size_t j = 0; j < kernel.size(); j++)
      kernel[j] = std::vector<bool>(k, true);

    auto image = PGM("house.pgm");
    naive_approach(image, kernel, max);
    image.write("house.dilated.naive.pgm");
  }
}

static void BM_Rendering_CPU_erode_naive(benchmark::State& state)
{
  size_t i = 0;
  for (auto _ : state)
  {
    size_t k = state.range(i++);
    
    std::vector<std::vector<bool>> kernel = std::vector<std::vector<bool>>(k);
    for (size_t j = 0; j < kernel.size(); j++)
      kernel[j] = std::vector<bool>(k, true);

    auto image = PGM("house.pgm");
    naive_approach(image, kernel, min);
    image.write("house.eroded.naive.pgm");
  }
}

static void BM_Rendering_CPU_dilate_vhgw(benchmark::State& state)
{
  size_t i = 0;
  for (auto _ : state)
  {
    size_t k = state.range(i++);

    auto image = PGM("house.pgm");
    dilate_vHGW(image, k);
    image.write("house.dilated.vHGW.pgm");
  }
}

static void BM_Rendering_CPU_erode_vhgw(benchmark::State& state)
{
  size_t i = 0;
  for (auto _ : state)
  {
    size_t k = state.range(i++);

    auto image = PGM("house.pgm");
    erode_vHGW(image, k);
    image.write("house.eroded.vHGW.pgm");
  }
}

// ------------------------------------ GPU ------------------------------------

static void BM_Rendering_GPU_dilate_naive(benchmark::State& state)
{
  size_t i = 0;
  for (auto _ : state)
  {
    size_t k = state.range(i++);
    
    auto kernel = std::vector<bool>(k * k, true);

    auto image = PGM("house.pgm");
    apply_2D(image.get_datas().data(), kernel.data(), k, k, image.get_width(), image.get_height(), true)
    image.write("house.dilated.naive.gpu.pgm");
  }
}

static void BM_Rendering_GPU_erode_naive(benchmark::State& state)
{
  size_t i = 0;
  for (auto _ : state)
  {
    size_t k = state.range(i++);
    
    auto kernel = std::vector<bool>(k * k, true);

    auto image = PGM("house.pgm");
    apply_2D(image.get_datas().data(), kernel.data(), k, k, image.get_width(), image.get_height(), false)
    image.write("house.eroded.naive.gpu.pgm");
  }
}

BM_Rendering_CPU_image_load_write
BM_Rendering_CPU_kernel
BM_Rendering_CPU_dilate_naive
BM_Rendering_CPU_erode_naive
BM_Rendering_CPU_dilate_vhgw
BM_Rendering_CPU_erode_vhgw
BM_Rendering_GPU_dilate_naive
BM_Rendering_GPU_erode_naive

BENCHMARK(BM_Rendering_CPU_image_load_write)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_Rendering_CPU_kernel)
->Unit(benchmark::kMillisecond)
->UseRealTime();
->Args(3)
->Args(6)
->Args(9)
->Args(12)
->Args(15)
->Args(18)
->Args(21)
->Args(24)

BENCHMARK(BM_Rendering_CPU_dilate_naive)
->Unit(benchmark::kMillisecond)
->UseRealTime();
->Args(3)
->Args(6)
->Args(9)
->Args(12)
->Args(15)
->Args(18)
->Args(21)
->Args(24)
 
BENCHMARK(BM_Rendering_CPU_erode_naive)
->Unit(benchmark::kMillisecond)
->UseRealTime();
->Args(3)
->Args(6)
->Args(9)
->Args(12)
->Args(15)
->Args(18)
->Args(21)
->Args(24)
  
BENCHMARK(BM_Rendering_CPU_dilate_vhgw)
->Unit(benchmark::kMillisecond)
->UseRealTime();
->Args(3)
->Args(6)
->Args(9)
->Args(12)
->Args(15)
->Args(18)
->Args(21)
->Args(24)
  
BENCHMARK(BM_Rendering_CPU_erode_vhgw)
->Unit(benchmark::kMillisecond)
->UseRealTime();
->Args(3)
->Args(6)
->Args(9)
->Args(12)
->Args(15)
->Args(18)
->Args(21)
->Args(24)
  
BENCHMARK(BM_Rendering_GPU_dilate_naive)
->Unit(benchmark::kMillisecond)
->UseRealTime();
->Args(3)
->Args(6)
->Args(9)
->Args(12)
->Args(15)
->Args(18)
->Args(21)
->Args(24)

BENCHMARK(BM_Rendering_GPU_erode_naive)
->Unit(benchmark::kMillisecond)
->UseRealTime();
->Args(3)
->Args(6)
->Args(9)
->Args(12)
->Args(15)
->Args(18)
->Args(21)
->Args(24)
  
  
BENCHMARK_MAIN();

