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

static void BM_Rendering_CPU_dilate(benchmark::State& state)
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

static void BM_Rendering_CPU_erode(benchmark::State& state)
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

BENCHMARK(BM_Rendering_cpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_Rendering_gpu)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK_MAIN();
