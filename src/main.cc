#include <iostream>
#include <limits>

#include "PGM.hh"
#include "algo.cc"

int main()
{
  size_t k = 20;

  auto image = PGM("house.pgm");
  dilate_vHGW(image, k);
  image.write("house.dilated.vHGW.pgm");

  image = PGM("house.pgm");
  erode_vHGW(image, k);
  image.write("house.eroded.vHGW.pgm");

  std::vector<std::vector<bool>> kernel = std::vector<std::vector<bool>>(k);
  for (size_t i = 0; i < kernel.size(); i++)
    kernel[i] = std::vector<bool>(k, true);

  image = PGM("house.pgm");
  naive_approach(image, kernel, max);
  image.write("house.dilated.naive.pgm");

  image = PGM("house.pgm");
  naive_approach(image, kernel, min);
  image.write("house.eroded.naive.pgm");

  return 0;
}
