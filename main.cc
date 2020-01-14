#include <iostream>

#include "PGM.hh"
#include "PGM.cc"
#include "hgw.hh"

size_t min(const size_t& a, const size_t& b)
{
  return a < b ? a : b;
}

size_t max(const size_t& a, const size_t& b)
{
  return a > b ? a : b;
}

void dilate(PGM& image, size_t k)
{
  vHGW(image.get_matrix(), k, max);
  vHGW(image.get_transpose_matrix(), k, max);
}

void erode(PGM& image, size_t k)
{
  vHGW(image.get_matrix(), k, min);
  vHGW(image.get_transpose_matrix(), k, min);
}

void cuda_dilate(PGM& image, size_t k)
{
  cuda_vHGW(image.get_matrix(), k, max);
  cuda_vHGW(image.get_transpose_matrix(), k, max);
}

void cuda_erode(PGM& image, size_t k)
{
  cuda_vHGW(image.get_matrix(), k, min);
  cuda_vHGW(image.get_transpose_matrix(), k, min);
}

int main()
{
  auto image = PGM("house.pgm");
  dilate(image, 20);
  image.write("house.dilated.pgm");

  image = PGM("house.pgm");
  erode(image, 20);
  image.write("house.eroded.pgm");

  return 0;
}
