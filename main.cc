#include <iostream>

#include "PGM.hh"
#include "PGM.cc"

int main()
{
  auto image = PGM("house.256.pgm");
  image.write("house.pgm");

  return 0;
}
