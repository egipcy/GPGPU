#include <iostream>

#include "PGM.hh"
#include "PGM.cc"

size_t min(const size_t& a, const size_t& b)
{
  return a < b ? a : b;
}

size_t max(const size_t& a, const size_t& b)
{
  return a > b ? a : b;
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
