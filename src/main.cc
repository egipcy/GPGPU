#include <iostream>
#include <limits>

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

void dilate_vHGW(PGM& image, size_t k)
{
  vHGW(image.get_matrix(), k, max);
  vHGW(image.get_transpose_matrix(), k, max);
}

void erode_vHGW(PGM& image, size_t k)
{
  vHGW(image.get_matrix(), k, min);
  vHGW(image.get_transpose_matrix(), k, min);
}

void naive_approach(PGM& image, const std::vector<std::vector<bool>>& kernel, size_t(*extremum)(const size_t&, const size_t&))
{
  auto datas = image.get_datas();

  size_t cst_extrem = 255 - extremum(255, 0);

  for (size_t i = 0; i < image.get_width(); i++)
    for (size_t j = 0; j < image.get_height(); j++)
    {
      size_t extrem = cst_extrem;

      for (size_t k_i = 0; k_i < kernel.size(); k_i++)
      {
        long x_i = i + k_i - kernel.size() / 2;
        if (x_i < 0 || x_i >= image.get_width())
          break;

        for (size_t k_j = 0; k_j < kernel[k_i].size(); k_j++)
        {
          if (!kernel[k_i][k_j])
            break;

          long x_j = j + k_j - kernel[k_i].size() / 2;
          if (x_j < 0 || x_j >= image.get_height())
            break;

          extrem = extremum(image.get_datas()[x_i * image.get_height() + x_j], extrem);
        }
      }

      datas[i * image.get_height() + j] = extrem;
    }

  image.get_datas() = datas;
}

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
