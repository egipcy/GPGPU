#pragma once

#include <vector>
#include <string>

class PGM
{
public:
  PGM(const std::string& filename);

  void write(const std::string& filename) const;

  std::vector<std::vector<size_t*>>& get_matrix();
  std::vector<std::vector<size_t*>>& get_transpose_matrix();

private:
  size_t width_;
  size_t height_;
  std::vector<size_t> datas_;
  std::vector<std::vector<size_t*>> matrix_;
  std::vector<std::vector<size_t*>> transpose_matrix_;
};
