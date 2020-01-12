#pragma once

#include <vector>
#include <string>

class PGM
{
public:
  PGM(const std::vector<std::vector<size_t>>& matrix);
  PGM(const std::string& filename);

  void write(const std::string& filename) const;

  std::vector<std::vector<size_t>>& get_matrix();

private:
  std::vector<std::vector<size_t>> matrix_;
};
