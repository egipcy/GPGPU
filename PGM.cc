#include "PGM.hh"

#include <fstream>
#include <cassert>
#include <sstream>

PGM::PGM(const std::vector<std::vector<size_t>>& matrix)
  : matrix_(matrix)
{ }

static std::vector<size_t> extract_ints(const std::string& str)
{
  std::vector<size_t> ret;

  std::stringstream ss;
  ss << str;
  while (!ss.eof())
  {
    size_t integer;
    std::string temp;
    ss >> temp;
    if (std::stringstream(temp) >> integer)
      ret.push_back(integer);
  }

  return ret;
}

static std::vector<size_t> extract_chars(const std::string& str, bool onebyte)
{
  std::vector<size_t> ret;

  size_t step = onebyte ? 1 : 2;
  for (size_t i = 0; i < str.size(); i+=step)
  {
    size_t integer = (unsigned char)str[i];
    if (!onebyte)
      integer = integer * 256 + str[i + 1];
    ret.push_back(integer);
  }

  return ret;
}

PGM::PGM(const std::string& filename)
{
  std::ifstream file(filename, std::ios::binary);

  std::string line;

  std::getline(file, line);
  assert(line == "P2" || line == "P5");
  bool binary = line == "P5";

  std::getline(file, line);
  auto v = extract_ints(line);
  size_t width = v[0];
  size_t height = v[1];

  std::getline(file, line);
  v = extract_ints(line);
  size_t maximum = v[0];

  if (!binary)
    while (std::getline(file, line))
      matrix_.push_back(extract_ints(line));
  else
  {
    std::getline(file, line);
    for (size_t i = 0; i < height; i++)
      matrix_.push_back(extract_chars(line.substr(i * width, width), maximum < 256));
  }

  file.close();
}

void PGM::write(const std::string& filename) const
{
  std::ofstream file(filename);

  file << "P2" << std::endl;
  file << matrix_.size() << " " << matrix_[0].size() << std::endl;

  int maximum = 0;
  for (size_t i = 0; i < matrix_.size(); i++)
    for (size_t j = 0; j < matrix_[i].size(); j++)
      if (matrix_[i][j] > maximum)
        maximum = matrix_[i][j];
  file << maximum << std::endl;

  for (size_t i = 0; i < matrix_.size(); i++)
  {
    for (size_t j = 0; j < matrix_[i].size() - 1; j++)
      file << matrix_[i][j] << " ";
    file << matrix_[i][matrix_[i].size() - 1] << std::endl;
  }

  file.close();
}

std::vector<std::vector<size_t>>& PGM::get_matrix()
{
  return matrix_;
}
