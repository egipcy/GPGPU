#include "PGM.hh"

#include <fstream>
#include <cassert>
#include <sstream>

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
  width_ = v[0];
  height_ = v[1];

  std::getline(file, line);
  v = extract_ints(line);
  size_t maximum = v[0];

  if (!binary)
  {
    datas_ = std::vector<size_t>(width_ * height_);
    size_t k = 0;
    while (std::getline(file, line))
    {
      auto l = extract_ints(line);
      for (size_t i = 0; i < l.size(); i++)
        datas_[k++] = l[i];
    }
  }
  else
  {
    std::getline(file, line);
    datas_ = extract_chars(line, maximum < 256);
  }

  matrix_ = std::vector<std::vector<size_t*>>(width_);
  for (size_t i = 0; i < width_; i++)
  {
    matrix_[i] = std::vector<size_t*>(height_);
    for (size_t j = 0; j < height_; j++)
      matrix_[i][j] = &datas_[i * height_ + j];
  }
  transpose_matrix_ = std::vector<std::vector<size_t*>>(height_);
  for (size_t j = 0; j < height_; j++)
  {
    transpose_matrix_[j] = std::vector<size_t*>(width_);
    for (size_t i = 0; i < width_; i++)
      transpose_matrix_[j][i] = &datas_[i * height_ + j];
  }

  file.close();
}

void PGM::write(const std::string& filename) const
{
  std::ofstream file(filename);

  file << "P2" << std::endl;
  file << width_ << " " << height_ << std::endl;

  int maximum = 0;
  for (size_t i = 0; i < datas_.size(); i++)
    if (datas_[i] > maximum)
      maximum = datas_[i];
  file << maximum << std::endl;

  for (size_t i = 0; i < width_; i++)
  {
    for (size_t j = 0; j < height_ - 1; j++)
      file << datas_[i * height_ + j] << " ";
    file << datas_[(i + 1) * height_] << std::endl;
  }

  file.close();
}

std::vector<std::vector<size_t*>>& PGM::get_matrix()
{
  return matrix_;
}

std::vector<std::vector<size_t*>>& PGM::get_transpose_matrix()
{
  return transpose_matrix_;
}

