#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

void cuda_vHGW(std::vector<std::vector<size_t*>>& matrix, size_t k, size_t(*extremum)(const size_t&, const size_t&));

void vHGW(std::vector<std::vector<size_t*>>& matrix, size_t k, size_t(*extremum)(const size_t&, const size_t&));

