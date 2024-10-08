#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_CUDA_ERR(ans) { check((ans), __FILE__, __LINE__); }
inline void check(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                << std::endl;
      std::cerr << cudaGetErrorString(code) << std::endl;
      if (abort) std::exit(EXIT_FAILURE);
   }
}

bool is_power_of_two(int n) {
    if (n <= 0) return false;
    return (n & (n - 1)) == 0;
}

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }
