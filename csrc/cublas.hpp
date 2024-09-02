#pragma once

#include <cublas_v2.h>

#define CUBLAS_ERR(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CUBLAS_STATUS_SUCCESS) 
   {
      fprintf(stderr, "CUBLASassert: %d %s %d\n", code, file, line);
      if (abort) exit(code);
   }
}