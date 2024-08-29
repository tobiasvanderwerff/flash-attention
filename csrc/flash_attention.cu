/*

   Q = (N,d)
   K = (N,d)
   V = (N,d)

   O = softmax(Q * K^T) * V

   or, how it's written in the Flash Attention paper:

   S = Q * K^T
   P = softmax(S)
   O = PV

*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>


__global__ void flash_attention(const float* Q, const float* K, const float* V, int seq_len, int d) {
    // TODO    
}