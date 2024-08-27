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

#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define TILE_SIZE 16  // TODO: convert to consexpr

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// __global__ void attention_1() {
//     /* Naive attention implementation (no flash attention) */
// }

__global__ void matmul_kernel_1(float* out, const float* A, const float* B, int h, int w, int k) {
    /* Naive matmul*/

    const int c = blockDim.x * blockIdx.x + threadIdx.x;
    const int r = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (r >= h || c >= w) return;

    float sum = 0.0f;
    for (int i = 0; i < k; ++i) sum += A[r*k + i] * B[i*w + c];
    out[r*w + c] = sum;
}

__global__ void matmul_kernel_2(float* out, const float* A, const float* B, int h, int w, int k) {
    /* Matmul using tiling */

    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int r = blockDim.y * by + ty;  // row
    const int c = blockDim.x * bx + tx;  // col
    const int idx = r * w + c;

    // TODO
}

/* 
====================================  C++ ==================================== 
*/

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

torch::Tensor matmul(const torch::Tensor& A, const torch::Tensor& B) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    int h = A.size(0);
    int w = B.size(1);
    int k = A.size(1);
    TORCH_CHECK(k==B.size(0), "Size mismatch!");
    auto out = torch::zeros({h, w}, A.options());

    dim3 bdim(16, 16);
    dim3 gdim(cdiv(w, bdim.x), cdiv(h, bdim.y));
    matmul_kernel_1<<<gdim, bdim>>>(
        out.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), h, w, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
