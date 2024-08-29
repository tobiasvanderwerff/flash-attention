
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>


__global__ void matmul_kernel_1(float* out, const float* A, const float* B, int h, int w, int k) {
    /* Naive matmul*/

    const int c = blockDim.x * blockIdx.x + threadIdx.x;
    const int r = blockDim.y * blockIdx.y + threadIdx.y;
    
    if (r >= h || c >= w) return;

    float sum = 0.0f;
    for (int i = 0; i < k; ++i) sum += A[r*k + i] * B[i*w + c];
    out[r*w + c] = sum;
}

template <int TILE_SIZE>
__global__ void matmul_kernel_2(float* out, const float* A, const float* B, int h, int w, int k) {
    /* Matmul using tiling.
    *
    *  NB: assumption is that BLOCK_SIZE = TILE_SIZE.
    */

    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int r = blockDim.y * blockIdx.y + ty;  // row
    const int c = blockDim.x * blockIdx.x + tx;  // col

    float sum = 0.0f;
    for (int ti = 0; ti < (k+TILE_SIZE-1)/TILE_SIZE; ++ti) {
        A_tile[ty][tx] = (r < h && ti*TILE_SIZE + tx < k) ? A[r*k + ti*TILE_SIZE + tx] : 0.f;
        B_tile[ty][tx] = (ti*TILE_SIZE + ty < k && c < w) ? B[(ti*TILE_SIZE + ty)*w + c] : 0.f;
        __syncthreads();

        for (int i = 0; i < TILE_SIZE; ++i) 
            sum += A_tile[ty][i] * B_tile[i][tx];
        __syncthreads();
    }

    if (r < h && c < w)
        out[r*w + c] = sum;
}

template <int TILE_SIZE>
void launch_matmul_kernel(dim3 gdim, dim3 bdim, float* out, const float* A, const float* B, int h, int w, int k) {
    // matmul_kernel_1<<<gdim, bdim>>>(out, A, B, h, w, k);
    matmul_kernel_2<TILE_SIZE><<<gdim, bdim>>>(out, A, B, h, w, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template void launch_matmul_kernel<16>(dim3 gdim, dim3 bdim, float* out, const float* A, const float* B, int h, int w, int k); 
template void launch_matmul_kernel<32>(dim3 gdim, dim3 bdim, float* out, const float* A, const float* B, int h, int w, int k); 
