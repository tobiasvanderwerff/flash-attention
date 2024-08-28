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
    const int r = blockDim.y * blockIdx.x + ty;  // row
    const int c = blockDim.x * blockIdx.y + tx;  // col

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

__device__ inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

template <int BLOCK_SIZE>
__global__ void softmax_kernel(float* out, const float* inp, int h, int w) {
    /* Softmax applied row-wise. */

    // V1: assume a single block handles one row

    __shared__ float shm_sum[BLOCK_SIZE];  // TODO: is this optimal shmem size? 
    // TODO: could also use a single shm (shm_sum) and replace shm with write to global memmory.

    int tx = threadIdx.x;
    int bx = blockIdx.x;

    // Thread coarsening
    float sum = 0;
    for (int bi = 0; bi < cdiv(w, BLOCK_SIZE); ++bi) {
        int idx = bx*w + bi*BLOCK_SIZE + tx;
        if (bi*BLOCK_SIZE + tx < w) {
            float e = expf(inp[idx]);  // TODO: use __expf?
            out[idx] = e; // TODO: consider: do not save but recalculate at L. 116 (test perf. to verify if it makes sense)
            shm_sum[tx] = e;
        } else {
            shm_sum[tx] = 0;
        }

        // TODO: use padding to allow for non-powers of 2 block sizes? 
        __syncthreads();
        for (int stride = BLOCK_SIZE >> 1; stride >= 1; stride >>= 1) {
            if (tx < stride)
                shm_sum[tx] += shm_sum[tx + stride];
            __syncthreads();
        }

        // Let all threads save the intermediate sum
        sum += shm_sum[0];
        __syncthreads();
    }

    for (int bi = 0; bi < cdiv(w, BLOCK_SIZE); ++bi) {
        if (bi*BLOCK_SIZE + tx < w) 
            out[bx*w + bi*BLOCK_SIZE + tx] /= sum;
    }
}

template <int BLOCK_SIZE>
void launch_softmax_kernel(int gdim, int bdim, float* out, const float* inp, int h, int w) { 
    softmax_kernel<BLOCK_SIZE><<<gdim, bdim>>>(out, inp, h, w);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template void launch_softmax_kernel<256>(int gdim, int bdim, float* out, const float* inp, int h, int w);


// __global__ void attention_1() {
//     /* Naive attention implementation (no flash attention) */
// }
