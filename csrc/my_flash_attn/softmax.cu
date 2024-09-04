#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>


constexpr int WARP_SIZE = 32;

__device__ inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

template <int BLOCK_SIZE>
__forceinline__ __device__ float compute_row_max(const float* inp, float* shm, int n) {
    /* Calculate max value of the row */
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    float max_val = -INFINITY;
    for (int bi = 0; bi < cdiv(n, BLOCK_SIZE); ++bi) { // Thread coarsening
        int col = bi*BLOCK_SIZE + tx;
        shm[tx] = (col < n) ? inp[bx*n + col] : -INFINITY;

        __syncthreads();
        for (int stride = BLOCK_SIZE >> 1; stride >= 1; stride >>= 1) {
            if (tx < stride)
                shm[tx] = fmaxf(shm[tx], shm[tx + stride]);
            __syncthreads();
        }

        max_val = fmaxf(max_val, shm[0]);
    }
    return max_val;
}

template <int BLOCK_SIZE>
__global__ void softmax_kernel_1(float* out, const float* inp, int h, int w) {
    /* Softmax applied row-wise. 

    softmax(x) = exp(x) / sum(exp(x))
    
    In this kernel, each block handles a single row.
    */

    __shared__ float shm[BLOCK_SIZE];

    const int tx = threadIdx.x;
    const int bx = blockIdx.x;

    // Calculate max value of the row
    float max_val = compute_row_max<BLOCK_SIZE>(inp, shm, w);

    float sum = 0.0f;
    for (int bi = 0; bi < cdiv(w, BLOCK_SIZE); ++bi) { // Thread coarsening
        // Calculate exponent element-wise
        int idx = bx*w + bi*BLOCK_SIZE + tx;
        if (bi*BLOCK_SIZE + tx < w) {
            float e = expf(inp[idx] - max_val);  // TODO: use __expf?
            out[idx] = e;
            shm[tx] = e;
        } else {
            shm[tx] = 0.0f;
        }

        // Calculate sum of exponents
        // Note: block size is assumed to be power of 2
        __syncthreads();
        for (int stride = BLOCK_SIZE >> 1; stride >= 1; stride >>= 1) {
            if (tx < stride)
                shm[tx] += shm[tx + stride];
            __syncthreads();
        }

        // Let all threads save the intermediate sum
        sum += shm[0];
        __syncthreads();
    }

    // Divide by exponent sum
    for (int bi = 0; bi < cdiv(w, BLOCK_SIZE); ++bi) { // Thread coarsening
        if (bi*BLOCK_SIZE + tx < w) 
            out[bx*w + bi*BLOCK_SIZE + tx] /= sum;
    }
}


template <typename T>
__forceinline__ __device__ void warp_reduce_sum(T* val) {
    #pragma unroll
    for (int stride = 1; stride < WARP_SIZE; stride *= 2)
        *val += __shfl_xor_sync(0xffffffff, *val, stride);
}

__forceinline__ __device__ void warp_reduce_max(float* max_val) {
    #pragma unroll
    for (int stride = 1; stride < WARP_SIZE; stride *= 2) {
        *max_val = fmaxf(*max_val, __shfl_xor_sync(0xffffffff, *max_val, stride));
    }
}

template <int BLOCK_SIZE>
__global__ void softmax_kernel_2(float* out, const float* inp, int h, int w) {
    /* Softmax applied row-wise. 

    Replaces shared memory with warp-level shuffles.
    It also uses packed data structures.

    I.e. instead of thread coarsening by iterating over blocks:
    - reduce 2x per block using 32 sized warps (32*32=1024, which is assumed to be max block size)
        - or, when k is small, iterate over warps
    - use a packed data structure (float4?)

    Inspiration:
    - https://github.com/facebookincubator/AITemplate/wiki/How-to-write-a-fast-Softmax-CUDA-kernel%3F
    - https://developer.nvidia.com/blog/register-cache-warp-cuda/

    NB: be careful with register spilling (e.g. 25 int registers for a single thread is pushing it). Quote from second link:
    > the efficiency of the register cache is predicated on the availability of
    > spare registers. Otherwise, registers start spilling to global memory,
    > leading to a dramatic performance drop, as is the case for k=25 in Figure 6
    */

    // TODO: is it always safe to assume that warpSize=32?
    __shared__ float shm[WARP_SIZE];

    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int warp_no = tx / WARP_SIZE;
    const int warp_idx = tx % WARP_SIZE;

    // Calculate max value of the row
    float max_val[1] = {-INFINITY};
    for (int bi = 0; bi < cdiv(w, BLOCK_SIZE); ++bi) { // Thread coarsening
        int col = bi*BLOCK_SIZE + tx;

        // warpSize=32, block size 1024
        const float val = (col < w) ? inp[bx*w + col] : -INFINITY;

        *max_val = fmaxf(*max_val, val);

        warp_reduce_max(max_val);

        // Load warp results into warp 0
        if (warp_idx == 0)
            shm[warp_no] = *max_val;

        __syncthreads();

        // Final reduction happens in warp 0
        if (warp_no == 0) {
            *max_val = shm[warp_idx];
            warp_reduce_max(max_val);
            if (tx == 0)
                shm[0] = *max_val;
        }
        __syncthreads();

        *max_val = shm[0];
        // if (tx == 0 && bx == 0) printf("Max val: %f\n", *max_val);
    }

    float sum = 0.0f;
    for (int bi = 0; bi < cdiv(w, BLOCK_SIZE); ++bi) { // Thread coarsening
        // Calculate exponent element-wise
        int col = bi*BLOCK_SIZE + tx;
        int idx = bx*w + col;
        const float inp_exp = (col < w) ? expf(inp[idx] - *max_val) : 0;  // TODO: use __expf?
        if (col < w) out[idx] = inp_exp;

        // Calculate sum of exponents
        float block_sum[1] = {inp_exp};

        warp_reduce_sum<float>(block_sum);

        if (warp_idx == 0) 
            shm[warp_no] = *block_sum;

        __syncthreads();

        if (warp_no == 0) {
            *block_sum = shm[warp_idx];
            warp_reduce_sum<float>(block_sum);
            if (tx == 0)
                shm[0] = *block_sum;
        }

        __syncthreads();

        sum += shm[0];
    }

    // Divide by exponent sum
    for (int bi = 0; bi < cdiv(w, BLOCK_SIZE); ++bi) { // Thread coarsening
        if (bi*BLOCK_SIZE + tx < w) 
            out[bx*w + bi*BLOCK_SIZE + tx] /= sum;
    }
}

template <int BLOCK_SIZE>
void launch_softmax_kernel(int gdim, int bdim, float* out, const float* inp, int h, int w, int kernel_no) { 
    switch (kernel_no) {
        case 1:
            softmax_kernel_1<BLOCK_SIZE><<<gdim, bdim>>>(out, inp, h, w); break;
        case 2:
            softmax_kernel_2<BLOCK_SIZE><<<gdim, bdim>>>(out, inp, h, w); break;
        default:
            return;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template void launch_softmax_kernel<64>(int gdim, int bdim, float* out, const float* inp, int h, int w, int kernel_no);
template void launch_softmax_kernel<128>(int gdim, int bdim, float* out, const float* inp, int h, int w, int kernel_no);
template void launch_softmax_kernel<256>(int gdim, int bdim, float* out, const float* inp, int h, int w, int kernel_no);
template void launch_softmax_kernel<512>(int gdim, int bdim, float* out, const float* inp, int h, int w, int kernel_no);
template void launch_softmax_kernel<1024>(int gdim, int bdim, float* out, const float* inp, int h, int w, int kernel_no);



