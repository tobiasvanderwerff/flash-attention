#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>


constexpr int WARP_SIZE = 32;

__device__ inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

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
    float max_val = -INFINITY;
    for (int bi = 0; bi < cdiv(w, BLOCK_SIZE); ++bi) { // Thread coarsening
        int col = bi*BLOCK_SIZE + tx;
        shm[tx] = (col < w) ? inp[bx*w + col] : -INFINITY;

        __syncthreads();
        for (int stride = BLOCK_SIZE >> 1; stride >= 1; stride >>= 1) {
            if (tx < stride)
                shm[tx] = fmaxf(shm[tx], shm[tx + stride]);
            __syncthreads();
        }

        max_val = fmaxf(max_val, shm[0]);
    }

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

    It reduces 2 times per block using size 32 warps (32*32=1024, which is assumed to be max block size)

    Inspiration:
    - https://github.com/facebookincubator/AITemplate/wiki/How-to-write-a-fast-Softmax-CUDA-kernel%3F
    - https://developer.nvidia.com/blog/register-cache-warp-cuda/
    */

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
__global__ void softmax_kernel_3(float4* out, const float4* inp, int h, int w) {
    /* Softmax applied row-wise. 

    Note: row size must be a factor of 4.

    - Replaces shared memory with warp-level shuffles.
    - Uses packed data structures (float4 instead of float).

    Inspiration:
    - https://github.com/facebookincubator/AITemplate/wiki/How-to-write-a-fast-Softmax-CUDA-kernel%3F
    - https://developer.nvidia.com/blog/register-cache-warp-cuda/

    NB: be careful with register spilling (e.g. 25 int registers for a single thread is pushing it). Quote from second link:
    > The efficiency of the register cache is predicated on the availability of
    > spare registers. Otherwise, registers start spilling to global memory,
    > leading to a dramatic performance drop.
    */

    __shared__ float shm[WARP_SIZE];

    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int warp_no = tx / WARP_SIZE;
    const int warp_idx = tx % WARP_SIZE;

    // Calculate max value of the row
    float max_val[1] = {-INFINITY};
    for (int bi = 0; bi < cdiv(w/4, BLOCK_SIZE); ++bi) { // Thread coarsening
        int col = bi*BLOCK_SIZE + tx;

        // warpSize=32, block size 1024
        const float4 val = (col < w/4) ? inp[bx*(w/4) + col] : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);

        *max_val = fmaxf(*max_val, fmaxf(fmaxf(val.w, val.x), fmaxf(val.y, val.z)));

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
    for (int bi = 0; bi < cdiv(w/4, BLOCK_SIZE); ++bi) { // Thread coarsening
        // Calculate exponents element-wise
        int col = bi*BLOCK_SIZE + tx;
        int idx = bx*(w/4) + col;
        float4 sum4 = {0.0f, 0.0f, 0.0f, 0.0f};
        if (col < w/4) {
            float4 val = inp[idx];
            sum4.x = expf(val.x - *max_val);
            sum4.y = expf(val.y - *max_val);
            sum4.z = expf(val.z - *max_val);
            sum4.w = expf(val.w - *max_val);
            out[idx] = sum4;
        } 

        // Calculate sum of exponents across the block
        const float partial_sum = sum4.x + sum4.y + sum4.z + sum4.w;
        float block_sum[1] = {partial_sum};

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
    for (int bi = 0; bi < cdiv(w/4, BLOCK_SIZE); ++bi) { // Thread coarsening
        if (bi*BLOCK_SIZE + tx < w/4) {
            int idx = bx*(w/4) + bi*BLOCK_SIZE + tx;
            float4 val = out[idx];
            val.x /= sum;
            val.y /= sum;
            val.z /= sum;
            val.w /= sum;
            out[idx] = val;
        }
    }
}


template <int BLOCK_SIZE>
void launch_softmax_kernel(int gdim, int bdim, float* out, float* inp, int h, int w, int kernel_no) { 
    switch (kernel_no) {
        case 1:
            softmax_kernel_1<BLOCK_SIZE><<<gdim, bdim>>>(out, inp, h, w); break;
        case 2:
            softmax_kernel_2<BLOCK_SIZE><<<gdim, bdim>>>(out, inp, h, w); break;
        case 3:
            softmax_kernel_3<BLOCK_SIZE><<<gdim, bdim>>>(
                reinterpret_cast<float4*>(out), reinterpret_cast<float4*>(inp), h, w); 
            break;
        default:
            return;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template void launch_softmax_kernel<64>(int gdim, int bdim, float* out, float* inp, int h, int w, int kernel_no);
template void launch_softmax_kernel<128>(int gdim, int bdim, float* out, float* inp, int h, int w, int kernel_no);
template void launch_softmax_kernel<256>(int gdim, int bdim, float* out, float* inp, int h, int w, int kernel_no);
template void launch_softmax_kernel<512>(int gdim, int bdim, float* out, float* inp, int h, int w, int kernel_no);
template void launch_softmax_kernel<1024>(int gdim, int bdim, float* out, float* inp, int h, int w, int kernel_no);



