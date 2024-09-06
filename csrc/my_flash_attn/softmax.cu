/*

References for softmax:
- https://github.com/facebookincubator/AITemplate/wiki/How-to-write-a-fast-Softmax-CUDA-kernel%3F
- https://oneflow2020.medium.com/how-to-implement-an-efficient-softmax-cuda-kernel-oneflow-performance-optimization-sharing-405ad56e9031
	- Mainly interesting for the code samples, which are quite optimized. They are hard to read though, because they are meant to be quite generic implementations of softmax.
*/

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>


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
                shm[tx] = max(shm[tx], shm[tx + stride]);
            __syncthreads();
        }

        max_val = max(max_val, shm[0]);
    }

    float sum = 0.0f;
    for (int bi = 0; bi < cdiv(w, BLOCK_SIZE); ++bi) { // Thread coarsening
        // Calculate exponent element-wise
        int idx = bx*w + bi*BLOCK_SIZE + tx;
        if (bi*BLOCK_SIZE + tx < w) {
            float e = exp(inp[idx] - max_val);
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
        *max_val = max(*max_val, __shfl_xor_sync(0xffffffff, *max_val, stride));
    }
}

template <int BLOCK_SIZE>
__global__ void softmax_kernel_2(float* out, const float* inp, int h, int w) {
    /* Softmax applied row-wise. 

    Replaces shared memory with warp-level shuffles.

    It reduces 2 times per block using size 32 warps (32*32=1024, which is assumed to be max block size)

    Inspiration:
    - https://developer.nvidia.com/blog/register-cache-warp-cuda/

    NB: be careful with register spilling (e.g. 25 int registers for a single
    thread is pushing it). Quote from the link:
    > The efficiency of the register cache is predicated on the availability of
    > spare registers. Otherwise, registers start spilling to global memory,
    > leading to a dramatic performance drop.
    */

    __shared__ float shm[WARP_SIZE];

    const int tx = threadIdx.x;
    const int row = blockIdx.x;
    const int row_offset = row * w;
    const int warp_no = tx / WARP_SIZE;
    const int lane = tx % WARP_SIZE;

    // Calculate max value of the row
    float max_val[1] = {-INFINITY};
    for (int bi = 0; bi < cdiv(w, BLOCK_SIZE); ++bi) { // Thread coarsening
        int col = bi*BLOCK_SIZE + tx;

        // warpSize=32, block size 1024
        const float val = (col < w) ? inp[row_offset + col] : -INFINITY;

        *max_val = max(*max_val, val);

        warp_reduce_max(max_val);

        // Load warp results into warp 0
        if (lane == 0)
            shm[warp_no] = *max_val;

        __syncthreads();

        // Final reduction happens in warp 0
        if (warp_no == 0) {
            *max_val = shm[lane];
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
        int idx = row_offset + col;
        const float inp_exp = (col < w) ? exp(inp[idx] - *max_val) : 0;
        if (col < w) out[idx] = inp_exp;

        // Calculate sum of exponents
        float block_sum[1] = {inp_exp};

        warp_reduce_sum<float>(block_sum);

        if (lane == 0) 
            shm[warp_no] = *block_sum;

        __syncthreads();

        if (warp_no == 0) {
            *block_sum = shm[lane];
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
            out[row_offset + bi*BLOCK_SIZE + tx] /= sum;
    }
}

template <int BLOCK_SIZE>
__global__ void softmax_kernel_3(float4* out, const float4* inp, int h, int w) {
    /* Softmax applied row-wise. 

    Note: row size must be a factor of 4.

    - Uses a combination of shared memory and warp-level shuffles.
    - Uses packed data structures (float4 instead of float).

    It should be noted that for some reason, this kernel seems to produce
    incorrect results more often when a block size of 128 is used (NB: mostly
    when the `--use_fast_math` compiler flag is turned on). On the same input,
    it sometimes produces correct results and sometimes incorrect results. This
    is not the case for block sizes of 32 and 1024. My best guess is that the
    non-determinism of floating point arithmetic is somehow more problematic for
    that particular block size, but I'm not sure what it is about the kernel
    that makes it more sensitive to this.
    */

    constexpr int num_warps = BLOCK_SIZE / WARP_SIZE;

    __shared__ float shm[num_warps];

    const int tx = threadIdx.x;
    const int row = blockIdx.x;
    const int row_offset = row * (w/4);
    const int warp_no = tx / WARP_SIZE;
    const int lane = tx % WARP_SIZE;

    static_assert(num_warps <= 32);

    // Calculate max value of the row
    float max_val[1] = {-INFINITY};
    for (int bi = 0; bi < cdiv(w/4, BLOCK_SIZE); ++bi) { // Thread coarsening
        int col = bi*BLOCK_SIZE + tx;

        // warpSize=32, block size 1024
        const float4 val = (col < w/4) ? inp[row_offset + col] : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);

        *max_val = max(*max_val, max(max(val.w, val.x), max(val.y, val.z)));

        warp_reduce_max(max_val);

        // Load all warp results into shmem
        if (lane == 0)
            shm[warp_no] = *max_val;

        __syncthreads();

        // Final reduction happens in warp 0
        if (warp_no == 0) {
            *max_val = (lane < num_warps) ? shm[lane] : -INFINITY;
            warp_reduce_max(max_val);
            if (tx == 0)
                shm[0] = *max_val;
        }
        __syncthreads();

        *max_val = shm[0];
    }
    // if (tx == 0 && row == 113) printf("\nMax val: %f\n", *max_val);

    float sum = 0.0f;
    for (int bi = 0; bi < cdiv(w/4, BLOCK_SIZE); ++bi) { // Thread coarsening
        // Calculate exponents element-wise
        int col = bi*BLOCK_SIZE + tx;
        int idx = row_offset + col;
        float4 sum4 = {0.0f, 0.0f, 0.0f, 0.0f};
        if (col < w/4) {
            float4 val = inp[idx];
            sum4.x = exp(val.x - *max_val);
            sum4.y = exp(val.y - *max_val);
            sum4.z = exp(val.z - *max_val);
            sum4.w = exp(val.w - *max_val);
            out[idx] = sum4;
        } 

        // Calculate sum of exponents across the block
        const float partial_sum = sum4.x + sum4.y + sum4.z + sum4.w;
        float block_sum[1] = {partial_sum};

        warp_reduce_sum<float>(block_sum);

        if (lane == 0) 
            shm[warp_no] = *block_sum;

        __syncthreads();

        if (warp_no == 0) {
            *block_sum = (lane < num_warps) ? shm[lane] : 0.0f;
            warp_reduce_sum<float>(block_sum);
            if (tx == 0)
                shm[0] = *block_sum;
        }

        __syncthreads();

        sum += shm[0];
    }
    // if (tx == 0 && row == 113) printf("\nSum: %f\n", sum);

    // if (tx == 0 && row == 0) {
    //     // Print some values
    //     printf("\n");
    //     printf("Before division by sum\n");
    //     for (int i = 0; i < 4; ++i) {
    //         printf("out[%d]: %f %f %f %f\n", i, out[i].x, out[i].y, out[i].z, out[i].w);
    //     }
    // }
    // Divide by exponent sum
    for (int bi = 0; bi < cdiv(w/4, BLOCK_SIZE); ++bi) { // Thread coarsening
        if (bi*BLOCK_SIZE + tx < w/4) {
            int idx = row_offset + bi*BLOCK_SIZE + tx;
            float4 val = out[idx];
            val.x /= sum;
            val.y /= sum;
            val.z /= sum;
            val.w /= sum;
            out[idx] = val;
        }
    }
    // if (tx == 0 && row == 0) {
    //     // Print some values
    //     printf("\n");
    //     printf("After division by sum\n");
    //     for (int i = 0; i < 4; ++i) {
    //         printf("out[%d]: %f %f %f %f\n", i, out[i].x, out[i].y, out[i].z, out[i].w);
    //     }
    // }
}

template <int BLOCK_SIZE>
__global__ void softmax_kernel_4(float4* out, const float4* inp, int h, int w) {
    /* Softmax applied row-wise. 

    Same as kernel 3 but omits shared memory.

    Note: row size must be a factor of 4.
    */

    const int tx = threadIdx.x;
    const int row = blockIdx.x;
    // const int row = 2 * blockIdx.y + blockIdx.x;
    // const int row = 2 * blockIdx.x + threadIdx.y;  // ncu complains about warp stalling if I do this

    // const int tx = threadIdx.x % WARP_SIZE;  // unusual, I know
    // const int row = 2 * blockIdx.x + threadIdx.x / WARP_SIZE;

    const int row_offset = row * (w/4);

    if (row >= h) return;

    // Calculate max value of the row
    float max_val[1] = {-INFINITY};
    for (int bi = 0; bi < cdiv(w/4, WARP_SIZE); ++bi) { // Thread coarsening
        int col = bi*WARP_SIZE + tx;

        const float4 val = (col < w/4) ? inp[row_offset + col] : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);

        *max_val = max(*max_val, max(max(val.w, val.x), max(val.y, val.z)));

        warp_reduce_max(max_val);
    }
    // if (tx == 0 && blockIdx.x == 0) printf("Max val: %f\n", *max_val);

    float sum = 0.0f;
    for (int bi = 0; bi < cdiv(w/4, WARP_SIZE); ++bi) { // Thread coarsening
        // Calculate exponents element-wise
        int col = bi*WARP_SIZE + tx;
        int idx = row_offset + col;
        float4 sum4 = {0.0f, 0.0f, 0.0f, 0.0f};
        if (col < w/4) {
            float4 val = inp[idx];
            sum4.x = exp(val.x - *max_val);
            sum4.y = exp(val.y - *max_val);
            sum4.z = exp(val.z - *max_val);
            sum4.w = exp(val.w - *max_val);
            out[idx] = sum4;
        } 

        // Calculate sum of exponents across the block
        float block_sum[1] = { sum4.x + sum4.y + sum4.z + sum4.w };

        warp_reduce_sum<float>(block_sum);

        sum += *block_sum;
    }

    // Divide by exponent sum
    for (int bi = 0; bi < cdiv(w/4, WARP_SIZE); ++bi) { // Thread coarsening
        if (bi*WARP_SIZE + tx < w/4) {
            int idx = row_offset + bi*WARP_SIZE + tx;
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
__global__ void softmax_kernel_5(float* out, const float* inp, int h, int w) {
    /* Softmax applied row-wise. 

    Same as kernel 1, but uses CUB for block-level reductions.
    */
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;

    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float shm;

    const int tx = threadIdx.x;
    const int bx = blockIdx.x;

    // Calculate max value of the row
    float max_val = -INFINITY;
    for (int bi = 0; bi < cdiv(w, BLOCK_SIZE); ++bi) { // Thread coarsening
        int col = bi*BLOCK_SIZE + tx;
        float val = (col < w) ? inp[bx*w + col] : -INFINITY;
        float block_max = BlockReduce(temp_storage).Reduce(val, cub::Max());
        max_val = max(max_val, block_max);
    }

    // Distribute result
    if (tx == 0) shm = max_val;
    __syncthreads();
    max_val = shm;

    // Calculate sum of the row
    float sum = 0.0f;
    for (int bi = 0; bi < cdiv(w, BLOCK_SIZE); ++bi) { // Thread coarsening
        int idx = bx*w + bi*BLOCK_SIZE + tx;
        float val = 0.0f;
        if (bi*BLOCK_SIZE + tx < w) {
            float e = exp(inp[idx] - max_val);
            out[idx] = e;
            val = e;
        } 
        sum += BlockReduce(temp_storage).Sum(val);
    }

    // Distribute result
    if (tx == 0) shm = sum;
    __syncthreads();
    sum = shm;

    // Divide by exponent sum
    for (int bi = 0; bi < cdiv(w, BLOCK_SIZE); ++bi) { // Thread coarsening
        if (bi*BLOCK_SIZE + tx < w) 
            out[bx*w + bi*BLOCK_SIZE + tx] /= sum;
    }
}


template <int BLOCK_SIZE>
void launch_softmax_kernel(dim3 gdim, dim3 bdim, float* out, float* inp, int h, int w, int kernel_no) { 
    switch (kernel_no) {
        case 1:
            softmax_kernel_1<BLOCK_SIZE><<<gdim, bdim>>>(out, inp, h, w); break;
        case 2:
            softmax_kernel_2<BLOCK_SIZE><<<gdim, bdim>>>(out, inp, h, w); break;
        case 3:
            softmax_kernel_3<BLOCK_SIZE><<<gdim, bdim>>>(
                reinterpret_cast<float4*>(out), reinterpret_cast<float4*>(inp), h, w); 
            break;
        case 4:
            softmax_kernel_4<BLOCK_SIZE><<<gdim, bdim>>>(
                reinterpret_cast<float4*>(out), reinterpret_cast<float4*>(inp), h, w); 
            break;
        case 5:
            softmax_kernel_5<BLOCK_SIZE><<<gdim, bdim>>>(out, inp, h, w); break;
        default:
            return;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template void launch_softmax_kernel<32>  (dim3 gdim, dim3 bdim, float* out, float* inp, int h, int w, int kernel_no);
template void launch_softmax_kernel<64>  (dim3 gdim, dim3 bdim, float* out, float* inp, int h, int w, int kernel_no);
template void launch_softmax_kernel<128> (dim3 gdim, dim3 bdim, float* out, float* inp, int h, int w, int kernel_no);
template void launch_softmax_kernel<256> (dim3 gdim, dim3 bdim, float* out, float* inp, int h, int w, int kernel_no);
template void launch_softmax_kernel<512> (dim3 gdim, dim3 bdim, float* out, float* inp, int h, int w, int kernel_no);
template void launch_softmax_kernel<1024>(dim3 gdim, dim3 bdim, float* out, float* inp, int h, int w, int kernel_no);



