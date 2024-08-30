#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

__device__ inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }


template <int BLOCK_SIZE>
__global__ void softmax_kernel_2(float* out, const float* inp, int h, int w) {
    // TODO
}

__inline__ __device__ void warp_reduce_sum(float* out, float val) {
    for (int stride = 1; stride < warpSize, stride *= 2)
        val += __shfl_xor_sync(0xffffffff, val, stride);
    out[0] = sum;
}

template <int BLOCK_SIZE>
__global__ void softmax_kernel(float* out, const float* inp, int h, int w) {
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

template <int BLOCK_SIZE>
void launch_softmax_kernel(int gdim, int bdim, float* out, const float* inp, int h, int w) { 
    softmax_kernel<BLOCK_SIZE><<<gdim, bdim>>>(out, inp, h, w);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template void launch_softmax_kernel<64>(int gdim, int bdim, float* out, const float* inp, int h, int w);
template void launch_softmax_kernel<128>(int gdim, int bdim, float* out, const float* inp, int h, int w);
template void launch_softmax_kernel<256>(int gdim, int bdim, float* out, const float* inp, int h, int w);
template void launch_softmax_kernel<512>(int gdim, int bdim, float* out, const float* inp, int h, int w);
template void launch_softmax_kernel<1024>(int gdim, int bdim, float* out, const float* inp, int h, int w);

