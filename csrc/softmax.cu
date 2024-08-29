#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

__device__ inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

template <int BLOCK_SIZE>
__global__ void softmax_kernel(float* out, const float* inp, int h, int w) {
    /* Softmax applied row-wise. 

    softmax(x) = exp(x) / sum(exp(x))
    
    In this kernel, each block handles a single row.
    */

    __shared__ float shm_sum[BLOCK_SIZE];

    int tx = threadIdx.x;
    int bx = blockIdx.x;

    float sum = 0.0f;
    // Thread coarsening
    for (int bi = 0; bi < cdiv(w, BLOCK_SIZE); ++bi) {
        // Calculate exponent element-wise
        int idx = bx*w + bi*BLOCK_SIZE + tx;
        if (bi*BLOCK_SIZE + tx < w) {
            float e = expf(inp[idx]);  // TODO: use __expf?
            out[idx] = e;
            shm_sum[tx] = e;
        } else {
            shm_sum[tx] = 0.0f;
        }

        // Calculate sum of exponents
        // Note: block size is assumed to be power of 2
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

    // Divide by exponent sum
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

template void launch_softmax_kernel<64>(int gdim, int bdim, float* out, const float* inp, int h, int w);
template void launch_softmax_kernel<128>(int gdim, int bdim, float* out, const float* inp, int h, int w);
template void launch_softmax_kernel<256>(int gdim, int bdim, float* out, const float* inp, int h, int w);
template void launch_softmax_kernel<512>(int gdim, int bdim, float* out, const float* inp, int h, int w);
template void launch_softmax_kernel<1024>(int gdim, int bdim, float* out, const float* inp, int h, int w);

