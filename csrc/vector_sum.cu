#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define BLOCK_SIZE 512

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CUDA_ERR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// NB: I'm assuming the array fits into two block sizes.

__global__ void vector_sum_kernel(float* inp, float* sum, unsigned int n) {
    __shared__ float shm[2*BLOCK_SIZE];

    const unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;

    // Load into shared memory
    shm[idx] = idx < n ? inp[idx] : 0;
    shm[idx + BLOCK_SIZE] = idx + BLOCK_SIZE < n ? inp[idx + BLOCK_SIZE] : 0;

    // Do the sum
    // Note that the shared memory size is always a power of 2, which
    // is a requirement for this to work.
    for (int stride = BLOCK_SIZE; stride >= 1; stride /= 2) {
        __syncthreads();
        if (idx < stride)
            shm[idx] += shm[idx + stride];
    }

    // Load the result
    if (threadIdx.x == 0)
        sum[0] = shm[0];
}

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

torch::Tensor vector_sum(torch::Tensor input) {
    CHECK_INPUT(input);
    unsigned int size = input.size(0);
    printf("size: %d\n", size);
    auto output = torch::empty(1, input.options());
    vector_sum_kernel<<<cdiv(size,BLOCK_SIZE), BLOCK_SIZE>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}