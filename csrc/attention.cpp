#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <iostream>

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

bool is_power_of_two(int n) {
    if (n <= 0) return false;
    return (n & (n - 1)) == 0;
}

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

template <int TILE_SIZE>
void launch_matmul_kernel(dim3 gdim, dim3 bdim, float* out, const float* A, const float* B, int h, int w, int k); 

template <int BLOCK_SIZE>
void launch_softmax_kernel(int gdim, int bdim, float* out, const float* inp, int h, int w); 


torch::Tensor my_matmul_out(torch::Tensor& out, const torch::Tensor& A, const torch::Tensor& B) {
    CHECK_INPUT(A); CHECK_INPUT(B); CHECK_INPUT(out);
    int h = A.size(0);
    int w = B.size(1);
    int k = A.size(1);
    TORCH_CHECK(k == B.size(0) && out.sizes() == std::vector<int64_t>({h, w}), "Size mismatch!");
    TORCH_CHECK(out.scalar_type() == A.scalar_type() && out.scalar_type() == B.scalar_type(), "Scalar type mismatch!");

    dim3 bdim(16, 16);
    dim3 gdim(cdiv(w, bdim.x), cdiv(h, bdim.y));
    launch_matmul_kernel<16>(
        gdim, bdim, out.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), h, w, k); 
    return out;
}

torch::Tensor my_matmul(const torch::Tensor& A, const torch::Tensor& B) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    int h = A.size(0);
    int w = B.size(1);
    int k = A.size(1);
    TORCH_CHECK(k==B.size(0), "Size mismatch!");
    auto out = torch::zeros({h, w}, A.options());

    /*
    cudaDeviceProp devProp;
    CUDA_ERR(cudaGetDeviceProperties(&devProp, 0));
    int maxThreads = devProp.maxThreadsPerBlock;
    // Dynamicly calculated shared memory size
    size_t requiredSize = static_cast(maxThreads) * 2 * sizeof(float);
    size_t size = min(devProp.sharedMemPerBlock, requiredSize);
    int TW = std::sqrt(maxThreads);
    std::cout << "Shared memory size: " << devProp.sharedMemPerBlock << std::endl;
    std::cout << "Max threads per block: " << maxThreads << std::endl;
    */

    const int tile_size = 16;

    dim3 bdim(tile_size, tile_size);
    dim3 gdim(cdiv(w, bdim.x), cdiv(h, bdim.y));

    auto f = [&](auto kf) { kf(
        gdim, bdim, out.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), h, w, k); 
    };

    switch(tile_size) {
        case 16:
            f(launch_matmul_kernel<16>); break;
        case 32:
            f(launch_matmul_kernel<32>); break;
        default:
            TORCH_CHECK(false, "Unsupported tile size: ", tile_size);
    }
    return out;
}


torch::Tensor my_softmax(const torch::Tensor& inp) {
    CHECK_INPUT(inp);
    int h = inp.size(0);
    int w = inp.size(1);
    auto out = torch::zeros({h, w}, inp.options());

    const int block_size = 256;
    const int blocks = h;

    TORCH_CHECK(is_power_of_two(block_size), "Block size is expected to be a power of 2. Got ", block_size);

    auto f = [&](auto kf) { kf(
        blocks, block_size, out.data_ptr<float>(), inp.data_ptr<float>(), h, w); 
    };

    switch(block_size) {
        case 64:
            f(launch_softmax_kernel<64>); break;
        case 128:
            f(launch_softmax_kernel<128>); break;
        case 256:
            f(launch_softmax_kernel<256>); break;
        case 512:
            f(launch_softmax_kernel<512>); break;
        case 1024:
            f(launch_softmax_kernel<1024>); break;
        default:
            TORCH_CHECK(false, "Unsupported block size: ", block_size);
    }
    return out;
}

torch::Tensor my_attention(const torch::Tensor& Q, const torch::Tensor& K, const torch::Tensor& V) {
    /* Naive attention implementation (no flash attention) 

        Require: Matrices Q, K, V ∈ R^N×d in HBM.
        1: Load Q, K by blocks from HBM, compute S = QK^T , write S to HBM.
        2: Read S from HBM, compute P = softmax(S), write P to HBM.
        3: Load P and V by blocks from HBM, compute O = PV, write O to HBM.
        4: Return O
    */
    CHECK_INPUT(Q); CHECK_INPUT(K); CHECK_INPUT(V);

    int n = Q.size(0);
    int d = Q.size(1);

    TORCH_CHECK(d == K.size(0) && n == K.size(1) &&
                n == V.size(0) && d == V.size(1), "Wrong sizes!");

    auto QK = my_matmul(Q, K);
    auto weights = my_softmax(QK);
    auto out = my_matmul(weights, V);

    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("my_matmul", torch::wrap_pybind_function(my_matmul), "my_matmul");
m.def("my_matmul_out", torch::wrap_pybind_function(my_matmul_out), "my_matmul_out");
m.def("my_softmax", torch::wrap_pybind_function(my_softmax), "my_softmax");
m.def("my_attention", torch::wrap_pybind_function(my_attention), "my_attention");
}