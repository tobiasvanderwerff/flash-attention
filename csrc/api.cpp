#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include "my_flash_attn/cublas.hpp"
#include "my_flash_attn/util.hpp"
#include "cutlass/cutlass.h"


template <int TILE_SIZE>
void launch_matmul_kernel(dim3 gdim, dim3 bdim, float* out, const float* A, const float* B, int h, int w, int k); 

torch::Tensor my_matmul_out(torch::Tensor& out, const torch::Tensor& A, const torch::Tensor& B) {
    CHECK_INPUT(A); CHECK_INPUT(B); CHECK_INPUT(out);
    int h = A.size(0);
    int w = B.size(1);
    int k = A.size(1);
    TORCH_CHECK(k == B.size(0) && out.sizes() == std::vector<int64_t>({h, w}), "Size mismatch!");
    TORCH_CHECK(out.scalar_type() == A.scalar_type() && out.scalar_type() == B.scalar_type(), "Scalar type mismatch!");

    /*
    cudaDeviceProp devProp;
    CHECK_CUDA_ERR(cudaGetDeviceProperties(&devProp, 0));
    int maxThreads = devProp.maxThreadsPerBlock;
    // Dynamicly calculated shared memory size
    size_t requiredSize = static_cast(maxThreads) * 2 * sizeof(float);
    size_t size = min(devProp.sharedMemPerBlock, requiredSize);
    int TW = std::sqrt(maxThreads);
    std::cout << "Shared memory size: " << devProp.sharedMemPerBlock << std::endl;
    std::cout << "Max threads per block: " << maxThreads << std::endl;
    */

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

torch::Tensor my_matmul_cublas(const torch::Tensor& A, const torch::Tensor& B) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    int m = A.size(0);
    int n = B.size(1);
    int k = A.size(1);
    TORCH_CHECK(k==B.size(0), "Size mismatch!");
    auto out = torch::zeros({m, n}, A.options());

    // TODO: find a better place to initialize/destroy the handle
    cublasHandle_t cublas_handle;
    CUBLAS_ERR(cublasCreate(&cublas_handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Note that CUBLAS expects column-major matrices. Since the input matrices are
    // row-major, we apply the following trick (see https://stackoverflow.com/a/56064726).
    // First note that that tranposing a matrix is the same as switching between row-major
    // and column-major layout. Then,
    // C_row-maj = C_col-maj^T
    //           = (A_col-maj * B_col-maj)^T  // expand C
    //           = B_col-maj^T * A_row-maj^T  // apply transpose rule (AB)^T = B^T A^T
    //           = B_row-maj * A_row-maj
    CUBLAS_ERR(
        cublasSgemm(cublas_handle, 
                    CUBLAS_OP_N, CUBLAS_OP_N, 
                    n, m, k, 
                    &alpha, 
                    B.data_ptr<float>(), n,
                    A.data_ptr<float>(), k,
                    &beta,
                    out.data_ptr<float>(), n)
    );

    CUBLAS_ERR(cublasDestroy(cublas_handle));

    return out;
}

template <int BLOCK_SIZE>
void launch_softmax_kernel(dim3 gdim, dim3 bdim, float* out, float* inp, int h, int w, int kernel_no); 

torch::Tensor my_softmax(const torch::Tensor& inp, int kernel_no = 1) {
    CHECK_INPUT(inp);
    int h = inp.size(0);
    int w = inp.size(1);
    auto out = torch::zeros({h, w}, inp.options());

    constexpr int block_size = 1024;
    // constexpr int block_size = 128;
    // constexpr int block_size = 64;
    // constexpr int block_size = 32;

    dim3 gdim, bdim;
    switch (kernel_no) {
        case 1:
            gdim = dim3(h);
            bdim = dim3(block_size);
            break;
        case 2:
            gdim = dim3(h);
            bdim = dim3(block_size);
            TORCH_CHECK(block_size == 1024, "Block size must be 1024 for kernel 2")
            break;
        case 3:
            gdim = dim3(h);
            bdim = dim3(block_size);
            TORCH_CHECK(block_size != 128, "Block size 128 produces unstable results for this kernel")
            TORCH_CHECK(w % 4 == 0, "Input size must be a multiple of 4 for kernel 3")
            TORCH_CHECK(block_size % 32 == 0, "Block size must be a multiple of 32 for kernel 3")
            break;
        case 4: 
            TORCH_CHECK(block_size == 32, "Block size must be 32 for kernel 4")
            if (1) {
                gdim = dim3(h);
                bdim = dim3(block_size);
            } else {
                // gdim = dim3(2, cdiv(h, 2));
                // bdim = dim3(block_size);
                // Increases theoretical occupancy but does not increase compute throughput (unclear why).
                gdim = dim3(cdiv(h, 2));
                // bdim = dim3(block_size, 2);
                bdim = dim3(2*block_size);
            }
            break;
        default:
            TORCH_CHECK(false, "Unsupported kernel number: ", kernel_no);
    }

    TORCH_CHECK(is_power_of_two(block_size), "Block size is expected to be a power of 2. Got ", block_size);

    launch_softmax_kernel<block_size>(gdim, bdim, out.data_ptr<float>(), inp.data_ptr<float>(), h, w, kernel_no);

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

    // auto QK = my_matmul(Q, K);
    auto QK = my_matmul_cublas(Q, K);
    auto weights = my_softmax(QK);
    // auto out = my_matmul(weights, V);
    auto out = my_matmul_cublas(weights, V);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("my_softmax", torch::wrap_pybind_function(my_softmax), "my_softmax");
m.def("my_attention", torch::wrap_pybind_function(my_attention), "my_attention");
m.def("my_matmul", torch::wrap_pybind_function(my_matmul), "my_matmul");
m.def("my_matmul_cublas", torch::wrap_pybind_function(my_matmul_cublas), "my_matmul_cublas");
m.def("my_matmul_out", torch::wrap_pybind_function(my_matmul_out), "my_matmul_out");
}