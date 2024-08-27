#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// torch::Tensor matmul(const torch::Tensor& A, const torch::Tensor& B);
// torch::Tensor matmul_out(torch::Tensor& out, const torch::Tensor& A, const torch::Tensor& B); 


void launch_matmul_kernel(dim3 gdim, dim3 bdim, float* out, const float* A, const float* B, int h, int w, int k); 

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

torch::Tensor matmul_out(torch::Tensor& out, const torch::Tensor& A, const torch::Tensor& B) {
    CHECK_INPUT(A); CHECK_INPUT(B); CHECK_INPUT(out);
    int h = A.size(0);
    int w = B.size(1);
    int k = A.size(1);
    TORCH_CHECK(k == B.size(0) && out.sizes() == std::vector<int64_t>({h, w}), "Size mismatch!");
    TORCH_CHECK(out.scalar_type() == A.scalar_type() && out.scalar_type() == B.scalar_type(), "Scalar type mismatch!");

    dim3 bdim(16, 16);
    dim3 gdim(cdiv(w, bdim.x), cdiv(h, bdim.y));
    // matmul_kernel_1<<<gdim, bdim>>>(
    //     out.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), h, w, k);
    launch_matmul_kernel(
        gdim, bdim, out.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), h, w, k); 
    return out;
}

torch::Tensor matmul(const torch::Tensor& A, const torch::Tensor& B) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    int h = A.size(0);
    int w = B.size(1);
    int k = A.size(1);
    TORCH_CHECK(k==B.size(0), "Size mismatch!");
    auto out = torch::zeros({h, w}, A.options());

    dim3 bdim(16, 16);
    dim3 gdim(cdiv(w, bdim.x), cdiv(h, bdim.y));
    // matmul_kernel_1<<<gdim, bdim>>>(
    //     out.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), h, w, k);
    launch_matmul_kernel(
        gdim, bdim, out.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), h, w, k); 
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("matmul", torch::wrap_pybind_function(matmul), "matmul");
m.def("matmul_out", torch::wrap_pybind_function(matmul_out), "matmul_out");
}