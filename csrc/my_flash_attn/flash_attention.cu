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


/*
__global__ void flash_attention_forward(float* out, const float* Q, const float* K, const float* V, int seq_len, int d_head) {
    // TODO    
}

// TODO: move to cpp file
torch::Tensor my_flash_attention_forward(const torch::Tensor& Q, const torch::Tensor& K, const torch::Tensor& V) {
    CHECK_INPUT(Q); CHECK_INPUT(K); CHECK_INPUT(V);

    int n = Q.size(0);
    int d = Q.size(1);

    TORCH_CHECK(n == K.size(0) && d == K.size(1) &&
                n == V.size(0) && d == V.size(1), "Wrong sizes!");

    auto out = torch::zeros({n, d}, A.options());
    auto l = torch::zeros({n}, A.options());
    auto m = torch::full({n}, -INFINITY, A.options());

    const int sram_size = ...; // TODO

    const int block_size_c = cdiv(sram_size, 4*d); 
    const int block_size_r = min(cdiv(sram_size, 4*d), d); 
    const int blocks = ...;  // TODO

    // auto f = [&](auto kf) { kf(
    //     blocks, block_size, out.data_ptr<float>(), inp.data_ptr<float>(), h, w); 
    // };

    // switch(block_size) {
    //     case 64:
    //         f(launch_softmax_kernel<64>); break;
    //     case 128:
    //         f(launch_softmax_kernel<128>); break;
    //     default:
    //         TORCH_CHECK(false, "Unsupported block size: ", block_size);

    return out;
}

// TODO: move to cpp file
// TODO: rewrite with proper arguments
template <int BLOCK_SIZE>
void launch_flash_attention_forward_kernel(dim3 gdim, dim3 bdim, float* out, const float* Q, const float* K, const float* V, int seq_len, int d_head) {
    flash_attention_forward<BLOCK_SIZE><<<gdim, bdim>>>(out, Q, K, V, seq_len, d_head);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
*/