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

#define TILE_SIZE 16

__global__ void attention_1() {
    /* Naive attention implementation (no flash attention) */
}

__global__ void matmul_kernel(float* out, const float* A, const float* B, int h, int w, int k) {
    /* Matmul using tiling */

    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int r = blockDim.y * by + ty;  // row
    const int c = blockDim.x * bx + tx;  // col
    const int idx = r * w + c;

    // TODO

}

 // TODO
/* torch::Tensor attention() {} */
