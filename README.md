# Flash Attention

An implementation of flash attention in CUDA C++, for learning purposes.

Requires CUDA >=11.4 for CUTLASS. See [CUTLASS docs for a list of supported GPUs](https://github.com/NVIDIA/cutlass/tree/main?tab=readme-ov-file#hardware).

## Setup

Install `ccache` (used for faster compilation by using caching):

```shell
sudo apt update
sudo apt install software-properties-common ccache
```

Install dependencies:

```shell
conda create -n flash-attention
conda activate flash-attention
conda install cuda -c nvidia/label/cuda-12.4.0
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia/label/cuda-12.4.0
pip install -r requirements.txt
```

## Run

Run tests:

```shell
pytest -v -ss
```

Run benchmark:

```shell
python test_matmul.py
```


## Observations

- At first, I wanted to use cuBLAS for matrix multiplication functionality over CUTLASS. However this does not work because, unlike CUTLASS, cuBLAS functions (e.g. for matmul) cannot be called inside CUDA kernels - only from host code. This makes cuBLAS unsuitable to implement Flash Attention, since it requires performing matmuls inside a kernel function (so that data does not get moved back and forth to global memory).
- ~~I don't see a clear advantage of using `torch.utils.cpp_extension.load` over `torch.utils.cpp_extension.load_inline`. In terms of compilation speed, I don't see a big difference.~~
    - Edit: it actually does help for faster compilation if you separate the .cpp and .cu files
- Wurlitzer doesn't seem to work in VSCode Jupyter notebooks. Specifically, it doesn't print the compilation output when running `load_inline`. This makes development in a notebook quite difficult because you cannot see the compiler errors.

## Lessons learned

- For large matrices and row-wise processing, using thread coarsening with smaller block sizes can actually lead to higher perf compared to using larger block sizes and fewer or no thread coarsening. E.g. for softmax. The point is that for larger matrices, the total number of threads in the GPU isn't sufficient to cover the whole matrix, so you have to find the best allocation of threads for processing the matrix. Let's say you compute softmax row-wise. You can then either dedicate more threads per row of the matrix (large block size) or process more rows in parallel with fewer threads per row (with more coarsening, i.e. single threads doing more work).
    - I guess the broader lesson from this is that if GPU resources are insufficient to perform a kernel fully parallelized in one run, you have to decide what are the best places to apply serialization.
    - See e.g. [this post](https://ajdillhoff.github.io/notes/gpu_performance_basics/#thread-coarsening) for an example of thread coarsening in the case of matrix multiplication.

## What's so special about Flash Attention?

Flash Attention is, at its core, a type of layer fusion (one of the most important optimizations to make neural nets faster). Specifically, the Attention operation consists of three basic operations - matmul, softmax, matmul (in the most simplified form). Flash Attention applies layer fusion by putting all three operations into a single kernel to avoid having to move data in and out of global memory for each kernel call. 

Normally, there are compilers that can do layer fusion for you in an automated way (e.g. `torch.compile`), but this only works for simple fusions. A compiler like `torch.compile`, couldn't have merged the three operations for attention because there is not enough shared memory to perform all three operations without writing to global memory. This is where Flash Attention uses a mathematical trick to compute softmax in an online fashion so that it can be decomposed across multiple thread blocks. This way, it can be computed in the same kernel as the matmuls. 

This optimization to the attention mechanism is a big deal because the attention function is used *a lot* in modern deep learning (LLMs are mostly large Transformers that recursively apply attention).

## TODO

- [x] try out the `torch.utils.cpp_extension.load` function
- [x] Test loop for testing correctness and speed
- [x] Write naive matmul kernel
- [x] Write matmul kernel with output 
- [x] Write tiled matmul kernel
- [x] Make compilation faster. It currently takes 45-50 seconds each time...
- [x] Write softmax kernel
- [x] Write naive attention kernel
- [x] See what solution chatgpt comes up with for softmax kernel
- [x] Python impl of flash attention
- [x] Try out cuBLAS
- [ ] Integrate with CUTLASS + CuTE (CUTLASS >=3.0)
- [ ] C++ impl of flash attention
- [ ] Set up NCU/(Nsight?) profiling on Lightning Studio 
- [ ] Profile kernels with NCU (eg to see whether an implementation is compute-bound or memory-bound and where things can be improved). Softmax is a good one to try out first.
- [ ] Look for ways to optimize softmax kernel
- [ ] Write transpose matmul kernel (?)
- [ ] (optional) Triton implementation
- [ ] (optional): try out using CUDA with Docker for potentially easier dependency management: https://github.com/pbridger/cuda-experiments/blob/main/Dockerfile 

