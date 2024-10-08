# Flash Attention

An implementation of flash attention in CUDA C++ & CUTLASS, for learning purposes.

Requires a C++17 host compiler and CUDA >=11.4 for CUTLASS. See [CUTLASS docs for a list of supported GPUs](https://github.com/NVIDIA/cutlass/tree/main?tab=readme-ov-file#hardware).

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
pytest tests -v -s
```

Run benchmark:

```shell
python test_matmul.py
```

## Performance

I've been using Softmax as a case study for implementing various CUDA optimization techniques. Below are some performance results for Softmax. See `csrc/my_flash_attn/softmax.cu` for the kernels.

Tested on a Nvidia T4 GPU using a g4dn.xlarge EC2 instance.

Softmax perf. on (1024x1024 input):

| Kernel   | Perf  | Description                                                | Block size |
|----------|-------|------------------------------------------------------------|------------|
| Kernel 1 | 110 µs | Shared memory implementation                               | 128       |
| Kernel 2 | 85 µs  | Kernel 1 + uses warp-level operations to reduce shared memory usage   | 1024 |
| Kernel 3 | 73 µs  | Kernel 2 + float4 instead of float datatypes               | 32 |
| Kernel 4 | 72 µs  | Kernel 3 but with no shared memory and size 32 block sizes | 32 |
| Kernel 5 | 71 µs  | Same as kernel 1, but uses CUB for block-level reductions  | 128 |
    

## Profiling kernels

In order to hone in on the actual performance of the CUDA kernels, the best approach is perhaps to use the `ncu` profiler (see [running ncu profiler](#running-ncu-profiler) section below if you want to run `ncu` on a cloud GPU instance). I found it easiest to profile the pytest test cases set up in `test_attention.py`. For example, if I want to profile the softmax kernel, I run:

```shell
sudo ncu -k regex:softmax_kernel* pytest tests -k "test_softmax_kernel[1024-1024]"
```

The first `-k` flag will make sure that only the `softmax_kernel` function is being profiled. The second `-k` flag is for `pytest`, and ensures that only a specific test case is run (in this case, the `test_softmax_kernel` function with arguments `[1024, 1024]`). 

If you want to also profile kernels from other packages (e.g. Pytorch), it is best to first run ncu without a kernel specifier to see the list of kernel names that are profiled:

```shell
sudo ncu pytest tests
```

Note that depending on the number of test cases, this might produce a lot of output. Consider focusing on a single test case in `test_attention.py` to avoid this (e.g. passing `-k "test_softmax_kernel[1024-1024]"`).

Tip: it can be quite illuminating to compare the `ncu` output of your custom kernels (e.g. for Softmax) to their corresponding Pytorch implementations. The Pytorch implementations are a good upper bound on performance because they will be heavily optimized. E.g. my initial softmax kernel implementation was 5x as slow as Pytorch, even though occupancy and compute throughput was higher in the custom kernel.

## Running ncu profiler on a cloud GPU instance

The Nsight profiler (`ncu`) is a very useful tool to profile CUDA kernels.  However, it will not run out of the box on cloud GPUs. If you run `ncu`, you get an output like this:

```shell
$ ncu ./benchmark
==PROF== Connected to process 2258 (/mnt/tobias/benchmark)
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on enabling permissions and to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM
```
 

To fix this, you can run `ncu` with `sudo`. Note however that when you run `sudo`, your environment variables change, which means that `ncu` may no longer be on the PATH. This can be fixed by specifying the full path to `ncu`. E.g.:

```bash
which ncu  # check ncu path
sudo /opt/conda/envs/flash-attention/bin/ncu  # pass ncu path
```

In my case, ncu is provided through Conda. To make running ncu more convenient, you can directly add your Conda path to the "sudoers" file. Do this as follows:

```shell
sudo visudo
```

 Add your conda environment's bin directory to the Defaults secure_path line: 

```shell
Defaults secure_path="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/path/to/conda/env/bin"
```

Replace /path/to/conda/env/bin with the actual path to your conda environment's bin directory.


You can now run ncu simply by prepending `sudo`:

```shell
sudo ncu
```

## Lessons learned

- Almost always, ncu gives you the information you need to diagnose an underperforming kernel. Make sure to use `--set full` for the most complete output.
- Floating point calculations are *non-deterministic*. This can be a major pain because in certain situations, a kernel can give correct results in one run and incorrect results in the next, given the exact same input (this is the case for `softmax_kernel_3`, see its docstring for more info). It helps to turn off the `--use_fast_math` compiler flag if you want more deterministic results.
- Pattern: make computation more sequential to avoid block syncrhonization and shared memory access. E.g. when using warp operations.
- Common pattern: vs. Pytorch kernel, L1 cache throughput is higher for my custom kernel, but L2 cache throughput is lower. I think DRAM throughput may also be lower.
- Maximizing theoretical occupancy is not an obvious win. For instance, I had a kernel (softmax kernel 4) that had a theoretical occupancy of only 50%. This was due to the fact that I used block sizes of 32, and was dealing with a max block amount per SM of 16 (for the T4 GPU), which led to 16*32=512 threads active on each SM (max. is 1024). I got the theoretical occupancy to 100% by doubling the block sizes (while keeping the computational layout the same), but somehow the kernel became *slower* than before. Moreover, even though the theoretical occupancy was higher, the actual compute throughput did not increase. I figured the effect would only show up for large enough arrays that exceeded the number of CUDA cores in the GPU, but this also did not seem to make a difference. I'm still not sure why exactly this happens - ncu says it has something to do with warps being stalled waiting for a scoreboard dependency on a L1TEX operation. ~~It may be caused by register spilling. When I double the block size, it reduces the number of registers per thread by a factor of 2 (in ncu, "Block Limit Registers" goes from 48 to 24), and Registers Per Thread has a value of 34, which seems to indicate register spilling. Using more registers than what is available would lead to using slower local memory for the registers.~~ Edit: There are no signs of register spilling.
- At first, I wanted to use cuBLAS for matrix multiplication functionality over CUTLASS. However this does not work because, unlike CUTLASS, cuBLAS functions (e.g. for matmul) cannot be called inside CUDA kernels - only from host code. This makes cuBLAS unsuitable to implement Flash Attention, since it requires performing matmuls inside a kernel function (so that data does not get moved back and forth to global memory).
- ~~I don't see a clear advantage of using `torch.utils.cpp_extension.load` over `torch.utils.cpp_extension.load_inline`. In terms of compilation speed, I don't see a big difference.~~
    - Edit: it actually does help for faster compilation if you separate the .cpp and .cu files
- For large matrices and row-wise processing, using thread coarsening with smaller block sizes can actually lead to higher perf compared to using larger block sizes and fewer or no thread coarsening. E.g. for softmax. The point is that for larger matrices, the total number of threads in the GPU isn't sufficient to cover the whole matrix, so you have to find the best allocation of threads for processing the matrix. Let's say you compute softmax row-wise. You can then either dedicate more threads per row of the matrix (large block size) or process more rows in parallel with fewer threads per row (with more coarsening, i.e. single threads doing more work).
    - I guess the broader lesson from this is that if GPU resources are insufficient to perform a kernel fully parallelized in one run, you have to decide what are the best places to apply serialization.
    - See e.g. [this post](https://ajdillhoff.github.io/notes/gpu_performance_basics/#thread-coarsening) for an example of thread coarsening in the case of matrix multiplication.
- Wurlitzer doesn't seem to work in VSCode Jupyter notebooks. Specifically, it doesn't print the compilation output when running `load_inline`. This makes development in a notebook quite difficult because you cannot see the compiler errors.


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
- [x] Set up NCU/(Nsight?) profiling on Lightning Studio 
- [x] Profile kernels with NCU (eg to see whether an implementation is compute-bound or memory-bound and where things can be improved). Softmax is a good one to try out first.
- [x] Integrate with CUTLASS + CuTE (CUTLASS >=3.0)
- [x] Try out CUTLASS [sgemm_1.cu](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_1.cu) example and compare perf to my custom matmul (result: 50 ms vs 300 ms, but latter includes launch overhead)
- [ ] Look to ncu output for ways to further optimize softmax kernel
- [ ] C++ impl of flash attention using CUTLASS
- [ ] Instead of measuring time manually using `benchmark.py`, run `sudo ncu python benchmark.py | grep -i duration`. This should give the actual CUDA runtime for each kernel.
- [ ] How to unit test device functions??
- [ ] (optional) Triton implementation
- [ ] (optional): try out using CUDA with Docker for potentially easier dependency management: https://github.com/pbridger/cuda-experiments/blob/main/Dockerfile 

