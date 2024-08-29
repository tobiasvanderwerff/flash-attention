# Flash Attention

An implementation of flash attention, for learning purposes.

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

- ~~I don't see a clear advantage of using `torch.utils.cpp_extension.load` over `torch.utils.cpp_extension.load_inline`. In terms of compilation speed, I don't see a big difference.~~
    - Edit: it actually does help for faster compilation if you separate the .cpp and .cu files
- Wurlitzer doesn't seem to work in VSCode Jupyter notebooks. Specifically, it doesn't print the compilation output when running `load_inline`. This makes development in a notebook quite difficult because you cannot see the compiler errors.

## Lessons learned

- For large matrices and row-wise processing, using thread coarsening with smaller block sizes can actually lead to higher perf compared to using larger block sizes and fewer or no thread coarsening. E.g. for softmax. The point is that for larger matrices, the total number of threads in the GPU isn't sufficient to cover the whole matrix, so you have to find the best allocation of threads for processing the matrix. Let's say you compute softmax row-wise. You can then either dedicate more threads per row of the matrix (large block size) or process more rows in parallel with fewer threads per row (with more coarsening, i.e. single threads doing more work).
    - I guess the broader lesson from this is that if GPU resources are insufficient to perform a kernel fully parallelized in one run, you have to decide what are the best places to apply serialization.
    - See e.g. [this post](https://ajdillhoff.github.io/notes/gpu_performance_basics/#thread-coarsening) for an example of thread coarsening in the case of matrix multiplication.


## TODO

- [x] try out the `torch.utils.cpp_extension.load` function
- [x] Test loop for testing correctness and speed
- [x] Write naive matmul kernel
- [x] Write matmul kernel with output 
- [x] Write tiled matmul kernel
- [x] Make compilation faster. It currently takes 45-50 seconds each time...
- [x] Write softmax kernel
- [ ] Write naive attention kernel
    - [ ] use/write transpose matmul kernel instead of assuming that K is passed already transposed
- [ ] Look for ways to optimize softmax kernel
- [ ] See what solution chatgpt comes up with for softmax kernel
- [ ] Try out Nsight profiler to get insight into performance bottlenecks
- [ ] Triton?
- [ ] (optional): try out using CUDA with Docker for potentially easier dependency management: https://github.com/pbridger/cuda-experiments/blob/main/Dockerfile 

