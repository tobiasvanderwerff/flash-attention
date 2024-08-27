# Flash Attention

An implementation of flash attention, for learning purposes.

## Setup

```shell
conda create -n flash-attention
conda activate flash-attention
conda install cuda -c nvidia/label/cuda-12.4.0
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia/label/cuda-12.4.0
```

## Observations

- I don't see a clear advantage of using `torch.utils.cpp_extension.load` over `torch.utils.cpp_extension.load_inline`. In terms of compilation speed, I don't see a big difference.
- Wurlitzer doesn't seem to work in VSCode Jupyter notebooks. Specifically, it doesn't print the compilation output when running `load_inline`. This makes development in a notebook quite difficult because you cannot see the compiler errors.


## TODO

- [x] try out the `torch.utils.cpp_extension.load` function
- [x] Test loop for testing correctness and speed
- [x] Write naive matmul kernel
- [x] Write matmul kernel with output 
- [x] Write tiled matmul kernel
- [ ] Make compilation faster. It currently takes 45-50 seconds each time...
- [ ] Write naive attention kernel
- [ ] fix compile errors not being shown in vscode notebook
- [ ] (optional): try out using CUDA with Docker for potentially easier dependency management: https://github.com/pbridger/cuda-experiments/blob/main/Dockerfile 

