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


## TODO

- [x] try out the `torch.utils.cpp_extension.load` function
- [x] Test loop for testing correctness and speed
- [x] Write naive matmul kernel
- [ ] Write tiled matmul kernel
- [ ] Write naive attention kernel
- [ ] (optional): try out using CUDA with Docker for potentially easier dependency management: https://github.com/pbridger/cuda-experiments/blob/main/Dockerfile 

