import os
import time
from pathlib import Path

import torch

from util import set_env_vars, load_and_compile_sources


def benchmark(f, *args):
    iters = 1000
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        out = f(*args)
    torch.cuda.synchronize()
    t1 = time.perf_counter_ns()
    print(f"Time: {(t1-t0)/iters/1e3:.2f} µs")

    """
    print("\nRunning PyTorch profiler...")
    with torch.profiler.profile() as prof:
        for i in range(iters):
            out = module.matmul(m1, m2)
            torch.cuda.synchronize()
    print(prof.key_averages().table())
    """

# Setup
set_env_vars()
module = load_and_compile_sources(Path("csrc"), verbose=False)


# Benchmark matmul
n = 1024
print(f"\nBenchmarking matmul ({n}x{n})...")
m1 = torch.randn(n, n, device="cuda")
m2 = torch.randn(n, n, device="cuda")
benchmark(module.my_matmul, m1, m2)

# Benchmark matmul
n = 1024
print(f"\nBenchmarking matmul cuBLAS ({n}x{n})...")
m1 = torch.randn(n, n, device="cuda")
m2 = torch.randn(n, n, device="cuda")
benchmark(module.my_matmul_cublas, m1, m2)

# Benchmark softmax
n = 1024
print(f"\nBenchmarking softmax ({n}x{n})...")
# x = torch.randn(n, n, device="cuda")
x = torch.randn(n, n, device="cuda")
benchmark(module.my_softmax, x)
