import os
import time
from pathlib import Path

import torch

from util import load_cuda, load_cuda_inline, set_env_vars, load_and_compile_sources

set_env_vars()
module = load_and_compile_sources(Path("csrc"))

n = 1024
m1 = torch.randn(n, n, device="cuda")
m2 = torch.randn(n, n, device="cuda")
out = torch.zeros_like(m1)

iters = 1000
torch.cuda.synchronize()
t0 = time.perf_counter_ns()
for _ in range(iters):
    module.matmul_out(out, m1, m2)
torch.cuda.synchronize()
t1 = time.perf_counter_ns()
print(f"Time: {(t1-t0)/iters/1e3:.2f} µs")

"""
print("\nRunning PyTorch profiler...")
with torch.profiler.profile() as prof:
    for i in range(iters):
        module.matmul_out(out, m1, m2)
        torch.cuda.synchronize()

print(prof.key_averages().table())
"""