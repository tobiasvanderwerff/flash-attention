import os
import time
from pathlib import Path
import torch
# import wurlitzer

from util import load_cuda, load_cuda_inline

# Set environment variables
os.environ['CUDA_LAUNCH_BLOCKING']='1'
device_props = torch.cuda.get_device_properties(0)
os.environ['TORCH_CUDA_ARCH_LIST'] = f'{device_props.major}.{device_props.minor}'

print(f"TORCH_CUDA_ARCH_LIST: {os.environ.get('TORCH_CUDA_ARCH_LIST')}")

# Load sources
src_dir = Path("csrc")
cu_path =  src_dir/"attention.cu"
cpp_path = src_dir/"attention.cpp"
funcs = ["matmul", "matmul_out"]
cuda_src = cu_path.read_text()
cpp_src = cpp_path.read_text()

# Compile
start_time = time.time()
module = load_cuda_inline(cuda_src, cpp_src, funcs, verbose=True, opt=True)
print(f"Compilation time: {(time.time()-start_time):.2f} s")

# Check correctness
torch.manual_seed(1)
n = 1000
m1 = torch.randn(n, n, device="cuda")
m2 = torch.randn(n, n, device="cuda")
res = module.matmul(m1, m2)
tr = torch.matmul(m1, m2).cpu()
is_correct = torch.isclose(res.cpu(), tr, atol=1e-4).all().item()

print(f"\nMax diff: {(res.cpu()-tr).abs().max().item()}")
print(f"Correct result: {is_correct}\n")

# Benchmark
print("Benchmarking...")
if is_correct:
    iters = 1000
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        module.matmul_out(res, m1, m2)
    torch.cuda.synchronize()
    t1 = time.perf_counter_ns()
    print(f"Time: {(t1-t0)/iters/1e3:.2f} µs")

    print("\nRunning PyTorch profiler...")
    with torch.profiler.profile() as prof:
        for i in range(iters):
            module.matmul_out(res, m1, m2)
            torch.cuda.synchronize()

    print(prof.key_averages().table())