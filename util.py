import os
import time

import torch
from torch.utils.cpp_extension import load_inline, load


def load_cuda_inline(cuda_src, cpp_src, funcs, opt=False, verbose=False, build_directory=None, name=None):
    if name is None:
        name = funcs[0]
    return load_inline(cuda_sources=[cuda_src], cpp_sources=[cpp_src], functions=funcs,
                       extra_cuda_cflags=["-O2"] if opt else [],
                       verbose=verbose, name=name, build_directory=build_directory)


def load_cuda(sources, opt=False, verbose=False, build_directory=None):
    return load(sources=sources, extra_cuda_cflags=["-O2"] if opt else [], 
                verbose=verbose, name="ext", build_directory=build_directory)


def set_env_vars():
    # Set environment variables
    os.environ['CXX'] = '/usr/lib/ccache/g++'
    os.environ['CC'] = '/usr/lib/ccache/gcc'
    os.environ['CUDA_LAUNCH_BLOCKING']='1'
    device_props = torch.cuda.get_device_properties(0)
    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{device_props.major}.{device_props.minor}'

    print(f"TORCH_CUDA_ARCH_LIST: {os.environ.get('TORCH_CUDA_ARCH_LIST')}")


def load_and_compile_sources(src_dir, verbose=True):
    # Load sources
    src = [p for p in src_dir.iterdir() if p.suffix == ".cu" or p.suffix == ".cpp"]
    print(f"Found sources: {[str(p) for p in src]}\n")

    # Compile
    start_time = time.time()
    module = load_cuda(src, verbose=verbose, opt=True)
    print(f"Compilation time: {(time.time()-start_time):.2f} s\n")
    return module