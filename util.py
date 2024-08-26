from torch.utils.cpp_extension import load_inline, load


def load_cuda_inline(cuda_src, cpp_src, funcs, opt=False, verbose=False, build_directory=None):
    return load_inline(cuda_sources=[cuda_src], cpp_sources=[cpp_src], functions=funcs,
                       extra_cuda_cflags=["-O2"] if opt else [], verbose=verbose, name="inline_ext",
                       build_directory=build_directory)


def load_cuda(sources, opt=False, verbose=False, build_directory=None):
    return load(sources=sources, extra_cuda_cflags=["-O2"] if opt else [], 
                verbose=verbose, name="ext", build_directory=build_directory)