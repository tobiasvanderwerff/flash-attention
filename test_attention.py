import os
import time
from pathlib import Path
import torch

import torch
import pytest

from util import load_cuda, load_cuda_inline, set_env_vars, load_and_compile_sources

set_env_vars()
module = load_and_compile_sources(Path("csrc"))

@pytest.mark.parametrize(
    "h,k,w", [(1024, 1024, 1024), (1024, 1000, 1024), (1000, 10, 1000), (999, 999, 999)]
)
def test_matmul_kernel(h, w, k):
    torch.manual_seed(1)
    m1 = torch.randn(h, k, device="cuda")
    m2 = torch.randn(k, w, device="cuda")

    out = module.matmul(m1, m2)
    out_pt = torch.matmul(m1, m2)
    assert torch.isclose(out, out_pt, atol=1e-4).all().item()


@pytest.mark.parametrize(
    "h,w", [(256, 256)]
    # "h,w", [(128, 128), (99, 99), (100, 2048), (100, 2047)]
)
def test_softmax_kernel(h, w):
    torch.manual_seed(1)
    x = torch.randn(h, w, device="cuda")
    # out = _softmax(x)
    out = module.softmax(x)
    out_pt = torch.nn.functional.softmax(x, dim=1)

    assert (out >= 0).all().item()
    assert (out <= 1).all().item()
    assert out.sum(dim=1).allclose(torch.ones(h, device="cuda"))
    assert torch.isclose(out, out_pt, atol=1e-4).all().item()


def _softmax(inp):
    inp_exp = torch.exp(inp)
    inp_exp_sum = inp_exp.sum(dim=1, keepdim=True)
    out = inp_exp / inp_exp_sum
    return out