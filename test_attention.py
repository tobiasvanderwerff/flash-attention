from pathlib import Path
import torch

import torch
import pytest

from util import set_env_vars, load_and_compile_sources

set_env_vars()
module = load_and_compile_sources(Path("csrc"))

@pytest.mark.parametrize(
    "h,k,w", [(2, 2, 20), (20, 2, 2), (3, 7, 11), (1024, 1024, 1024), 
              (1000, 10, 10), (10, 10, 1000), (999, 999, 999)]
)
def test_matmul_kernel(h, w, k):
    torch.manual_seed(1)
    m1 = torch.randn(h, k, device="cuda")
    m2 = torch.randn(k, w, device="cuda")

    out = module.my_matmul(m1, m2)
    out_pt = torch.matmul(m1, m2)
    assert torch.isclose(out, out_pt, atol=1e-4).all().item()


@pytest.mark.parametrize(
    "h,w", [(256, 256), (99, 99), (100, 2048), (100, 2047), (1024, 1024)]
)
def test_softmax_kernel(h, w):
    torch.manual_seed(1)
    x = torch.randn(h, w, device="cuda")
    # out = _softmax(x)
    out = module.my_softmax(x)
    out_pt = torch.nn.functional.softmax(x, dim=1)

    assert (out >= 0).all().item()
    assert (out <= 1).all().item()
    assert out.sum(dim=1).allclose(torch.ones(h, device="cuda"))
    assert torch.isclose(out, out_pt, atol=1e-4).all().item()


@pytest.mark.parametrize(
    "seq_len,d", [(2, 2), (1, 1), (2, 20), (2, 99), (20, 2), (99, 20), (1024, 64), (1023, 63), (1025, 65)]
)
def test_attention_kernel(seq_len, d):
    torch.manual_seed(1)
    Q = torch.randn(seq_len, d, device="cuda", dtype=torch.float32)
    K = torch.randn(seq_len, d, device="cuda", dtype=torch.float32)
    V = torch.randn(seq_len, d, device="cuda", dtype=torch.float32)

    out = module.my_attention(Q, K.transpose(0, 1).contiguous(), V)
    out_pt = _attention(Q, K, V)
    # out_pt = torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=1.0)

    assert out.shape == out_pt.shape
    assert ((out - out_pt).abs() < 1e-4).all().item()
    # assert torch.isclose(out, out_pt, atol=1e-4).all().item()


def _softmax(inp):
    inp_exp = torch.exp(inp - inp.max(dim=1, keepdim=True)[0])
    inp_exp_sum = inp_exp.sum(dim=1, keepdim=True)
    out = inp_exp / inp_exp_sum
    return out

def _attention(Q, K, V):
    assert Q.ndim == K.ndim == V.ndim == 2
    assert Q.shape == K.shape == V.shape
    QK = Q @ K.transpose(0, 1)  # (N, N)
    # weights = torch.nn.functional.softmax(QK, dim=-1)
    weights = _softmax(QK)
    out = weights @ V  # (N,N) x (N,D) = (N,D)
    return out