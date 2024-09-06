from math import ceil, inf
from pathlib import Path
import torch
import pytest

# from my_flash_attn.util import set_env_vars, load_and_compile_sources

# We need to import the CUDA kernels after importing torch
import my_flash_attn_cuda

ABS_TOL = 1e-4
REL_TOL = 1e-1

def print_diffs(out, out_ref):
    out_1d = out.flatten()
    out_ref_1d = out_ref.flatten()
    for idx, (e_o, e_o_ref) in enumerate(zip(out_1d, out_ref_1d)):
        diff = e_o - e_o_ref
        abs_diff = abs(diff)
        abs_ref = abs(e_o_ref + 1e-5)
        relative_diff = abs_diff / abs_ref
         
        if abs_diff > ABS_TOL or relative_diff > REL_TOL:
            row = idx // out.shape[1]
            col = idx % out.shape[1]
            print(f"==== diff ==== [{row}][{col}], test: {e_o}, ref: {e_o_ref}")
        # if abs_diff > ABS_TOL or relative_diff > REL_TOL:
        #     print(f"==== diff ==== {idx}, test: {e_o}, ref: {e_o_ref}")


def _softmax(inp):
    inp_exp = torch.exp(inp - inp.max(dim=1, keepdim=True)[0])
    inp_exp_sum = inp_exp.sum(dim=1, keepdim=True)
    out = inp_exp / inp_exp_sum
    return out

def _attention_forward(Q, K, V):
    assert Q.ndim == K.ndim == V.ndim == 2
    assert Q.shape == K.shape == V.shape
    QK = Q @ K.transpose(0, 1)  # (N, N)
    # weights = torch.nn.functional.softmax(QK, dim=-1)
    weights = _softmax(QK)
    out = weights @ V  # (N,N) x (N,D) = (N,D)
    return out

def _flash_attention_forward(Q, K, V):
    assert Q.ndim == K.ndim == V.ndim == 2
    assert Q.shape == K.shape == V.shape

    n = Q.shape[0]
    d = Q.shape[1]
    
    sram_size = 64 * 1024  # 64kb (dummy)
    blk_sz_c = ceil(sram_size / (4*d))
    blk_sz_r = min(blk_sz_c, d)

    out = torch.zeros((n, d), device=Q.device)
    l = torch.zeros((n,), device=Q.device)
    m = torch.full((n,), -inf, device=Q.device)

    t_c = ceil(n / blk_sz_c)
    t_r = ceil(n / blk_sz_r)

    for j in range(t_c):
        # L. 6 
        # Load from HBM to SRAM
        K_j = K[j*blk_sz_c:(j+1)*blk_sz_c, :]
        V_j = V[j*blk_sz_c:(j+1)*blk_sz_c, :]

        for i in range(t_r):
            # L. 8 
            # Load from HBM to SRAM
            Q_i = Q[i*blk_sz_r:(i+1)*blk_sz_r, :]
            O_i = out[i*blk_sz_r:(i+1)*blk_sz_r, :]
            l_i = l[i*blk_sz_r:(i+1)*blk_sz_r]
            m_i = m[i*blk_sz_r:(i+1)*blk_sz_r]

            # L. 9
            QK_ij = Q_i @ K_j.T  # (block_size_r, block_size_c)

            # L. 10
            m_ij = QK_ij.max(dim=1)[0]  # (block_size_r,)
            P_ij = torch.exp(QK_ij - m_ij[:, None])  # (block_size_r, block_size_c)
            l_ij = P_ij.sum(dim=1)  # (block_size_r,)

            # L. 11
            m_i_new = torch.max(m_i, m_ij)
            l_i_new = torch.exp(m_i - m_i_new) * l_i + torch.exp(m_ij - m_i_new) * l_ij

            # L. 12
            l_i_new_inv = torch.inverse(torch.diag_embed(l_i_new))
            O_i =  l_i_new_inv @ (torch.diag_embed(l_i * torch.exp(m_i - m_i_new)) @ O_i)
            O_i += l_i_new_inv @ (torch.diag_embed(torch.exp(m_ij - m_i_new)) @ P_ij @ V_j)

            # L. 13
            # Write to HBM
            out[i*blk_sz_r:(i+1)*blk_sz_r, :] = O_i
            m[i*blk_sz_r:(i+1)*blk_sz_r] = m_i_new
            l[i*blk_sz_r:(i+1)*blk_sz_r] = l_i_new

    return out

# set_env_vars()
# my_flash_attn_cuda = load_and_compile_sources(Path("csrc"))

@pytest.mark.parametrize(
    "h,k,w", [(2, 2, 20), (20, 2, 2), (3, 7, 11), (1024, 1024, 1024), 
              (1000, 10, 10), (10, 10, 1000), (999, 999, 999)]
)
def test_matmul_kernel(h, w, k):
    torch.manual_seed(1)
    m1 = torch.randn(h, k, device="cuda")
    m2 = torch.randn(k, w, device="cuda")

    out = my_flash_attn_cuda.my_matmul(m1, m2)
    out_pt = torch.matmul(m1, m2)
    assert torch.isclose(out, out_pt, atol=1e-4).all().item()

@pytest.mark.parametrize(
    "h,k,w", [(2, 2, 20), (20, 2, 2), (3, 7, 11), (1024, 1024, 1024), 
              (1000, 10, 10), (10, 10, 1000), (999, 999, 999)]
)
def test_matmul_cublas_kernel(h, w, k):
    torch.manual_seed(1)
    m1 = torch.randn(h, k, device="cuda")
    m2 = torch.randn(k, w, device="cuda")

    out = my_flash_attn_cuda.my_matmul_cublas(m1, m2)
    out_pt = torch.matmul(m1, m2)
    assert torch.isclose(out, out_pt, atol=1e-4).all().item()

@pytest.mark.parametrize(
    "h,w", [(255, 516)]
    # "h,w", [(256, 256), (99, 99), (100, 2048), (100, 2047), (1024, 1024)]
    # "h,w", [(256, 256), (99, 99), (100, 2048), (100, 2047), (1024, 1024), (4096, 4096)]
)
def test_softmax_kernel(h, w):
    torch.manual_seed(1)
    x = torch.randn(h, w, device="cuda", dtype=torch.float32)
    # out = _softmax(x)
    # print(x.max(dim=1)[0][0])
    out = my_flash_attn_cuda.my_softmax(x, 5)
    out_pt = torch.nn.functional.softmax(x, dim=1)

    assert (out >= 0).all().item()
    assert (out <= 1).all().item()
    # print_diffs(out, out_pt)
    s = out.sum(dim=1)
    assert s.allclose(torch.ones(h, dtype=torch.float32, device="cuda"))
    # assert out.sum(dim=1).allclose(torch.ones(h, dtype=torch.float32, device="cuda"))
    assert torch.isclose(out, out_pt, atol=1e-4).all().item()


@pytest.mark.parametrize(
    "seq_len,d", [(2, 2), (1, 1), (2, 20), (2, 99), (20, 2), (99, 20), (1024, 64), (1023, 63), (1025, 65)]
)
def test_attention_kernel(seq_len, d):
    torch.manual_seed(1)
    Q = torch.randn(seq_len, d, device="cuda", dtype=torch.float32)
    K = torch.randn(seq_len, d, device="cuda", dtype=torch.float32)
    V = torch.randn(seq_len, d, device="cuda", dtype=torch.float32)

    out = my_flash_attn_cuda.my_attention(Q, K.transpose(0, 1).contiguous(), V)
    # out_pt = _attention_forward(Q, K, V)
    out_pt = torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=1.0)

    assert out.shape == out_pt.shape
    assert ((out - out_pt).abs() < 1e-4).all().item()
    # assert torch.isclose(out, out_pt, atol=1e-4).all().item()


@pytest.mark.parametrize(
    "seq_len,d", [(2, 2), (1, 1), (2, 20), (2, 99), (20, 2), (99, 20), (1024, 64), (1023, 63), (1025, 65)]
)
def test_flash_attention_kernel(seq_len, d):
    torch.manual_seed(1)
    Q = torch.randn(seq_len, d, device="cuda", dtype=torch.float32)
    K = torch.randn(seq_len, d, device="cuda", dtype=torch.float32)
    V = torch.randn(seq_len, d, device="cuda", dtype=torch.float32)

    # out = my_flash_attn_cuda.my_attention(Q, K.transpose(0, 1).contiguous(), V)
    out = _flash_attention_forward(Q, K, V)
    out_pt = torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=1.0)

    assert out.shape == out_pt.shape
    assert ((out - out_pt).abs() < 1e-4).all().item()
