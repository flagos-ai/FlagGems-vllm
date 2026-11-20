# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from itertools import product

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.marlin_gemm import UINT4B8_ID, marlin_gemm

from . import accuracy_utils as utils
from . import conftest as cfg

GROUP_SIZE = 128


def _pack_marlin_int4(codes, out_dim, in_dim):
    """Scatter uint4b8 codes into the Marlin 16x64-unit int32 layout.

    Inverts the kernel addressing: unit (kb, pid_n) holds 128 words; word
    waddr = 4*((n%8)*4 + (k%8)//2) + n//16, nibble J = 4*(k%2)+2*(k//8)+(n//8)%2.
    Output shape (K//16, N*2), linear layout equals (K//16)*(N//64) units.
    """
    words = torch.zeros(
        (in_dim // 16) * (out_dim // 64) * 128, dtype=torch.int32, device=codes.device
    )
    kk = torch.arange(16, device=codes.device)
    n_loc = torch.arange(64, device=codes.device)
    waddr = 4 * (((n_loc % 8) * 4).unsqueeze(0) + ((kk % 8) // 2).unsqueeze(1)) + (
        n_loc // 16
    ).unsqueeze(0)
    J = (4 * (kk % 2) + 2 * (kk // 8)).unsqueeze(1) + ((n_loc // 8) % 2).unsqueeze(0)
    for kb in range(in_dim // 16):
        for pid_n in range(out_dim // 64):
            base = (kb * (out_dim // 64) + pid_n) * 128
            tile = codes[pid_n * 64 : (pid_n + 1) * 64, kb * 16 : (kb + 1) * 16].T
            words.index_put_(
                (base + waddr.reshape(-1).long(),),
                (tile << (4 * J)).reshape(-1).to(torch.int32),
                accumulate=True,
            )
    return words.view(in_dim // 16, out_dim * 2)


def _marlin_permute_scales(s_kn, out_dim):
    """Apply vLLM get_scale_perm to (K//gs, N) scales: 64-column chunks."""
    perm = torch.tensor(
        [i + 8 * j for i in range(8) for j in range(8)], device=s_kn.device
    )
    return s_kn.reshape(-1, 64)[:, perm].reshape(-1, out_dim).contiguous()


def _quantize_uint4b8(w, group_size=GROUP_SIZE):
    """Symmetric per-group uint4b8 quantization (torch, test-only).

    Returns (w_ref, q_marlin, scales): dequantized reference, Marlin-repacked
    int32 (K//16, N*2) weights and marlin-permuted (K//gs, N) scales.
    """
    out_dim, in_dim = w.shape
    grouped = w.reshape(out_dim, in_dim // group_size, group_size).float()
    scales = (grouped.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 7.0).squeeze(-1)
    q = torch.round(grouped / scales.unsqueeze(-1)).clamp(-7, 7) + 8
    w_ref = ((q - 8) * scales.unsqueeze(-1)).to(w.dtype).reshape(out_dim, in_dim)
    q_marlin = _pack_marlin_int4(
        q.to(torch.int32).reshape(out_dim, in_dim), out_dim, in_dim
    )
    scales_kn = scales.t().to(w.dtype).contiguous()
    scales_perm = _marlin_permute_scales(scales_kn, out_dim)
    return w_ref.contiguous(), q_marlin.contiguous(), scales_perm


def _make_inputs(size_m, size_n, size_k, dtype, device, with_bias=False):
    w = torch.randn((size_n, size_k), dtype=dtype, device=device) * 0.1
    w_ref, b_q_weight, b_scales = _quantize_uint4b8(w)
    a = torch.randn((size_m, size_k), dtype=dtype, device=device)
    b_bias = torch.randn((size_n,), dtype=dtype, device=device) if with_bias else None
    ref = a.float() @ w_ref.float().T
    if with_bias:
        ref += b_bias.float()
    ref = ref.to(dtype)
    return a, b_q_weight, b_scales, b_bias, ref


def _run_marlin_gemm(
    a,
    b_q_weight,
    b_scales,
    size_m,
    size_n,
    size_k,
    b_bias=None,
    c=None,
    g_idx=None,
    b_q_type=UINT4B8_ID,
):
    workspace = torch.empty(0, dtype=a.dtype, device=a.device)
    return marlin_gemm(
        a=a,
        c=c,
        b_q_weight=b_q_weight,
        b_bias=b_bias,
        b_scales=b_scales,
        a_scales=None,
        global_scale=None,
        b_zeros=None,
        g_idx=g_idx,
        perm=None,
        workspace=workspace,
        b_q_type=b_q_type,
        size_m=size_m,
        size_n=size_n,
        size_k=size_k,
    )


# Batch size
_marlin_gemm_ms = [1, 16, 48, 128]
_marlin_gemm_ns = [256, 512]
_marlin_gemm_ks = [256, 512]
_marlin_gemm_shapes = list(product(_marlin_gemm_ms, _marlin_gemm_ns, _marlin_gemm_ks))
if cfg.QUICK_MODE:
    _marlin_gemm_shapes = random.sample(_marlin_gemm_shapes, 8)


@pytest.mark.marlin_gemm
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("shape", _marlin_gemm_shapes)
@pytest.mark.parametrize("dtype,atol", [(torch.float16, 5e-3), (torch.bfloat16, 5e-2)])
def test_marlin_gemm(shape, dtype, atol):
    size_m, size_n, size_k = shape
    device = flaggems_vllm.device
    a, b_q_weight, b_scales, _, ref = _make_inputs(
        size_m, size_n, size_k, dtype, device
    )

    out = _run_marlin_gemm(a, b_q_weight, b_scales, size_m, size_n, size_k)

    assert out.shape == (size_m, size_n)
    utils.gems_assert_close(out, ref, dtype, atol=atol)


@pytest.mark.marlin_gemm
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_marlin_gemm_with_bias():
    device = flaggems_vllm.device
    a, b_q_weight, b_scales, b_bias, ref = _make_inputs(
        32, 256, 256, torch.float16, device, with_bias=True
    )

    out = _run_marlin_gemm(a, b_q_weight, b_scales, 32, 256, 256, b_bias=b_bias)

    utils.gems_assert_close(out, ref, torch.float16, atol=5e-3)


@pytest.mark.marlin_gemm
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_marlin_gemm_out_buffer():
    device = flaggems_vllm.device
    a, b_q_weight, b_scales, _, ref = _make_inputs(16, 256, 256, torch.float16, device)
    c = torch.empty((16, 256), dtype=torch.float16, device=device)

    out = _run_marlin_gemm(a, b_q_weight, b_scales, 16, 256, 256, c=c)

    assert out is c
    utils.gems_assert_close(out, ref, torch.float16, atol=5e-3)


@pytest.mark.marlin_gemm
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_marlin_gemm_unsupported_quant_type():
    device = flaggems_vllm.device
    a, b_q_weight, b_scales, _, _ = _make_inputs(16, 256, 256, torch.float16, device)
    with pytest.raises(NotImplementedError):
        _run_marlin_gemm(a, b_q_weight, b_scales, 16, 256, 256, b_q_type=UINT4B8_ID + 1)


@pytest.mark.marlin_gemm
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_marlin_gemm_unsupported_act_order():
    device = flaggems_vllm.device
    a, b_q_weight, b_scales, _, _ = _make_inputs(16, 256, 256, torch.float16, device)
    g_idx = torch.zeros(256, dtype=torch.int32, device=device)
    with pytest.raises(NotImplementedError):
        _run_marlin_gemm(a, b_q_weight, b_scales, 16, 256, 256, g_idx=g_idx)
