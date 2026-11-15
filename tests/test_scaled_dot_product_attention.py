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

from contextlib import nullcontext

import numpy as np
import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import flaggems_vllm
from flaggems_vllm.ops.attention import scaled_dot_product_attention
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import random_utils

from . import accuracy_utils as utils
from . import conftest as cfg
from .conftest import QUICK_MODE

device = flaggems_vllm.device
vendor_name = flaggems_vllm.vendor_name

if QUICK_MODE:
    LEGACY_SHAPES = [
        (4, 8, 8, 1024, 1024, 64, False),
    ]
    CAUSAL_CHOICES = [False]
    FLOAT_DTYPES = [torch.float16]
else:
    LEGACY_SHAPES = [
        (4, 8, 8, 1024, 1024, 64, False),
        (4, 8, 8, 1024, 1024, 128, False),
        (4, 8, 8, 2048, 256, 64, False),
        (4, 8, 8, 2048, 256, 128, False),
        (4, 8, 8, 17, 1030, 64, False),
        (4, 8, 8, 17, 1030, 128, False),
        # adopted from FlagAttention `test_attention_fwd`:
        (2, 4, 4, 512, 612, 128, False),
        (2, 4, 4, 1024, 1034, 64, False),
        (2, 4, 4, 2048, 2048, 32, False),
        (2, 4, 4, 4096, 4096, 16, False),
        (2, 4, 4, 4001, 4001, 32, False),
        (2, 4, 4, 4001, 4096, 64, False),
        (2, 4, 4, 4096, 4000, 128, False),
        (1, 2, 2, 8192, 8202, 16, False),
        (1, 2, 2, 8192, 8192, 32, False),
        # test for mqa/gqa
        (2, 4, 2, 512, 612, 128, True),
        (2, 4, 1, 1024, 1034, 64, True),
        (2, 4, 2, 2048, 2048, 32, True),
        (2, 4, 1, 4096, 4096, 16, True),
        (2, 4, 2, 4001, 4001, 32, True),
        (2, 4, 1, 4001, 4096, 64, True),
        (2, 4, 2, 4096, 4000, 128, True),
        (1, 2, 1, 8192, 8202, 16, True),
        (1, 2, 1, 8192, 8192, 32, True),
    ]
    CAUSAL_CHOICES = [False, True]
    FLOAT_DTYPES = [torch.float16, torch.bfloat16]


def make_input(
    batch,
    num_head,
    num_head_k,
    q_seq_len,
    kv_seq_len,
    head_size,
    dtype,
    device,
    requires_grad=False,
):
    random_utils.set_philox_state(
        42 if vendor_name == "cambricon" else 1234567890, 0, device
    )
    q_shape = (batch, num_head, q_seq_len, head_size)
    kv_shape = (batch, num_head_k, kv_seq_len, head_size)
    q = torch.empty(q_shape, dtype=dtype, device=device).uniform_(-0.05, 0.05)
    k = torch.empty(kv_shape, dtype=dtype, device=device).uniform_(-0.05, 0.05)
    v = torch.empty(kv_shape, dtype=dtype, device=device).uniform_(-0.05, 0.05)
    if requires_grad:
        q.requires_grad_()
        k.requires_grad_()
        v.requires_grad_()
    return q, k, v


def torch_sdpa(q, k, v, scale, is_causal, enable_gqa=False):
    if torch.__version__ < "2.5":
        return torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            scale=scale,
            is_causal=is_causal,
        )

    if vendor_name in ["cambricon", "iluvatar"] and cfg.TO_CPU:
        ctx = sdpa_kernel(backends=[SDPBackend.MATH])
    else:
        ctx = nullcontext()

    with ctx:
        torch_result = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            scale=scale,
            is_causal=is_causal,
            enable_gqa=enable_gqa,
        )
    return torch_result


@pytest.mark.skipif(
    torch.__version__ < "2.5", reason="Low Pytorch Version: enable_gqa not supported"
)
@pytest.mark.scaled_dot_product_attention_forward
@pytest.mark.parametrize(
    "batch, num_q_head, num_kv_head, q_seq_len, kv_seq_len, head_size, enable_gqa",
    LEGACY_SHAPES,
)
@pytest.mark.parametrize("is_causal", CAUSAL_CHOICES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_scaled_dot_product_attention_legacy(
    monkeypatch,
    batch,
    num_q_head,
    num_kv_head,
    q_seq_len,
    kv_seq_len,
    head_size,
    is_causal,
    dtype,
    enable_gqa,
):
    if vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    device = torch_device_fn.current_device()
    q, k, v = make_input(
        batch,
        num_q_head,
        num_kv_head,
        q_seq_len,
        kv_seq_len,
        head_size,
        dtype,
        device,
        requires_grad=True,
    )
    ref_q = utils.to_reference(q, False)
    ref_k = utils.to_reference(k, False)
    ref_v = utils.to_reference(v, False)
    scale = float(1.0 / np.sqrt(head_size))

    # forward
    torch_result = torch_sdpa(
        ref_q, ref_k, ref_v, scale, is_causal, enable_gqa=enable_gqa
    )

    gems_result = scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        scale=scale,
        is_causal=is_causal,
        enable_gqa=enable_gqa,
    )

    utils.gems_assert_close(gems_result, torch_result, dtype)


@pytest.mark.skipif(vendor_name == "metax", reason="Issue #2849: Not working")
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="Issue #2849: Not working")
@pytest.mark.skipif(vendor_name == "sunrise", reason="Compiler Error")
@pytest.mark.skipif(
    torch.__version__ < "2.5", reason="Low Pytorch Version: enable_gqa not supported"
)
@pytest.mark.scaled_dot_product_attention_backward
@pytest.mark.parametrize(
    "batch, num_q_head, num_kv_head, q_seq_len, kv_seq_len, head_size, enable_gqa",
    LEGACY_SHAPES,
)
@pytest.mark.parametrize("is_causal", CAUSAL_CHOICES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_scaled_dot_product_attention_legacy_backward(
    batch,
    num_q_head,
    num_kv_head,
    q_seq_len,
    kv_seq_len,
    head_size,
    is_causal,
    dtype,
    enable_gqa,
):
    device = torch_device_fn.current_device()
    q, k, v = make_input(
        batch,
        num_q_head,
        num_kv_head,
        q_seq_len,
        kv_seq_len,
        head_size,
        dtype,
        device,
        requires_grad=True,
    )
    ref_q = utils.to_reference(q, False).detach().requires_grad_(True)
    ref_k = utils.to_reference(k, False).detach().requires_grad_(True)
    ref_v = utils.to_reference(v, False).detach().requires_grad_(True)
    scale = float(1.0 / np.sqrt(head_size))

    # forward
    torch_result = torch_sdpa(
        ref_q, ref_k, ref_v, scale, is_causal, enable_gqa=enable_gqa
    )

    gems_result = scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        scale=scale,
        is_causal=is_causal,
        enable_gqa=enable_gqa,
    )

    utils.gems_assert_close(gems_result, torch_result, dtype)

    if vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)

    # backward
    ref_dout = torch.randn_like(ref_q)
    torch_result.backward(ref_dout)
    gems_result.backward(ref_dout.to(gems_result.device))
    torch_q_grad = ref_q.grad.clone() if ref_q.grad is not None else None
    torch_k_grad = ref_k.grad.clone() if ref_k.grad is not None else None
    torch_v_grad = ref_v.grad.clone() if ref_v.grad is not None else None
    gems_q_grad = q.grad.clone() if q.grad is not None else None
    gems_k_grad = k.grad.clone() if k.grad is not None else None
    gems_v_grad = v.grad.clone() if v.grad is not None else None

    # NOTE: NaN may arise in the gradients, this behavior aligns with PyTorch's SDPA
    utils.gems_assert_close(gems_q_grad, torch_q_grad, dtype, equal_nan=True)
    utils.gems_assert_close(gems_k_grad, torch_k_grad, dtype, equal_nan=True)

    # dV is more sensitive to softmax recomputation errors in flash attention backward
    # because it lacks the centering term (dP - D) that suppresses errors in dK/dQ.
    # The backward recomputes the softmax from the saved output rather than reusing
    # the forward's probabilities, so the error grows with the sequence length:
    # every extra KV position adds a rounding step to the recomputed row sum. The
    # tiers below are sized for the longest shapes (up to 8192), where the
    # short-sequence values are exceeded by roughly an order of magnitude on a
    # handful of elements out of millions.
    # GQA: different float accumulation order across Q heads vs PyTorch kernel
    # bf16: only 8 mantissa bits → largest recomputation error
    # fp16: 11 mantissa bits → moderate error
    is_gqa = enable_gqa and num_q_head != num_kv_head
    if is_gqa:
        if dtype == torch.bfloat16:
            v_atol = 2e-1
        elif dtype == torch.float16:
            v_atol = 4e-2
        else:
            v_atol = 5e-3
    else:
        if dtype == torch.bfloat16:
            v_atol = 1e-1 if vendor_name == "hygon" else 5e-2
        elif dtype == torch.float16:
            v_atol = 2e-2
        else:
            v_atol = 3e-3
    utils.gems_assert_close(
        gems_v_grad, torch_v_grad, dtype, equal_nan=True, atol=v_atol
    )


def _mask_broadcast_reference(q, k, v, bias, enable_gqa=False):
    with sdpa_kernel(SDPBackend.MATH):
        return torch.nn.functional.scaled_dot_product_attention(
            q.float(),
            k.float(),
            v.float(),
            attn_mask=bias.float(),
            enable_gqa=enable_gqa,
        )


def _mask_broadcast_inputs(dtype, mask_kind, strided):
    torch.manual_seed(42)
    b, h, sq, sk, d = 2, 4, 65, 97, 72

    def make_qkv(seq):
        # Model vision projections are sliced and transposed, not BHSD-contiguous.
        x = torch.randn(b, seq, h, 3 * d, device=device, dtype=dtype)
        x = x[..., :d].transpose(1, 2)
        return x if strided else x.contiguous()

    q, k, v = make_qkv(sq), make_qkv(sk), make_qkv(sk)
    shapes = {
        "padding": (b, 1, 1, sk),
        "batch": (1, h, sq, sk),
        "matrix": (sq, sk),
        "expanded": (b, 1, 1, sk),
    }
    # A sliced mask also checks that the non-broadcast strides are preserved.
    shape = shapes[mask_kind]
    bias = torch.zeros(*shape[:-1], 2 * sk, device=device, dtype=dtype)
    bias = bias[..., ::2]
    bias[..., sk // 2 :] = torch.finfo(dtype).min
    if mask_kind == "expanded":
        bias = bias.expand(b, h, sq, sk)
    return q, k, v, bias


# The kernel indexes the full [B, H_q, S_q, S_k] attention matrix, so compact
# masks must be addressed with broadcast (zero) strides (FlagGems issue #6869).
@pytest.mark.skipif(cfg.TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.scaled_dot_product_attention
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("mask_kind", ["padding", "batch", "matrix", "expanded"])
def test_scaled_dot_product_attention_mask_broadcast(dtype, strided, mask_kind):
    q, k, v, bias = _mask_broadcast_inputs(dtype, mask_kind, strided)
    expected = _mask_broadcast_reference(q, k, v, bias)
    actual = scaled_dot_product_attention(q, k, v, attn_mask=bias)
    torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)


@pytest.mark.skipif(cfg.TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(device != "cuda", reason="Requires CUDA-compatible graphs")
@pytest.mark.scaled_dot_product_attention
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scaled_dot_product_attention_mask_broadcast_graph(dtype):
    q, k, v, bias = _mask_broadcast_inputs(dtype, "padding", True)
    scaled_dot_product_attention(q, k, v, attn_mask=bias)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = scaled_dot_product_attention(q, k, v, attn_mask=bias)

    for seed in (43, 44):
        torch.manual_seed(seed)
        q.normal_()
        k.normal_()
        v.normal_()
        # Change mask data as well as Q/K/V between replays.
        bias.zero_()
        bias[..., (seed % 3 + 1) * 20 :] = torch.finfo(dtype).min
        graph.replay()
        torch.cuda.synchronize()
        expected = _mask_broadcast_reference(q, k, v, bias)
        torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)


@pytest.mark.skipif(cfg.TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.scaled_dot_product_attention
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scaled_dot_product_attention_mask_broadcast_gqa(dtype):
    q, k, v, bias = _mask_broadcast_inputs(dtype, "padding", True)
    k, v = k[:, :2], v[:, :2]
    expected = _mask_broadcast_reference(q, k, v, bias, enable_gqa=True)
    actual = scaled_dot_product_attention(q, k, v, attn_mask=bias, enable_gqa=True)
    torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)
