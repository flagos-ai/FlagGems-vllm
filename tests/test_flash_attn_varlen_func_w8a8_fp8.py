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

import inspect
import math

import pytest
import torch
import triton

import flaggems_vllm
from flaggems_vllm.ops import flash_attn_varlen_func_w8a8_fp8 as w8a8_varlen
from flaggems_vllm.ops.attention import flash_attn_varlen_func as fa2_varlen

from . import accuracy_utils as utils
from . import conftest as cfg

device = flaggems_vllm.device
vendor_name = flaggems_vllm.vendor_name
W8A8_HEAD_SIZES = tuple(range(8, 257, 8))
W8A8_FP8_DTYPES = [
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
    )
    if dtype is not None
]

if cfg.QUICK_MODE:
    W8A8_CONFIGS = [(1, 16, 512, 512)]
else:
    W8A8_CONFIGS = [
        (1, 16, 17, 1030),
        (1, 16, 512, 512),
        (2, 16, 1024, 1024),
        (4, 16, 2048, 2048),
        (8, 32, 512, 512),
    ]


def _supports_hopper_fp8() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


def _get_fp8_dtype():
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        pytest.skip("torch.float8_e4m3fn is not available")
    return dtype


def _hadamard_matrix(dim, tensor_device):
    assert dim > 0, "head_size must be positive"
    # Use orthogonal blocks so non-power-of-two dimensions need no padding.
    blocks = []
    remaining = dim
    while remaining:
        block_dim = 1 << (remaining.bit_length() - 1)
        matrix = torch.tensor([[1.0]], device=tensor_device)
        while matrix.shape[0] < block_dim:
            matrix = torch.cat(
                (
                    torch.cat((matrix, matrix), dim=1),
                    torch.cat((matrix, -matrix), dim=1),
                ),
                dim=0,
            )
        blocks.append(matrix / math.sqrt(block_dim))
        remaining -= block_dim
    return torch.block_diag(*blocks)


def _apply_incoherent_qk(x):
    matrix = _hadamard_matrix(x.shape[-1], x.device).to(torch.float32)
    return torch.matmul(x.float(), matrix).to(x.dtype)


def _cu_seqlens_from_lengths(lengths, tensor_device):
    lengths_tensor = torch.tensor(lengths, dtype=torch.int32, device=tensor_device)
    return torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=tensor_device),
            lengths_tensor.cumsum(0, dtype=torch.int32),
        )
    )


def _quantize_varlen_per_block_fp8(x, lengths, fp8_dtype, block_size=128):
    num_heads = x.shape[1]
    fp8_max = float(torch.finfo(fp8_dtype).max)
    num_blocks = triton.cdiv(max(lengths), block_size)
    quantized = torch.empty_like(x, dtype=fp8_dtype)
    descale = torch.ones(
        (len(lengths), num_heads, num_blocks),
        device=x.device,
        dtype=torch.float32,
    )

    token_offset = 0
    for batch_idx, seq_len in enumerate(lengths):
        for block_idx in range(triton.cdiv(seq_len, block_size)):
            lo = token_offset + block_idx * block_size
            hi = min(token_offset + seq_len, lo + block_size)
            tile = x[lo:hi].float()
            scale = (tile.abs().amax(dim=(0, 2)) / fp8_max).clamp_min(
                torch.finfo(torch.float32).tiny
            )
            quantized[lo:hi] = torch.clamp(
                tile / scale[None, :, None], -fp8_max, fp8_max
            ).to(fp8_dtype)
            descale[batch_idx, :, block_idx] = scale
        token_offset += seq_len

    return quantized.contiguous(), descale.contiguous()


def _dequantize_varlen_per_block_fp8(x, lengths, descale, dtype, block_size=128):
    dequantized = torch.empty_like(x, dtype=dtype)
    token_offset = 0
    for batch_idx, seq_len in enumerate(lengths):
        for block_idx in range(triton.cdiv(seq_len, block_size)):
            lo = token_offset + block_idx * block_size
            hi = min(token_offset + seq_len, lo + block_size)
            dequantized[lo:hi] = (
                x[lo:hi].float() * descale[batch_idx, :, block_idx][None, :, None]
            ).to(dtype)
        token_offset += seq_len
    return dequantized.contiguous()


def _quantize_qkv_w8a8(q, k, v, q_lengths, kv_lengths, fp8_dtype=None):
    fp8_dtype = _get_fp8_dtype() if fp8_dtype is None else fp8_dtype
    q_fp8, q_descale = _quantize_varlen_per_block_fp8(
        _apply_incoherent_qk(q), q_lengths, fp8_dtype
    )
    k_fp8, k_descale = _quantize_varlen_per_block_fp8(
        _apply_incoherent_qk(k), kv_lengths, fp8_dtype
    )
    v_fp8, v_descale = _quantize_varlen_per_block_fp8(v, kv_lengths, fp8_dtype)
    return q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale


def _make_packed_inputs(q_lengths, kv_lengths, num_heads, head_size, dtype):
    q = torch.empty(
        (sum(q_lengths), num_heads, head_size), device=device, dtype=dtype
    ).uniform_(-0.05, 0.05)
    k = torch.empty(
        (sum(kv_lengths), num_heads, head_size), device=device, dtype=dtype
    ).uniform_(-0.05, 0.05)
    v = torch.empty_like(k).uniform_(-0.05, 0.05)
    return q, k, v


def _with_padded_strides(x, copy_data=True):
    storage = torch.empty(
        (x.shape[0] * 2, x.shape[1] * 2, x.shape[2]),
        dtype=x.dtype,
        device=x.device,
    )
    result = storage[::2, ::2, :]
    if copy_data:
        result.copy_(x)
    return result


def _run_w8a8_varlen(
    q,
    k,
    v,
    q_lengths,
    kv_lengths,
    scale,
    causal,
    return_softmax_lse=False,
    fp8_dtype=None,
    strided=False,
):
    (
        q_fp8,
        k_fp8,
        v_fp8,
        q_descale,
        k_descale,
        v_descale,
    ) = _quantize_qkv_w8a8(q, k, v, q_lengths, kv_lengths, fp8_dtype=fp8_dtype)
    cu_seqlens_q = _cu_seqlens_from_lengths(q_lengths, q.device)
    cu_seqlens_k = _cu_seqlens_from_lengths(kv_lengths, q.device)
    out = torch.empty_like(q)
    if strided:
        q_fp8, k_fp8, v_fp8 = map(_with_padded_strides, (q_fp8, k_fp8, v_fp8))
        out = _with_padded_strides(out, copy_data=False)
    result = w8a8_varlen(
        q_fp8,
        k_fp8,
        v_fp8,
        max(q_lengths),
        cu_seqlens_q,
        max(kv_lengths),
        cu_seqlens_k,
        softmax_scale=scale,
        causal=causal,
        out=out,
        return_softmax_lse=return_softmax_lse,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    assert (result[0] if return_softmax_lse else result) is out
    reference_inputs = (
        _dequantize_varlen_per_block_fp8(q_fp8, q_lengths, q_descale, torch.bfloat16),
        _dequantize_varlen_per_block_fp8(k_fp8, kv_lengths, k_descale, torch.bfloat16),
        _dequantize_varlen_per_block_fp8(v_fp8, kv_lengths, v_descale, torch.bfloat16),
    )
    return result, reference_inputs, cu_seqlens_q, cu_seqlens_k


def _vllm_varlen_reference(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    scale,
    causal,
    return_softmax_lse=False,
    seqused_k=None,
    block_table=None,
):
    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

    max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
    max_seqlen_k = int(
        (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()
        if cu_seqlens_k is not None
        else seqused_k.max().item()
    )
    return flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        cu_seqlens_k=cu_seqlens_k,
        seqused_k=seqused_k,
        softmax_scale=scale,
        causal=causal,
        block_table=block_table,
        out=torch.empty_like(q),
        return_softmax_lse=return_softmax_lse,
        fa_version=2,
    )


def _assert_w8a8_attention_close(actual, expected):
    torch.testing.assert_close(
        actual.float(), expected.float(), rtol=1.0e-2, atol=2.0e-2
    )


def _assert_w8a8_lse_close(result, lse, expected_lse, q_lengths, kv_lengths, causal):
    # Fully masked causal rows use implementation-specific LSE sentinels.
    valid_rows = torch.cat(
        [
            (
                torch.arange(q_len, device=device) >= max(0, q_len - kv_len)
                if causal
                else torch.ones(q_len, device=device, dtype=torch.bool)
            )
            for q_len, kv_len in zip(q_lengths, kv_lengths)
        ]
    )
    assert torch.isfinite(lse[:, valid_rows]).all()
    torch.testing.assert_close(
        lse[:, valid_rows], expected_lse[:, valid_rows], rtol=1.0e-2, atol=5.0e-2
    )
    torch.testing.assert_close(
        result[~valid_rows], torch.zeros_like(result[~valid_rows]), rtol=0, atol=0
    )


def _check_w8a8_head_dim(
    head_size,
    dtype,
    fp8_dtype,
    q_lengths,
    kv_lengths,
    causal,
    return_lse,
    strided=False,
):
    utils.init_seed(1234567890)
    q, k, v = _make_packed_inputs(q_lengths, kv_lengths, 4, head_size, dtype)
    # Unit-scale inputs exercise quantization and attention beyond near-uniform logits.
    for tensor in (q, k, v):
        tensor.normal_()
    scale = 1.0 / math.sqrt(head_size)
    actual, (ref_q, ref_k, ref_v), cu_q, cu_k = _run_w8a8_varlen(
        q,
        k,
        v,
        q_lengths,
        kv_lengths,
        scale,
        causal,
        return_softmax_lse=return_lse,
        fp8_dtype=fp8_dtype,
        strided=strided,
    )
    expected = _vllm_varlen_reference(
        ref_q,
        ref_k,
        ref_v,
        cu_q,
        cu_k,
        scale,
        causal,
        return_softmax_lse=return_lse,
    )
    if return_lse:
        actual, lse = actual
        expected, expected_lse = expected
        _assert_w8a8_lse_close(actual, lse, expected_lse, q_lengths, kv_lengths, causal)
    _assert_w8a8_attention_close(actual, expected)


@pytest.mark.flash_attn_varlen_func_w8a8_fp8
def test_flash_attn_varlen_func_w8a8_fp8_signature():
    assert inspect.signature(w8a8_varlen) == inspect.signature(fa2_varlen)
    assert flaggems_vllm.flash_attn_varlen_func_w8a8_fp8 is w8a8_varlen


@pytest.mark.flash_attn_varlen_func_w8a8_fp8
@pytest.mark.skipif(cfg.TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(vendor_name != "nvidia", reason="NVIDIA-only path")
@pytest.mark.skipif(
    not _supports_hopper_fp8(), reason="Requires NVIDIA Hopper or newer"
)
@pytest.mark.skipif(
    getattr(torch, "float8_e4m3fn", None) is None,
    reason="FP8 is not available",
)
@pytest.mark.parametrize("batch,num_heads,q_seq_len,kv_seq_len", W8A8_CONFIGS)
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_attn_varlen_func_w8a8_fp8(
    batch, num_heads, q_seq_len, kv_seq_len, head_size, causal, dtype
):
    utils.init_seed(1234567890)
    q_lengths = [q_seq_len] * batch
    kv_lengths = [kv_seq_len] * batch
    q, k, v = _make_packed_inputs(q_lengths, kv_lengths, num_heads, head_size, dtype)
    scale = 1.0 / math.sqrt(head_size)
    result, (ref_q, ref_k, ref_v), cu_q, cu_k = _run_w8a8_varlen(
        q, k, v, q_lengths, kv_lengths, scale, causal
    )
    expected = _vllm_varlen_reference(ref_q, ref_k, ref_v, cu_q, cu_k, scale, causal)
    _assert_w8a8_attention_close(result, expected)


@pytest.mark.flash_attn_varlen_func_w8a8_fp8
@pytest.mark.skipif(cfg.TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(vendor_name != "nvidia", reason="NVIDIA-only path")
@pytest.mark.skipif(
    not _supports_hopper_fp8(), reason="Requires NVIDIA Hopper or newer"
)
@pytest.mark.parametrize("head_size", W8A8_HEAD_SIZES)
@pytest.mark.parametrize("fp8_dtype", W8A8_FP8_DTYPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["uniform", "ragged"])
def test_flash_attn_varlen_func_w8a8_fp8_all_head_dims(
    head_size, fp8_dtype, dtype, layout
):
    if layout == "uniform":
        q_lengths, kv_lengths = [128, 128], [256, 256]
    else:
        q_lengths, kv_lengths = [129, 17], [65, 257]
    _check_w8a8_head_dim(
        head_size,
        dtype,
        fp8_dtype,
        q_lengths,
        kv_lengths,
        causal=layout == "ragged",
        return_lse=layout == "ragged",
    )


@pytest.mark.flash_attn_varlen_func_w8a8_fp8
@pytest.mark.skipif(cfg.TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(vendor_name != "nvidia", reason="NVIDIA-only path")
@pytest.mark.skipif(
    not _supports_hopper_fp8(), reason="Requires NVIDIA Hopper or newer"
)
@pytest.mark.parametrize("head_size", [8, 24, 96, 192, 248, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["uniform", "ragged"])
def test_flash_attn_varlen_func_w8a8_fp8_head_dim_strides(head_size, dtype, layout):
    if layout == "uniform":
        q_lengths, kv_lengths = [128, 128], [256, 256]
    else:
        q_lengths, kv_lengths = [129, 17], [65, 257]
    _check_w8a8_head_dim(
        head_size,
        dtype,
        _get_fp8_dtype(),
        q_lengths,
        kv_lengths,
        causal=layout == "uniform",
        return_lse=True,
        strided=True,
    )


@pytest.mark.flash_attn_varlen_func_w8a8_fp8
@pytest.mark.skipif(cfg.TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(vendor_name != "nvidia", reason="NVIDIA-only path")
@pytest.mark.skipif(
    not _supports_hopper_fp8(), reason="Requires NVIDIA Hopper or newer"
)
@pytest.mark.parametrize("head_size", [8, 24, 96, 136, 192, 248, 256])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_varlen_func_w8a8_fp8_head_dim_long_sequences(head_size, causal):
    _check_w8a8_head_dim(
        head_size,
        torch.bfloat16,
        _get_fp8_dtype(),
        [2049, 1023],
        [4097, 3073],
        causal=causal,
        return_lse=True,
    )


@pytest.mark.flash_attn_varlen_func_w8a8_fp8
@pytest.mark.skipif(cfg.TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(vendor_name != "nvidia", reason="NVIDIA-only path")
@pytest.mark.skipif(
    not _supports_hopper_fp8(), reason="Requires NVIDIA Hopper or newer"
)
@pytest.mark.parametrize("head_size", [8, 96, 256])
@pytest.mark.parametrize("kv_len", [1, 128])
def test_flash_attn_varlen_func_w8a8_fp8_head_dim_short_kv(head_size, kv_len):
    _check_w8a8_head_dim(
        head_size,
        torch.bfloat16,
        _get_fp8_dtype(),
        [17],
        [kv_len],
        causal=True,
        return_lse=True,
    )


@pytest.mark.flash_attn_varlen_func_w8a8_fp8
@pytest.mark.skipif(cfg.TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(vendor_name != "nvidia", reason="NVIDIA-only path")
@pytest.mark.skipif(
    not _supports_hopper_fp8(), reason="Requires NVIDIA Hopper or newer"
)
@pytest.mark.skipif(
    getattr(torch, "float8_e4m3fn", None) is None,
    reason="FP8 is not available",
)
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("q_len,kv_len", [(129, 257), (17, 8192), (1024, 8192)])
@pytest.mark.parametrize("fp8_dtype", W8A8_FP8_DTYPES)
def test_flash_attn_varlen_func_w8a8_fp8_uniform_lse(
    head_size, causal, q_len, kv_len, fp8_dtype
):
    utils.init_seed(1234567890)
    dtype = torch.bfloat16
    batch, num_heads = 2, 4
    q_lengths = [q_len] * batch
    kv_lengths = [kv_len] * batch
    q, k, v = _make_packed_inputs(q_lengths, kv_lengths, num_heads, head_size, dtype)
    scale = 1.0 / math.sqrt(head_size)
    (result, lse), (ref_q, ref_k, ref_v), cu_q, cu_k = _run_w8a8_varlen(
        q,
        k,
        v,
        q_lengths,
        kv_lengths,
        scale,
        causal,
        return_softmax_lse=True,
        fp8_dtype=fp8_dtype,
    )
    expected, expected_lse = _vllm_varlen_reference(
        ref_q,
        ref_k,
        ref_v,
        cu_q,
        cu_k,
        scale,
        causal,
        return_softmax_lse=True,
    )
    _assert_w8a8_attention_close(result, expected)
    torch.testing.assert_close(lse, expected_lse, rtol=1.0e-2, atol=5.0e-2)


@pytest.mark.flash_attn_varlen_func_w8a8_fp8
@pytest.mark.skipif(cfg.TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(vendor_name != "nvidia", reason="NVIDIA-only path")
@pytest.mark.skipif(
    not _supports_hopper_fp8(), reason="Requires NVIDIA Hopper or newer"
)
@pytest.mark.skipif(
    getattr(torch, "float8_e4m3fn", None) is None,
    reason="FP8 is not available",
)
@pytest.mark.parametrize(
    "head_size,dtype,causal,q_lengths,kv_lengths",
    [
        (64, torch.float16, False, [17, 129], [33, 257]),
        (128, torch.bfloat16, True, [33, 257], [17, 129]),
        (64, torch.float16, False, [128, 257], [257, 129]),
        (64, torch.float16, True, [129, 512], [65, 777]),
        (64, torch.bfloat16, True, [129, 513], [65, 777]),
        (64, torch.float16, True, [32, 128, 512, 4096], [1, 17, 129, 8192]),
        (64, torch.bfloat16, True, [32, 128, 512, 4096], [1, 17, 129, 8192]),
        (128, torch.float16, False, [129, 1024], [257, 1537]),
        (128, torch.bfloat16, False, [129, 1025], [257, 1537]),
        (128, torch.float16, False, [129, 4096], [257, 8192]),
        (128, torch.bfloat16, False, [129, 4096], [257, 8192]),
    ],
)
def test_flash_attn_varlen_func_w8a8_fp8_ragged(
    head_size, dtype, causal, q_lengths, kv_lengths
):
    utils.init_seed(1234567890)
    num_heads = 8
    q, k, v = _make_packed_inputs(q_lengths, kv_lengths, num_heads, head_size, dtype)
    scale = 1.0 / math.sqrt(head_size)
    (result, lse), (ref_q, ref_k, ref_v), cu_q, cu_k = _run_w8a8_varlen(
        q,
        k,
        v,
        q_lengths,
        kv_lengths,
        scale,
        causal,
        return_softmax_lse=True,
    )
    expected, expected_lse = _vllm_varlen_reference(
        ref_q,
        ref_k,
        ref_v,
        cu_q,
        cu_k,
        scale,
        causal,
        return_softmax_lse=True,
    )
    _assert_w8a8_attention_close(result, expected)

    # Fully masked causal rows use implementation-specific LSE sentinels.
    valid_rows = torch.cat(
        [
            (
                torch.arange(q_len, device=device) >= max(0, q_len - kv_len)
                if causal
                else torch.ones(q_len, device=device, dtype=torch.bool)
            )
            for q_len, kv_len in zip(q_lengths, kv_lengths)
        ]
    )
    torch.testing.assert_close(
        lse[:, valid_rows],
        expected_lse[:, valid_rows],
        rtol=1.0e-2,
        atol=5.0e-2,
    )


@pytest.mark.flash_attn_varlen_func_w8a8_fp8
@pytest.mark.skipif(cfg.TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(vendor_name != "nvidia", reason="NVIDIA-only path")
@pytest.mark.skipif(
    not _supports_hopper_fp8(), reason="Requires NVIDIA Hopper or newer"
)
@pytest.mark.skipif(
    getattr(torch, "float8_e4m3fn", None) is None,
    reason="FP8 is not available",
)
@pytest.mark.parametrize("head_size", W8A8_HEAD_SIZES)
@pytest.mark.parametrize("fp8_dtype", W8A8_FP8_DTYPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_attn_varlen_func_w8a8_fp8_paged_cache(head_size, fp8_dtype, dtype):
    utils.init_seed(1234567890)
    causal = dtype == torch.float16
    num_heads = 8
    q_lengths = [33, 17]
    kv_lengths = [129, 65]
    block_size = 64
    q, k, v = _make_packed_inputs(q_lengths, kv_lengths, num_heads, head_size, dtype)
    for tensor in (q, k, v):
        tensor.normal_()
    (
        q_fp8,
        k_fp8,
        v_fp8,
        q_descale,
        k_descale,
        v_descale,
    ) = _quantize_qkv_w8a8(q, k, v, q_lengths, kv_lengths, fp8_dtype=fp8_dtype)
    page_table = torch.tensor([[5, 1, 7], [3, 6, 0]], dtype=torch.int32, device=device)
    num_pages = 8

    def to_paged_cache(packed):
        cache = torch.zeros(
            (num_pages, block_size, num_heads, head_size),
            dtype=packed.dtype,
            device=device,
        )
        token_offset = 0
        for batch_idx, seq_len in enumerate(kv_lengths):
            for logical_page in range(triton.cdiv(seq_len, block_size)):
                lo = token_offset + logical_page * block_size
                page_tokens = min(block_size, seq_len - logical_page * block_size)
                physical_page = page_table[batch_idx, logical_page].item()
                cache[physical_page, :page_tokens] = packed[lo : lo + page_tokens]
            token_offset += seq_len
        return cache

    cu_q = _cu_seqlens_from_lengths(q_lengths, device)
    seqused_k = torch.tensor(kv_lengths, dtype=torch.int32, device=device)
    scale = 1.0 / math.sqrt(head_size)
    result, lse = w8a8_varlen(
        q_fp8,
        to_paged_cache(k_fp8),
        to_paged_cache(v_fp8),
        max(q_lengths),
        cu_q,
        max(kv_lengths),
        seqused_k=seqused_k,
        softmax_scale=scale,
        causal=causal,
        block_table=page_table,
        out=torch.empty_like(q),
        return_softmax_lse=True,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    ref_q = _dequantize_varlen_per_block_fp8(
        q_fp8, q_lengths, q_descale, torch.bfloat16
    )
    ref_k = _dequantize_varlen_per_block_fp8(
        k_fp8, kv_lengths, k_descale, torch.bfloat16
    )
    ref_v = _dequantize_varlen_per_block_fp8(
        v_fp8, kv_lengths, v_descale, torch.bfloat16
    )
    expected, expected_lse = _vllm_varlen_reference(
        ref_q,
        to_paged_cache(ref_k),
        to_paged_cache(ref_v),
        cu_q,
        None,
        scale,
        causal,
        return_softmax_lse=True,
        seqused_k=seqused_k,
        block_table=page_table,
    )
    _assert_w8a8_attention_close(result, expected)
    _assert_w8a8_lse_close(result, lse, expected_lse, q_lengths, kv_lengths, causal)


@pytest.mark.flash_attn_varlen_func_w8a8_fp8
@pytest.mark.skipif(cfg.TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(vendor_name != "nvidia", reason="NVIDIA-only path")
@pytest.mark.skipif(
    not _supports_hopper_fp8(), reason="Requires NVIDIA Hopper or newer"
)
@pytest.mark.skipif(
    getattr(torch, "float8_e4m3fn", None) is None,
    reason="FP8 is not available",
)
def test_flash_attn_varlen_func_w8a8_fp8_rejects_unsupported_inputs():
    fp8_dtype = _get_fp8_dtype()
    q = torch.empty((128, 4, 64), dtype=fp8_dtype, device=device)
    k = torch.empty((128, 2, 64), dtype=fp8_dtype, device=device)
    v = torch.empty_like(k)
    q_descale = torch.ones((1, 4, 1), dtype=torch.float32, device=device)
    kv_descale = torch.ones((1, 2, 1), dtype=torch.float32, device=device)
    cu_seqlens = torch.tensor([0, 128], dtype=torch.int32, device=device)

    with pytest.raises(NotImplementedError, match="GQA is not supported"):
        w8a8_varlen(
            q,
            k,
            v,
            128,
            cu_seqlens,
            128,
            cu_seqlens,
            q_descale=q_descale,
            k_descale=kv_descale,
            v_descale=kv_descale,
        )

    k = torch.empty_like(q)
    v = torch.empty_like(q)
    with pytest.raises(NotImplementedError, match="dropout is not supported"):
        w8a8_varlen(
            q,
            k,
            v,
            128,
            cu_seqlens,
            128,
            cu_seqlens,
            dropout_p=0.1,
        )
    with pytest.raises(ValueError, match="q_descale is required"):
        w8a8_varlen(q, k, v, 128, cu_seqlens, 128, cu_seqlens)
    with pytest.raises(NotImplementedError, match="without a paged KV cache"):
        w8a8_varlen(
            q,
            k,
            v,
            128,
            cu_seqlens,
            128,
            seqused_k=torch.tensor([128], dtype=torch.int32, device=device),
        )

    paged_k = torch.empty((2, 64, 4, 64), dtype=fp8_dtype, device=device)
    with pytest.raises(ValueError, match="block_table"):
        w8a8_varlen(
            q,
            paged_k,
            torch.empty_like(paged_k),
            128,
            cu_seqlens,
            128,
            seqused_k=torch.tensor([128], dtype=torch.int32, device=device),
            block_table=torch.zeros((1, 1, 1), dtype=torch.int32, device=device),
        )


@pytest.mark.flash_attn_varlen_func_w8a8_fp8
@pytest.mark.skipif(cfg.TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(vendor_name != "nvidia", reason="NVIDIA-only path")
@pytest.mark.skipif(
    not _supports_hopper_fp8(), reason="Requires NVIDIA Hopper or newer"
)
@pytest.mark.parametrize("head_size", [0, 7, 65, 264])
def test_flash_attn_varlen_func_w8a8_fp8_rejects_invalid_head_dim(head_size):
    q = torch.empty((1, 4, head_size), dtype=_get_fp8_dtype(), device=device)
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32, device=device)
    with pytest.raises(NotImplementedError, match="head_dim"):
        w8a8_varlen(
            q, torch.empty_like(q), torch.empty_like(q), 1, cu_seqlens, 1, cu_seqlens
        )


@pytest.mark.flash_attn_varlen_func_w8a8_fp8
@pytest.mark.skipif(cfg.TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(vendor_name != "nvidia", reason="NVIDIA-only path")
@pytest.mark.skipif(
    not _supports_hopper_fp8(), reason="Requires NVIDIA Hopper or newer"
)
def test_flash_attn_varlen_func_w8a8_fp8_rejects_mismatched_head_dims():
    fp8_dtype = _get_fp8_dtype()
    q = torch.empty((1, 4, 64), dtype=fp8_dtype, device=device)
    k = torch.empty((1, 4, 96), dtype=fp8_dtype, device=device)
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="head_dim"):
        w8a8_varlen(q, k, torch.empty_like(k), 1, cu_seqlens, 1, cu_seqlens)
    with pytest.raises(ValueError, match="identical shapes"):
        w8a8_varlen(q, torch.empty_like(q), k, 1, cu_seqlens, 1, cu_seqlens)


@pytest.mark.flash_attn_varlen_func_w8a8_fp8
@pytest.mark.skipif(cfg.TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(vendor_name != "nvidia", reason="NVIDIA-only path")
@pytest.mark.skipif(
    not _supports_hopper_fp8(), reason="Requires NVIDIA Hopper or newer"
)
@pytest.mark.skipif(
    getattr(torch, "float8_e4m3fn", None) is None,
    reason="FP8 is not available",
)
def test_flash_attn_varlen_func_w8a8_fp8_out_identity():
    fp8_dtype = _get_fp8_dtype()
    q = torch.zeros((128, 4, 64), dtype=fp8_dtype, device=device)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    descale = torch.ones((1, 4, 1), dtype=torch.float32, device=device)
    cu_seqlens = torch.tensor([0, 128], dtype=torch.int32, device=device)
    out = torch.empty((128, 4, 64), dtype=torch.bfloat16, device=device)

    result = w8a8_varlen(
        q,
        k,
        v,
        128,
        cu_seqlens,
        128,
        cu_seqlens,
        out=out,
        q_descale=descale,
        k_descale=descale,
        v_descale=descale,
    )
    assert result is out
