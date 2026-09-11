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

import math

import pytest
import torch
import triton

import flaggems_vllm
from flaggems_vllm.ops import flash_attn_varlen_func_w8a8_fp8 as w8a8_varlen

from . import base, utils

vendor_name = flaggems_vllm.vendor_name


def _supports_hopper_fp8() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


def _get_fp8_dtype():
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        pytest.skip("torch.float8_e4m3fn is not available")
    return dtype


def _hadamard_matrix(dim, device):
    assert dim > 0 and dim & (dim - 1) == 0, "head_size must be a power of two"
    matrix = torch.tensor([[1.0]], device=device)
    while matrix.shape[0] < dim:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix / math.sqrt(dim)


def _apply_incoherent_qk(x):
    matrix = _hadamard_matrix(x.shape[-1], x.device).to(torch.float32)
    return torch.matmul(x.float(), matrix).to(x.dtype)


def _quantize_varlen_per_block_fp8(x, seq_lens, fp8_dtype, block_size=128):
    total_tokens, num_heads, _ = x.shape
    assert total_tokens == sum(seq_lens)
    fp8_max = float(torch.finfo(fp8_dtype).max)
    num_blocks = triton.cdiv(max(seq_lens), block_size)
    quantized = torch.empty_like(x, dtype=fp8_dtype)
    descale = torch.ones(
        (len(seq_lens), num_heads, num_blocks),
        device=x.device,
        dtype=torch.float32,
    )

    token_offset = 0
    for batch_idx, seq_len in enumerate(seq_lens):
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


def _quantize_qkv_w8a8(q, k, v, q_seq_lens, kv_seq_lens):
    fp8_dtype = _get_fp8_dtype()
    q_fp8, q_descale = _quantize_varlen_per_block_fp8(
        _apply_incoherent_qk(q), q_seq_lens, fp8_dtype
    )
    k_fp8, k_descale = _quantize_varlen_per_block_fp8(
        _apply_incoherent_qk(k), kv_seq_lens, fp8_dtype
    )
    v_fp8, v_descale = _quantize_varlen_per_block_fp8(v, kv_seq_lens, fp8_dtype)
    return q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale


def baseline_flash_attn_varlen_func_w8a8_fp8(
    q,
    k,
    v,
    q_fp8,
    k_fp8,
    v_fp8,
    q_descale,
    k_descale,
    v_descale,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k,
    scale,
    causal,
    baseline_out,
    w8a8_out,
):
    return flaggems_vllm.flash_attn_varlen_func(
        q,
        k,
        v,
        max_seqlen_q,
        cu_seqlens_q,
        max_seqlen_k,
        cu_seqlens_k,
        softmax_scale=scale,
        causal=causal,
        out=baseline_out,
    )


def gems_flash_attn_varlen_func_w8a8_fp8(
    q,
    k,
    v,
    q_fp8,
    k_fp8,
    v_fp8,
    q_descale,
    k_descale,
    v_descale,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k,
    scale,
    causal,
    baseline_out,
    w8a8_out,
):
    return w8a8_varlen(
        q_fp8,
        k_fp8,
        v_fp8,
        max_seqlen_q,
        cu_seqlens_q,
        max_seqlen_k,
        cu_seqlens_k,
        softmax_scale=scale,
        causal=causal,
        out=w8a8_out,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )


class FlashAttnVarlenFuncW8A8FP8Benchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        all_shapes = []

        for batch in (1, 2, 4, 8):
            all_shapes.extend(
                [
                    (batch, 512, 16, 128, False),
                    (batch, 512, 32, 64, False),
                    (batch, 512, 16, 128, True),
                    (batch, 512, 32, 64, True),
                ]
            )

        for batch in (1, 2, 4, 8):
            for seq_len in (1024, 2048, 4096, 8192):
                all_shapes.extend(
                    [
                        (batch, seq_len, 16, 128, False),
                        (batch, seq_len, 32, 64, False),
                    ]
                )

        all_shapes.extend(
            [
                (8, 8192, 16, 128, True),
                (8, 8192, 32, 64, True),
            ]
        )

        ragged_q = (32, 128, 512, 4096)
        ragged_kv = (1, 17, 129, 8192)
        for num_heads, head_size in ((32, 64), (16, 128)):
            for causal in (False, True):
                all_shapes.append((ragged_q, ragged_kv, num_heads, head_size, causal))

        core_shapes = [
            (1, 512, 16, 128, False),
            (1, 512, 32, 64, False),
            (2, 512, 16, 128, True),
            (1, 2048, 32, 64, False),
            (4, 4096, 32, 64, False),
            (8, 8192, 16, 128, True),
            (ragged_q, ragged_kv, 32, 64, False),
        ]
        self.shapes = (
            all_shapes
            if base.Config.bench_level == base.consts.BenchLevel.COMPREHENSIVE
            else core_shapes
        )

    def set_more_shapes(self):
        return []


def _make_deterministic_ragged_lengths(batch, max_seq_len):
    if batch == 1:
        return (max_seq_len,), (max_seq_len,)

    step = max(1, max_seq_len // (2 * batch))
    q_seq_lens = tuple(
        max(1, max_seq_len - batch_idx * step - batch_idx % 3)
        for batch_idx in range(batch)
    )
    kv_seq_lens = q_seq_lens[1:] + q_seq_lens[:1]
    return q_seq_lens, kv_seq_lens


def _make_cu_seqlens(seq_lens, device):
    return torch.tensor(
        (0,) + seq_lens,
        device=device,
        dtype=torch.int32,
    ).cumsum(dim=0, dtype=torch.int32)


def flash_attn_varlen_func_w8a8_fp8_input_fn(config, dtype, device):
    batch_or_q_lens, seq_len_or_kv_lens, num_heads, head_size, causal = config
    if isinstance(batch_or_q_lens, (list, tuple)):
        q_seq_lens = tuple(batch_or_q_lens)
        kv_seq_lens = tuple(seq_len_or_kv_lens)
        assert len(q_seq_lens) == len(kv_seq_lens)
    else:
        q_seq_lens, kv_seq_lens = _make_deterministic_ragged_lengths(
            batch_or_q_lens, seq_len_or_kv_lens
        )

    q = torch.empty(
        (sum(q_seq_lens), num_heads, head_size), device=device, dtype=dtype
    ).uniform_(-0.05, 0.05)
    k = torch.empty(
        (sum(kv_seq_lens), num_heads, head_size), device=device, dtype=dtype
    ).uniform_(-0.05, 0.05)
    v = torch.empty_like(k).uniform_(-0.05, 0.05)
    (
        q_fp8,
        k_fp8,
        v_fp8,
        q_descale,
        k_descale,
        v_descale,
    ) = _quantize_qkv_w8a8(q, k, v, q_seq_lens, kv_seq_lens)
    cu_seqlens_q = _make_cu_seqlens(q_seq_lens, device)
    cu_seqlens_k = _make_cu_seqlens(kv_seq_lens, device)
    baseline_out = torch.empty_like(q)
    w8a8_out = torch.empty_like(q)
    torch.cuda.synchronize()

    yield (
        q,
        k,
        v,
        q_fp8,
        k_fp8,
        v_fp8,
        q_descale,
        k_descale,
        v_descale,
        max(q_seq_lens),
        cu_seqlens_q,
        max(kv_seq_lens),
        cu_seqlens_k,
        1.0 / math.sqrt(head_size),
        causal,
        baseline_out,
        w8a8_out,
    )


@pytest.mark.skipif(
    utils.SkipVersion("torch", "<2.7"),
    reason="Torch version prior to 2.7 is not compatible with vLLM.",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(flaggems_vllm.device == "cpu", reason="Unsupported in CPU mode")
@pytest.mark.skipif(vendor_name != "nvidia", reason="NVIDIA-only path")
@pytest.mark.skipif(
    not _supports_hopper_fp8(), reason="Requires NVIDIA Hopper or newer"
)
@pytest.mark.skipif(
    getattr(torch, "float8_e4m3fn", None) is None,
    reason="FP8 is not available",
)
@pytest.mark.flash_attn_varlen_func_w8a8_fp8
def test_flash_attn_varlen_func_w8a8_fp8():
    bench = FlashAttnVarlenFuncW8A8FP8Benchmark(
        op_name="flash_attn_varlen_func_w8a8_fp8",
        input_fn=flash_attn_varlen_func_w8a8_fp8_input_fn,
        torch_op=baseline_flash_attn_varlen_func_w8a8_fp8,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.set_gems(gems_flash_attn_varlen_func_w8a8_fp8)
    bench.run()
