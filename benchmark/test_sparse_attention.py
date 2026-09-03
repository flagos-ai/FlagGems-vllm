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

import flaggems_vllm

from . import base

# sparse_attention shape layout:
# (batch, seq_len, kv_len, topk, heads, dim)
SPARSE_ATTENTION_SHAPES = [
    (16, 1, 136, 136, 8, 512),
    (16, 1, 392, 385, 8, 512),
    (16, 1, 392, 386, 8, 512),
    (16, 1, 392, 387, 8, 512),
    (32, 1, 392, 388, 8, 512),
    (32, 1, 392, 389, 8, 512),
    (32, 1, 392, 390, 8, 512),
    (32, 1, 392, 391, 8, 512),
    (64, 1, 136, 136, 8, 512),
    (64, 1, 392, 385, 8, 512),
    (64, 1, 392, 388, 8, 512),
    (64, 1, 392, 389, 8, 512),
]

QMAX_INT8 = 127.0
QMAX_FP8 = 448.0
KV_BLOCK = 64


def _quantize_kv(kv, quant_dtype):
    """Reference KV quantizer: per-64-position-block symmetric scale.

    Returns (kv_q, kv_descale). kv_q is the quantized KV in the given dtype,
    kv_descale is (B, ceil(kv_len / 64)) fp32.
    """
    batch, kv_len, dim = kv.shape
    qmax = QMAX_FP8 if quant_dtype == torch.float8_e4m3fn else QMAX_INT8
    nblocks = math.ceil(kv_len / KV_BLOCK)
    kv_pad = torch.zeros(
        (batch, nblocks * KV_BLOCK, dim), dtype=kv.dtype, device=kv.device
    )
    kv_pad[:, :kv_len, :] = kv
    kv_r = kv_pad.view(batch, nblocks, KV_BLOCK, dim).float()
    blk_max = kv_r.abs().amax(dim=(2, 3))
    scale_b = blk_max / qmax
    inv_b = torch.where(blk_max == 0.0, 0.0, 1.0 / scale_b)
    if quant_dtype == torch.float8_e4m3fn:
        kv_q_r = (kv_r * inv_b[:, :, None, None]).to(torch.float8_e4m3fn)
    else:
        kv_q_r = torch.clamp(
            torch.round(kv_r * inv_b[:, :, None, None]), -128.0, 127.0
        ).to(torch.int8)
    kv_q = kv_q_r.view(batch, nblocks * KV_BLOCK, dim)[:, :kv_len, :]
    return kv_q, scale_b


def torch_sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Full-precision fp32 torch baseline for the plain operator."""
    batch, seq_len, heads, dim = q.shape
    topk = topk_idxs.shape[-1]

    kv_expanded = kv[:, None, :, :].expand(batch, seq_len, -1, dim)
    idx_expanded = topk_idxs[:, :, :, None].expand(batch, seq_len, topk, dim).long()
    gathered_kv = torch.gather(kv_expanded, 2, idx_expanded)

    scores = (
        torch.einsum("bmhd,bmtd->bmht", q.float(), gathered_kv.float()) * softmax_scale
    )
    scores = torch.where(topk_idxs[:, :, None, :] >= 0, scores, float("-inf"))
    sink = attn_sink[None, None, :, None].expand(batch, seq_len, heads, 1)
    attn = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)

    out = torch.einsum("bmht,bmtd->bmhd", attn[:, :, :, :-1], gathered_kv.float())
    return out.to(q.dtype)


def torch_sparse_attention_quant(
    q, kv_q, kv_descale, attn_sink, topk_idxs, softmax_scale
):
    """fp32 torch baseline for the quantized-KV sparse attention.

    Dequantizes the quantized KV cache (kv_q * per-block kv_descale) and runs
    the same dense gather+einsum attention in fp32. quant_dtype is not needed:
    the descale already carries the dtype-dependent scale.
    """
    batch, seq_len, heads, dim = q.shape
    topk = topk_idxs.shape[-1]
    kv_len = kv_q.shape[1]

    idx = topk_idxs.long()
    idx_safe = idx.clamp(min=0)
    desc = torch.gather(
        kv_descale.unsqueeze(1).expand(batch, seq_len, -1), 2, idx_safe // KV_BLOCK
    )  # (b, m, topk) fp32
    kv_g = torch.gather(
        kv_q.unsqueeze(1).expand(batch, seq_len, kv_len, dim),
        2,
        idx.unsqueeze(-1).expand(batch, seq_len, topk, dim),
    )  # (b, m, topk, d) quant dtype
    kv_deq = kv_g.float() * desc[..., None]

    scores = torch.einsum("bmhd,bmtd->bmht", q.float(), kv_deq) * softmax_scale
    scores = torch.where(topk_idxs[:, :, None, :] >= 0, scores, float("-inf"))
    sink = attn_sink[None, None, :, None].expand(batch, seq_len, heads, 1)
    attn = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)

    out = torch.einsum("bmht,bmtd->bmhd", attn[:, :, :, :-1], kv_deq)
    return out.to(q.dtype)


def _make_inputs(batch, seq_len, kv_len, topk, heads, dim, seed, dtype):
    torch.manual_seed(seed)
    q = torch.randn((batch, seq_len, heads, dim), dtype=dtype, device="cuda")
    kv = torch.randn((batch, kv_len, dim), dtype=dtype, device="cuda")
    attn_sink = torch.zeros((heads,), dtype=torch.float32, device="cuda")
    topk_idxs = torch.randint(
        0, kv_len, (batch, seq_len, topk), dtype=torch.int32, device="cuda"
    )
    return q, kv, attn_sink, topk_idxs


class SparseAttentionBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SPARSE_ATTENTION_SHAPES[:]
        self.shape_desc = "B, M, KV_LEN, TOPK, H, D"

    def set_more_shapes(self):
        return None

    def get_input_iter(self, dtype):
        for seed, shape in enumerate(self.shapes):
            batch, seq_len, kv_len, topk, heads, dim = shape
            q, kv, attn_sink, topk_idxs = _make_inputs(
                batch, seq_len, kv_len, topk, heads, dim, 2026 + seed, dtype
            )
            yield q, kv, attn_sink, topk_idxs, 1.0 / math.sqrt(dim)


@pytest.mark.skipif(flaggems_vllm.device == "cpu", reason="Unsupported in CPU mode")
@pytest.mark.sparse_attn_triton
def test_sparse_attn_triton():
    bench = SparseAttentionBenchmark(
        op_name="sparse_attention",
        torch_op=torch_sparse_attention,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(flaggems_vllm.sparse_attn_triton)
    bench.run()


class SparseAttentionQuantBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SPARSE_ATTENTION_SHAPES[:]
        self.shape_desc = "B, M, KV_LEN, TOPK, H, D"

    def set_more_shapes(self):
        return None

    def __init__(self, quant_dtype, *args, **kwargs):
        self.quant_dtype = quant_dtype
        super().__init__(*args, **kwargs)

    def get_input_iter(self, dtype):
        for seed, shape in enumerate(self.shapes):
            batch, seq_len, kv_len, topk, heads, dim = shape
            q, kv, attn_sink, topk_idxs = _make_inputs(
                batch, seq_len, kv_len, topk, heads, dim, 2026 + seed, dtype
            )
            kv_q, kv_descale = _quantize_kv(kv, self.quant_dtype)
            yield q, kv_q, kv_descale, attn_sink, topk_idxs, 1.0 / math.sqrt(dim)


@pytest.mark.skipif(flaggems_vllm.device == "cpu", reason="Unsupported in CPU mode")
@pytest.mark.sparse_attn_quant
def test_sparse_attn_triton_quant_int8():
    bench = SparseAttentionQuantBenchmark(
        op_name="sparse_attention_quant_int8",
        torch_op=torch_sparse_attention_quant,
        dtypes=[torch.bfloat16],
        quant_dtype=torch.int8,
    )
    bench.set_gems(flaggems_vllm.sparse_attn_triton_quant_int8)
    bench.run()


@pytest.mark.skipif(flaggems_vllm.device == "cpu", reason="Unsupported in CPU mode")
@pytest.mark.sparse_attn_quant
def test_sparse_attn_triton_quant_fp8():
    bench = SparseAttentionQuantBenchmark(
        op_name="sparse_attention_quant_fp8",
        torch_op=torch_sparse_attention_quant,
        dtypes=[torch.bfloat16],
        quant_dtype=torch.float8_e4m3fn,
    )
    bench.set_gems(flaggems_vllm.sparse_attn_triton_quant_fp8)
    bench.run()
