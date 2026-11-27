# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare strided paged K/V attention with a PyTorch selected-page baseline."""

import pytest
import torch

import flaggems_vllm

from . import base

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend", reason="requires Ascend BF16 paged K/V"
)


def _attention(q, cache, ids, table, seq, prefix, scale, output):
    # These benchmark cases have one request and one KV head.
    pages = table[0, ids[0].long()]
    key, value = (view[pages, :, 0].flatten(1, 2).float() for view in cache)
    scores = q.float() @ key.transpose(1, 2) * scale
    offsets = torch.arange(128, device=q.device)
    positions = (ids[0, :, :, None] * 128 + offsets).flatten(1)
    ends = torch.minimum(prefix + torch.arange(len(q), device=q.device), seq - 1)
    scores.masked_fill_(positions[:, None, :] > ends[:, None, None], -torch.inf)
    output.copy_(scores.softmax(-1) @ value)


class SparseBenchmark(base.Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16]
    DEFAULT_SHAPE_DESC = "query tokens, context length"

    def set_shapes(self, shape_file_path=None):
        self.shapes = (
            [(1, 128), (1, 16384)] if self.op_name.endswith("decode") else [(32, 4096)]
        )

    def get_input_iter(self, dtype):
        for rows, length in self.shapes:
            q = torch.randn(rows, 4, 128, device="npu", dtype=dtype)
            cache = torch.randn(
                length // 128, 2, 128, 1, 128, device=q.device, dtype=dtype
            )
            table = torch.arange(len(cache), device=q.device, dtype=torch.int32)[
                None, :
            ]
            ids = torch.linspace(
                0, len(cache) - 1, min(len(cache), 16), device=q.device
            ).int()
            ids = ids.view(1, 1, -1).expand(1, rows, -1)
            seq = torch.full((1,), length, device=q.device, dtype=torch.int32)
            cu = torch.tensor([0, rows], device=q.device, dtype=torch.int32)
            yield q, cache.unbind(
                1
            ), ids, table, cu, seq, seq - rows, rows, 1, 128**-0.5, torch.empty_like(
                q
            )


def _reference(q, cache, ids, table, cu, seq, prefix, rows, heads, scale, out):
    _attention(q, cache, ids, table, seq, prefix, scale, out)


def _decode(q, cache, ids, table, cu, seq, prefix, rows, heads, scale, out):
    flaggems_vllm.minimax_m3_sparse_attn_decode(
        q, cache, ids, table, seq, heads, scale, out, rows
    )


@pytest.mark.minimax_m3_sparse_attn
def test_ascend_prefill():
    SparseBenchmark(
        op_name="minimax_m3_sparse_attn",
        torch_op=_reference,
        gems_op=flaggems_vllm.minimax_m3_sparse_attn,
    ).run()


@pytest.mark.minimax_m3_sparse_attn_decode
def test_ascend_decode():
    SparseBenchmark(
        op_name="minimax_m3_sparse_attn_decode", torch_op=_reference, gems_op=_decode
    ).run()
