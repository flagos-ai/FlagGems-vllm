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

import pytest
import torch

import flaggems_vllm as gems

from . import base

pytestmark = [
    pytest.mark.minimax_m3_decode_qk_score,
    pytest.mark.skipif(
        gems.vendor_name != "ascend"
        or not torch.npu.get_device_name().startswith("Ascend950"),
        reason="Atlas A5 specialization",
    ),
]


def _inputs(shape, dtype, device):
    batch, length = shape
    pages = (length + 127) // 128
    q = torch.randn((batch, 1, 128), dtype=dtype, device=device)
    cache = torch.randn((batch * pages, 128, 128), dtype=dtype, device=device)
    table = torch.arange(batch * pages, dtype=torch.int32, device=device).reshape(
        batch, pages
    )
    lengths = torch.full((batch,), length, dtype=torch.int32, device=device)
    return (q, cache, table, lengths), {}


def _reference():
    import importlib

    importlib.import_module("vllm_ascend.ops.triton.triton_utils")
    _decode_qk_score_kernel = importlib.import_module(
        "vllm_ascend.models.minimax_m3.ops.msa_m3_triton_a5"
    )._decode_qk_score_kernel
    from triton.runtime.jit import JITFunction

    while not isinstance(_decode_qk_score_kernel, JITFunction):
        _decode_qk_score_kernel = _decode_qk_score_kernel.fn

    def native(
        q,
        index_cache,
        block_table,
        seq_lens,
        decode_query_len=1,
        init_blocks=0,
        local_blocks=1,
    ):
        tokens, heads, dim = q.shape
        scores = torch.full(
            (heads, tokens, block_table.shape[1]),
            -float("inf"),
            dtype=torch.float32,
            device=q.device,
        )
        chunks = min(4, block_table.shape[1])
        _decode_qk_score_kernel[(tokens, chunks)](
            q,
            index_cache,
            scores,
            block_table,
            seq_lens,
            heads,
            dim,
            decode_query_len,
            init_blocks,
            local_blocks,
            *q.stride(),
            *index_cache.stride(),
            *scores.stride(),
            block_table.stride(0),
            BLOCK_SIZE_K=128,
            NUM_CHUNKS=chunks,
        )
        return scores

    return native


def _input_fn(shape, dtype, device):
    args, kwargs = _inputs(shape, dtype, device)
    yield (*args, kwargs)


class A5Benchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


def test_minimax_m3_decode_qk_score():
    b = A5Benchmark(
        op_name="minimax_m3_decode_qk_score",
        torch_op=_reference(),
        input_fn=_input_fn,
        dtypes=[torch.bfloat16],
    )
    b.set_gems(gems.minimax_m3_decode_qk_score)
    b.run()
