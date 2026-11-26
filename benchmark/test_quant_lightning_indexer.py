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

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from vllm-plugin-FL quant_lightning_indexer.py.
"""Torch reference and native vLLM-Ascend comparisons using shared benchmark modes.

Active set covers small decode, model decode, odd/general, and bounded prefill.
Latency is reported in milliseconds; SpeedUp = device reference / FlagGems-vllm
(higher is better). Timing, warmup and recording follow the CLI mode. Acceptance
uses the independent CPU oracle before timing.
"""

from types import SimpleNamespace

import pytest
import torch

import flaggems_vllm
from tests.test_quant_lightning_indexer import (
    call_vllm_ascend_indexer,
    check,
    data,
    invoke,
    make_vllm_ascend_case,
    require_vllm_ascend_indexer,
)

from . import base

pytestmark = [
    pytest.mark.quant_lightning_indexer,
    pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="Requires Ascend"),
]

# (tokens, heads, dimension, physical pages, page size, request table columns, topk)
ACTIVE_SET = [
    (1, 64, 128, 8, 32, 4, 32),
    (4, 64, 128, 40, 32, 32, 512),
    (4, 3, 33, 6, 8, 3, 7),
    (32, 64, 128, 40, 32, 8, 128),
]


def make_case(case):
    t, h, d, pages, block, cols, top = case
    # data()'s four-row fixture is extended here without changing the seed or
    # input distributions used by the correctness suite.
    args = data(t=1, h=h, d=d, pages=pages, block=block, top=top)
    generator = torch.Generator().manual_seed(2451 + t)
    args[0] = torch.randint(
        -7, 8, (t, h, d), generator=generator, dtype=torch.int8
    ).npu()
    args[2] = torch.rand((t, h), generator=generator).npu()
    args[3] = (torch.arange(2 * cols).reshape(2, cols) % pages).long().npu()
    args[4] = (torch.arange(t) % 2).long().npu()
    args[5] = torch.full((t,), cols * block * 4 - 1, dtype=torch.int64).npu()
    args[6] = torch.full((2,), cols * block, dtype=torch.int64).npu()
    args[8] = torch.rand((t, h, 1), generator=generator).npu()
    return args


def torch_reference(
    query,
    key_cache,
    weights,
    block_table,
    request_ids,
    positions,
    key_lengths,
    topk,
    compress_ratio=4,
    query_scale=None,
    key_scale_cache=None,
):
    """Independent FP32 device baseline for the all-valid benchmark workloads.

    The benchmark deliberately uses valid page/request IDs and C4 only; the
    functional suite's independent CPU oracle checks invalid metadata semantics.
    """
    assert compress_ratio == 4
    pages = block_table[request_ids.long()].long()
    keys = key_cache[pages].flatten(1, 2).float()
    q = query.float()
    if query_scale is not None:
        q = q * query_scale.reshape(*query.shape[:2], 1)
    if key_scale_cache is not None:
        keys = keys * key_scale_cache[pages].flatten(1, 2)
    dots = torch.einsum("thd,tcd->thc", q, keys)
    scores = (dots.relu() * weights[..., None]).sum(1)
    cap = keys.shape[1]
    limit = torch.minimum((positions + 1) // 4, key_lengths[request_ids.long()])
    valid = torch.arange(cap, device=query.device)[None, :] < limit[:, None]
    scores = scores.masked_fill(~valid, -torch.inf)
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    ids = order[:, : min(topk, cap)]
    picked = scores.gather(1, ids)
    ids = ids.masked_fill(~torch.isfinite(picked), -1).to(torch.int32)
    if topk > cap:
        ids = torch.cat(
            (
                ids,
                torch.full(
                    (query.shape[0], topk - cap),
                    -1,
                    dtype=torch.int32,
                    device=query.device,
                ),
            ),
            1,
        )
    return ids


_REFERENCE = SimpleNamespace(quant_lightning_indexer=torch_reference)


def _candidate(*args):
    return invoke(flaggems_vllm, args)


def _baseline(*args):
    return invoke(_REFERENCE, args)


class QuantLightningIndexerBenchmark(base.Benchmark):
    DEFAULT_SHAPES = ACTIVE_SET
    DEFAULT_SHAPE_DESC = "(tokens, heads, dimension, pages, block, columns, topk)"

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            args = make_case(shape)
            assert args[0].dtype == dtype
            # Independent CPU oracle checks both paths outside the timed region.
            check(args)
            check(args, mod=_REFERENCE)
            yield tuple(args)


def test_quant_lightning_indexer_benchmark():
    bench = QuantLightningIndexerBenchmark(
        op_name="quant_lightning_indexer",
        torch_op=_baseline,
        dtypes=[torch.int8],
    )
    bench.set_gems(_candidate)
    bench.run()


class QuantLightningIndexerVllmAscendBenchmark(base.Benchmark):
    DEFAULT_SHAPES = [(1,), (16,)]
    DEFAULT_SHAPE_DESC = "(tokens), heads=64, head_dim=128, TopK=32"

    def __init__(self, native, metadata_op):
        super().__init__(
            op_name="quant_lightning_indexer_vllm_ascend",
            torch_op=self.native_call,
            dtypes=[torch.int8],
        )
        self.native = native
        self.metadata_op = metadata_op

    def native_call(self, args, native_kwargs):
        return call_vllm_ascend_indexer(self.native, args, native_kwargs)

    @staticmethod
    def candidate_call(args, native_kwargs):
        return _candidate(*args)

    def get_input_iter(self, dtype):
        for (tokens,) in self.shapes:
            if not isinstance(tokens, int) or not 1 <= tokens <= 128:
                raise ValueError("Native indexer benchmark expects 1 to 128 tokens")
            args, kwargs = make_vllm_ascend_case(tokens, self.metadata_op)
            assert args[0].dtype == dtype
            expected = check(args, exact_ids=True)
            actual = self.native_call(args, kwargs).cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            # Native metadata is built once before the timed calls.
            yield args, {"native_kwargs": kwargs}


def test_quant_lightning_indexer_vllm_ascend_benchmark():
    native, metadata_op = require_vllm_ascend_indexer()
    bench = QuantLightningIndexerVllmAscendBenchmark(native, metadata_op)
    bench.set_gems(bench.candidate_call)
    bench.run()
