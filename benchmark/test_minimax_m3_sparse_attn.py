# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Latency benchmark for the MiniMax M3 sparse attention shim.

The baseline is vLLM's own MiniMax M3 operator, per this repository's
convention (``SpeedUp = latency_vllm_baseline / latency_flaggems``). It is
imported lazily because it only exists in vLLM builds that ship the packed
4D KV layout ``[num_blocks, num_kv_heads, 128, 2 * head_dim]``; older builds
use a 5D layout and are skipped rather than silently compared.

Sparse-attention latency depends heavily on how clustered the selected block
ids are, so each shape is run under both a compact and a scattered top-k
distribution and the span statistics are reported as an extra metric: a
speedup is not interpretable without the distribution that produced it.
"""

import pytest
import torch

import flaggems_vllm
from tests.test_minimax_m3_sparse_attn import (
    _native_takes_packed_pages,
    make_decode_inputs,
    make_inputs,
    topk_span_stats,
)

from . import base

native = pytest.importorskip(
    "vllm.models.minimax_m3.common.ops.sparse_attn",
    reason="requires the vLLM MiniMax M3 packed-KV operator",
)

pytestmark = [
    pytest.mark.skipif(
        flaggems_vllm.vendor_name != "nvidia" or not torch.cuda.is_available(),
        reason="requires an NVIDIA GPU",
    ),
    # The module imports on older vLLM too, so importorskip above is not
    # enough: those builds only fail once called, on kv_cache.stride(4).
    pytest.mark.skipif(
        not _native_takes_packed_pages(),
        reason="installed vLLM operator predates the packed 4D KV layout",
    ),
]

_QLENS = [512, 2048, 4096, 8192]
_DECODE_REQS = [1, 16, 64, 128]
_DISTRIBUTIONS = ["compact", "scattered"]


class _MSABenchmark(base.Benchmark):
    """Shared plumbing for the two MSA phases.

    ``BenchmarkMetrics`` has a fixed field set and the run loop only computes
    ``gbps``/``tflops`` by name, so the selected-index span cannot be added as
    a custom metric. It rides along in ``shape_detail`` instead, which takes
    arbitrary values -- the span is the most predictive input property for this
    operator (the same shape and top-k count can differ severalfold between a
    clustered and a dispersed selection), so a latency row without it is not
    interpretable.
    """

    DEFAULT_DTYPES = [torch.bfloat16]

    def set_shapes(self, shape_file_path=None):
        self.shapes = [(n, dist) for n in self._sizes for dist in _DISTRIBUTIONS]

    def record_shapes(self, *args, **kwargs):
        stats = topk_span_stats(args[2])
        return (
            super().record_shapes(*args, **kwargs),
            f"span_median={stats.get('span_median')}",
            f"redundancy_median={stats.get('redundancy_median')}",
        )


class MinimaxM3SparseAttnBenchmark(_MSABenchmark):
    DEFAULT_SHAPE_DESC = "query_len, topk_distribution"
    _sizes = _QLENS

    def get_input_iter(self, dtype):
        for qlen, mode in self.shapes:
            args = make_inputs(qlens=(qlen,), dtype=dtype, mode=mode, context=16384)
            yield (*args, {"output": torch.empty_like(args[0])})


class MinimaxM3SparseAttnDecodeBenchmark(_MSABenchmark):
    DEFAULT_SHAPE_DESC = "num_reqs, topk_distribution"
    _sizes = _DECODE_REQS

    def get_input_iter(self, dtype):
        for num_reqs, mode in self.shapes:
            args = make_decode_inputs(
                num_reqs=num_reqs, dtype=dtype, mode=mode, context=16384
            )
            yield (
                *args[:7],
                {"output": torch.empty_like(args[0]), "decode_query_len": args[7]},
            )


@pytest.mark.minimax_m3_sparse_attn
def test_perf_minimax_m3_sparse_attn():
    bench = MinimaxM3SparseAttnBenchmark(
        op_name="minimax_m3_sparse_attn",
        torch_op=native.minimax_m3_sparse_attn,
        gems_op=flaggems_vllm.minimax_m3_sparse_attn,
    )
    bench.run()


@pytest.mark.minimax_m3_sparse_attn_decode
def test_perf_minimax_m3_sparse_attn_decode():
    bench = MinimaxM3SparseAttnDecodeBenchmark(
        op_name="minimax_m3_sparse_attn_decode",
        torch_op=native.minimax_m3_sparse_attn_decode,
        gems_op=flaggems_vllm.minimax_m3_sparse_attn_decode,
    )
    bench.run()
