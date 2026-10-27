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

"""Performance of the QSA fused prepare operator.

The baseline is the upstream vLLM operator, driven with identical inputs.  It
only exists in the NVIDIA build today; elsewhere the benchmark still runs and
reports the frozen-baseline ratio instead of a local speedup.
"""

from __future__ import annotations

import pytest
import torch

import flaggems_vllm
from flaggems_vllm import qsa_prepare
from tests.test_qsa_prepare import (
    COMPRESS_RATIO,
    HEAD_DIM,
    MROPE_SECTION,
    QsaPrepareCase,
)

from . import base

VENDOR = flaggems_vllm.vendor_name
# Only the NVIDIA build ships an upstream qwen4_exp implementation today.
VLLM_NATIVE_VENDORS = {"nvidia"}

if VENDOR in VLLM_NATIVE_VENDORS:
    from vllm.models.qwen4_exp.nvidia.ops.qsa_prepare import (
        qsa_prepare as _vllm_native_qsa_prepare,
    )


def _get_baseline_op():
    """Return the vendor-native implementation, or None when there is none."""

    if VENDOR in VLLM_NATIVE_VENDORS:
        return _vllm_native_qsa_prepare
    return None


# Each entry is a batch scenario, not a tensor size: _input_fn expands one entry
# into about two dozen operands whose shapes are unrelated to each other, from
# (128,) norm weights up to a (32768, 64) rope table. The head counts, head dims,
# block size and compress ratio are fixed by the model, so what varies between
# entries is the batch layout plus the few knobs that select a different code
# path (state_size and the indexer dtype).

FP8 = torch.float8_e4m3fn


def _single(tokens, **overrides):
    """One single-request prefill of `tokens` queries with no history."""

    return ((tokens,), (tokens,), (0,), overrides)


def _decode(n_requests, seq_len=512, history=8, **overrides):
    """One pure decode step: `n_requests` single-token requests behind histories."""

    return (
        (seq_len,) * n_requests,
        (1,) * n_requests,
        (history,) * n_requests,
        overrides,
    )


def _steady_state(n_decode, n_prefill, chunk=512, decode_seq=512):
    """A steady-state step: decode requests and prefill chunks in one batch."""

    return (
        (decode_seq,) * n_decode + (chunk,) * n_prefill,
        (1,) * n_decode + (chunk,) * n_prefill,
        (8,) * n_decode + (0,) * n_prefill,
        {},
    )


SHAPES = [
    _steady_state(53, 1, chunk=16331, decode_seq=32768),
    _steady_state(54, 1, chunk=16330, decode_seq=32768),
    _steady_state(56, 1, chunk=16328, decode_seq=32768),
    _steady_state(32, 1, chunk=16352, decode_seq=32768),
    _steady_state(16, 1, chunk=16368, decode_seq=32768),
    _steady_state(8, 1, chunk=16376, decode_seq=32768),
    _single(16384),
    _single(6173),
    _single(2372),
    _decode(53, seq_len=32768),
    _decode(54, seq_len=32768),
    ((260, 259, 138), (1, 1, 37), (8, 8, 8), {}),
    _decode(16, state_size=8),
    _single(4097, indexer_dtype=FP8),
]


class QsaPrepareBenchmark(base.GenericBenchmark):
    """Use the common runner with the QSA prepare batch layouts."""

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = list(SHAPES)
        self.shape_desc = "seq_lens / query_lens / history_lens"

    def init_user_config(self):
        super().init_user_config()
        if self.torch_op is None:
            # No vendor-native kernel on this platform: timing it would raise,
            # and a gems-vs-gems ratio says nothing. compared_speedup is still
            # produced from the frozen baseline outside this metric list.
            self.to_bench_metrics = [
                metric
                for metric in self.to_bench_metrics
                if metric not in ("latency_base", "speedup")
            ]

    def record_shapes(self, *args, **kwargs):
        """Report the batch layout rather than every operand.

        The tensor shapes stay as torch.Size on purpose: the frozen-baseline
        lookup matches on the multiset of shapes found here, and the seven
        T=16384 layouts share their first two tensors, so state_cache (it carries
        num_requests) and main_kv_cache are what make every entry unique.
        """

        query = args[0]
        state_cache = args[8]
        kv_cache = kwargs["main_kv_cache"]
        reported = {
            "requests": int(state_cache.shape[0]),
            "tokens": int(query.shape[0]),
            "query": query.shape,
            "state_cache": state_cache.shape,
            "main_qkv": kwargs["main_qkv"].shape,
            "main_kv_cache": kv_cache.shape,
        }
        if int(state_cache.shape[1]) != 4:
            reported["state_size"] = int(state_cache.shape[1])
        if kv_cache.dtype == FP8:
            reported["fp8"] = True
        return reported


def _input_fn(shape, dtype, device):
    """One batch scenario per entry, buffers allocated once."""

    seq_lens, query_lens, history_lens, overrides = shape
    _ = dtype
    case = QsaPrepareCase(
        seq_lens=list(seq_lens),
        query_lens=list(query_lens),
        history_lens=list(history_lens),
        device=device,
        **overrides,
    )
    yield (
        *case._launch_args(case.outputs()),
        dict(
            compress_ratio=COMPRESS_RATIO,
            mrope_section=MROPE_SECTION,
            rope_pos_offset=HEAD_DIM,
            **case.main,
        ),
    )


@pytest.mark.qsa_prepare
def test_qsa_prepare():
    """The shipped kernel against the upstream implementation."""

    bench = QsaPrepareBenchmark(
        input_fn=_input_fn,
        op_name="qsa_prepare",
        torch_op=_get_baseline_op(),
        gems_op=qsa_prepare,
        dtypes=[torch.bfloat16],
    )
    bench.run()


pytestmark = pytest.mark.skipif(
    flaggems_vllm.device == "cpu" or not torch.cuda.is_available(),
    reason="QSA prepare benchmark requires an accelerator",
)
