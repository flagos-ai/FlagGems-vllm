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

"""Performance of the Qwen4 QSA fused prepare operator.

The baseline is the upstream vLLM operator, driven with identical inputs.
"""

from __future__ import annotations

import pytest
import torch
from vllm.models.qwen4_exp.nvidia.ops.qsa_prepare import qsa_prepare as vllm_qsa_prepare

import flaggems_vllm
from flaggems_vllm.ops.qwen4.qsa_prepare import qwen_qsa_prepare
from tests.test_qwen4_qsa_prepare import (
    COMPRESS_RATIO,
    HEAD_DIM,
    MROPE_SECTION,
    QsaPrepareCase,
)

from . import base

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
    """A steady-state step: decode requests and prefill chunks in one batch.

    This is the mixture vLLM's scheduler produces by default (chunked prefill
    with a shared token budget), so the two kinds of request contend inside the
    same launch.
    """

    return (
        (decode_seq,) * n_decode + (chunk,) * n_prefill,
        (1,) * n_decode + (chunk,) * n_prefill,
        (8,) * n_decode + (0,) * n_prefill,
        {},
    )


SHAPES = [
    # Single-request prefill sweep. 512 is the weakest point of the whole table;
    # 1024 is where the main-attention path switches to token tiling.
    _single(37),
    _single(512),
    _single(1024),
    _single(4097),
    _single(8192),
    # Upstream test_qsa_prepare.py's MIXED_BATCH: two decode steps and one
    # 37-token prefill running concurrently, all behind 8 tokens of history.
    # The token count alone does not describe this case — the K path decouples
    # from it, since compressed_cache scales with sum(seq_lens) while the K path
    # runs one program per request.
    ((260, 259, 138), (1, 1, 37), (8, 8, 8), {}),
    # Steady-state mixes at two concurrency levels. These are the layouts where a
    # decode request's single-token commit shares the launch with full-width
    # prefill chunks.
    _steady_state(8, 2),
    _steady_state(48, 4),
    # A pure decode step: sixteen single-token requests behind 512-token
    # histories. Here the decoupling is at its extreme — sum(seq_lens) is 8192
    # and compressed_cache holds 512 blocks, while T is only 16 and every path in
    # the grid runs at its lowest program count.
    _decode(16),
    # The same layout with state_size 8. The ring commit writes
    # min(query_len, STATE_SIZE) rows, so at query_len=1 a larger state_size only
    # widens the vector load that is masked down to a single live row — this is
    # the worst case for that change, and 8 is the other value the model uses.
    _decode(16, state_size=8),
    # An fp8 indexer dtype, which makes the paged KV cache half as wide. The
    # large single-request layout is where that traffic matters most.
    _single(4097, indexer_dtype=FP8),
    # The ceilings vLLM's defaults put on one step. On this class of GPU (its
    # >= 70 GiB, non-A100 tier) max_num_seqs is 1024, and max_num_batched_tokens
    # is 8192 for an OpenAI server or 16384 for offline throughput, so the
    # single-request 8192 above is already the serving budget.
    _decode(1024),
    # Both ceilings at once: 1010 + 14 = 1024 sequences carrying 8178 tokens,
    # which is as close to the 8192 serving budget as max_num_seqs allows.
    _steady_state(1010, 14),
    _single(16384),
]


class Qwen4QsaPrepareBenchmark(base.GenericBenchmark):
    """Use the common runner with the QSA prepare batch layouts."""

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = list(SHAPES)
        self.shape_desc = "seq_lens / query_lens / history_lens"

    def record_shapes(self, *args, **kwargs):
        """Report the batch layout rather than every operand.

        The kernel takes sixteen tensors plus twenty keyword operands, so the
        default report would print all of their sizes and run past the width of
        the table. These describe the layout the shape stands for, plus the two
        knobs that select a different code path when they leave their default.
        """

        query = args[0]
        # state_cache is the first operand whose leading dimension is the
        # request count rather than the token count.
        state_cache = args[8]
        kv_cache = kwargs["main_kv_cache"]
        reported = {
            "requests": int(state_cache.shape[0]),
            "tokens": int(query.shape[0]),
            "query": list(query.shape),
            "main_qkv": list(kwargs["main_qkv"].shape),
        }
        if int(state_cache.shape[1]) != 4:
            reported["state_size"] = int(state_cache.shape[1])
        if kv_cache.dtype == FP8:
            reported["fp8"] = True
        return reported


def _input_fn(shape, dtype, device):
    """One batch scenario per entry, buffers allocated once.

    The operator writes its outputs in place, so the buffers are prepared here
    rather than inside the timed callable.
    """

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


@pytest.mark.qwen4_qsa_prepare
def test_qwen_qsa_prepare():
    """The shipped kernel against the upstream implementation."""

    bench = Qwen4QsaPrepareBenchmark(
        input_fn=_input_fn,
        op_name="qwen_qsa_prepare",
        torch_op=vllm_qsa_prepare,
        gems_op=qwen_qsa_prepare,
        dtypes=[torch.bfloat16],
    )
    bench.run()


pytestmark = pytest.mark.skipif(
    flaggems_vllm.device == "cpu" or not torch.cuda.is_available(),
    reason="Qwen4 QSA prepare benchmark requires an accelerator",
)
