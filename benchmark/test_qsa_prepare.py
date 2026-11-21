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

On CUDA builds the baseline is the upstream vLLM operator, driven with
identical inputs. Ascend has no equivalent fused operator, so the Ascend
baseline composes the vllm-ascend operators that cover the same work: the main
split/norm/rope kernel, ``_gemma_rmsnorm``, ``apply_qsa_rope``,
``qsa_compress_groups_with_ratio`` and ``qsa_store_cache_rows`` for every cache
store (main K/V per head, raw ring, ring rope tail). Only metadata preparation
and the ``.contiguous()`` the vendor kernel requires stay on torch - the same
split the cross-validated baseline uses. Set ``QSA_ASCEND_BASELINE_IO=torch``
to write those stores with torch indexing instead (the older, slower form).
That row is still an ***approximate*** baseline; the reported speedup uses it
as the denominator and is labelled ``baseline=composed``.
"""

from __future__ import annotations

import os

import pytest
import torch

import flaggems_vllm
from flaggems_vllm import qsa_prepare
from tests.qsa_prepare_reference import build_ascend_baseline
from tests.test_qsa_prepare import (
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
VENDOR = flaggems_vllm.vendor_name

# Buffers with a float8 indexer dtype cannot even be zero-initialised on a
# backend without native float8 support, so those shapes are dropped there.
HAS_NATIVE_FP8 = hasattr(torch, "float8_e4m3fn") and (
    flaggems_vllm.SUPPORTED_FP8_DTYPE == torch.float8_e4m3fn
)


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

if not HAS_NATIVE_FP8:
    SHAPES = [shape for shape in SHAPES if shape[3].get("indexer_dtype") is not FP8]

# ------------------------------------------------------------------ baselines
HAS_VLLM_NATIVE = False
try:  # upstream vLLM ships the fused operator on CUDA builds only
    from vllm.models.qwen4_exp.nvidia.ops.qsa_prepare import (
        qsa_prepare as _vllm_qsa_prepare,
    )

    HAS_VLLM_NATIVE = True
except Exception:  # noqa: BLE001 - any import failure means "not available"
    _vllm_qsa_prepare = None


_BASELINE_IO = os.environ.get("QSA_ASCEND_BASELINE_IO", "vendor").strip().lower()
_composed_baseline = build_ascend_baseline(_BASELINE_IO != "torch")

if VENDOR == "ascend":
    # vLLM's native qsa_prepare kernel is CUDA-only, and newer vendor builds drop
    # _gemma_rmsnorm, so the composition may be unavailable: report absolute
    # latency instead of falling back to the unrunnable native baseline.
    if _composed_baseline is not None:
        BASELINE_KIND = "composed" if _BASELINE_IO != "torch" else "composed-torch"
        BASELINE_OP = _composed_baseline
    else:
        BASELINE_KIND = "none"
        BASELINE_OP = None
elif HAS_VLLM_NATIVE:
    BASELINE_KIND = "native"
    BASELINE_OP = _vllm_qsa_prepare
elif _composed_baseline is not None:
    BASELINE_KIND = "composed" if _BASELINE_IO != "torch" else "composed-torch"
    BASELINE_OP = _composed_baseline
else:
    BASELINE_KIND = "none"
    BASELINE_OP = None


class QsaPrepareBenchmark(base.GenericBenchmark):
    """Use the common runner with the QSA prepare batch layouts."""

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = list(SHAPES)
        self.shape_desc = "seq_lens / query_lens / history_lens"

    def record_shapes(self, *args, **kwargs):
        """Report the batch layout rather than every operand."""

        query = args[0]
        state_cache = args[8]
        kv_cache = kwargs["main_kv_cache"]
        reported = {
            "requests": int(state_cache.shape[0]),
            "tokens": int(query.shape[0]),
            "query": list(query.shape),
            "main_qkv": list(kwargs["main_qkv"].shape),
            "baseline": BASELINE_KIND,
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
    """The shipped kernel against the vendor operator or its composition."""

    bench = QsaPrepareBenchmark(
        input_fn=_input_fn,
        op_name="qsa_prepare",
        torch_op=BASELINE_OP,
        gems_op=qsa_prepare,
        dtypes=[torch.bfloat16],
    )
    bench.run()


pytestmark = pytest.mark.skipif(
    flaggems_vllm.device == "cpu",
    reason="QSA prepare benchmark requires an accelerator",
)
