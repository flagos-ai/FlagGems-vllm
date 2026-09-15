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

"""Ascend performance benchmark for chunk_gated_delta_rule_fwd.

Compares two independent Ascend implementations of the chunk_gated_delta_rule_fwd:

- baseline: ``vllm_ascend.ops.triton.fla.chunk.chunk_gated_delta_rule_fwd``
  (Triton preprocess + AscendC ``_C_ascend`` fwd_h / fwd_o custom ops). It is
  invoked standalone, outside a vLLM engine, with neutral ForwardContext /
  PCP-group stand-ins and prebuilt chunk metadata (as serving builds in
  forward_prepare; without it the AscendC fwd_h under-allocates its chunk
  state buffer for varlen tails) — the same approach vllm-ascend's own op UT
  takes (``tests/ut/ops/a2/test_gdn_chunk_meta.py``). Requires an installed
  vllm-ascend; the benchmark is skipped otherwise. Note: this implementation
  casts q/k/w/u to bfloat16 internally, so float16 cases also measure its
  cast overhead.

Example::

    PYTHONPATH=src pytest -s benchmark/test_FLA/test_chunk_gated_delta_rule_ascend.py \
    --mode operator --warmup 3 --iter 10 \
    --dtypes bfloat16 --dtypes float16
"""

import pytest
import torch
import torch.nn.functional as F

import flaggems_vllm
from benchmark.base import Benchmark

try:
    from vllm_ascend.ops.triton.fla import chunk as _vllm_ascend_chunk
    from vllm_ascend.ops.triton.fla.utils import (
        prepare_chunk_indices as _vllm_ascend_prepare_chunk_indices,
        prepare_chunk_offsets as _vllm_ascend_prepare_chunk_offsets,
    )
    from vllm_ascend.ops.triton.triton_utils import (
        init_device_properties_triton as _vllm_ascend_init_device_properties,
    )
    from vllm_ascend.utils import enable_custom_op as _vllm_ascend_enable_custom_op

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

_ASCEND_FWD_MODULE = (
    "flaggems_vllm.runtime.backend._ascend.ops.chunk_gated_delta_rule_fwd"
)

BATCH_SIZE = 1
NUM_KV_HEADS = 16
NUM_HEADS = 32
KEY_DIM = 128
VALUE_DIM = 128

if VLLM_AVAILABLE:
    # The vllm-ascend forward reads the vLLM ForwardContext and the PCP group,
    # neither of which exists in a standalone benchmark process (both assert
    # when unset). Install neutral single-rank stand-ins — mirroring
    # vllm-ascend's own op UT — and make sure the _C_ascend custom ops load.
    _vllm_ascend_enable_custom_op()
    # Standalone processes must also seed the AICore counts their launchers
    # read (normally done by the vllm_ascend worker at startup).
    _vllm_ascend_init_device_properties()

    class _StandaloneForwardContext:
        attn_metadata = None

    class _StandalonePcpGroup:
        world_size = 1
        rank_in_group = 0

    _vllm_ascend_chunk.get_forward_context = lambda: _StandaloneForwardContext()
    _vllm_ascend_chunk.get_pcp_group = lambda: _StandalonePcpGroup()


# The vllm-ascend AscendC fwd_h op maps packed chunks from prebuilt
# chunk_indices; without them its h buffer is sized from the packed length
# alone, which under-allocates when sequences have tail chunks (out-of-bounds
# write, aicore exception). Serving always passes prebuilt metadata built in
# forward_prepare, so build it once per cu_seqlens here — same fields as the
_PREBUILT_META_CACHE = {}


def _vllm_ascend_prebuilt_meta(cu_seqlens):
    entry = _PREBUILT_META_CACHE.get(id(cu_seqlens))
    if entry is not None and entry[0] is cu_seqlens:
        return entry[1]

    chunk_size = 64
    cumsum_block_size = 1 << (((2**18) // (NUM_HEADS * chunk_size)) - 1).bit_length()
    chunk_indices = _vllm_ascend_prepare_chunk_indices(cu_seqlens, chunk_size)
    meta = type(
        "PrebuiltChunkMeta",
        (),
        {
            "block_indices_cumsum": _vllm_ascend_prepare_chunk_indices(
                cu_seqlens, cumsum_block_size
            ),
            "cu_seqlens_host": tuple(cu_seqlens.tolist()),
            "chunk_indices_chunk64": chunk_indices,
            "chunk_indices_chunk64_host": tuple(chunk_indices.flatten().tolist()),
            "chunk_offsets_chunk64": _vllm_ascend_prepare_chunk_offsets(
                cu_seqlens, chunk_size
            ),
            "update_chunk_offsets_chunk64": None,
            "final_chunk_indices_chunk64": None,
            "chunk_indices_large_block": _vllm_ascend_prepare_chunk_indices(
                cu_seqlens, 608 * 2
            ),
            "keep_meta": None,
            "cu_seqlens_kern": None,
        },
    )()
    _PREBUILT_META_CACHE[id(cu_seqlens)] = (cu_seqlens, meta)
    return meta


def _assert_top_level_ascend_provider() -> None:
    provider = flaggems_vllm.chunk_gated_delta_rule_fwd
    if provider.__module__ != _ASCEND_FWD_MODULE:
        raise RuntimeError(
            "the top-level chunk_gated_delta_rule_fwd is not registered from the "
            f"Ascend module: {provider.__module__}"
        )


def _vllm_ascend_chunk_gated_delta_rule_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    output_final_state,
    cu_seqlens,
):
    """Baseline: vLLM-Ascend FLA chunk forward (seven-tuple protocol)."""
    result = _vllm_ascend_chunk.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        prebuilt_meta=_vllm_ascend_prebuilt_meta(cu_seqlens),
    )
    return result[1], result[3]


def _ascend_chunk_gated_delta_rule_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    output_final_state,
    cu_seqlens,
):
    """Candidate: direct Ascend low-level forward (seven-tuple protocol)."""
    result = flaggems_vllm.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    return result[1], result[3]


class ChunkGatedDeltaRuleFwdAscendBenchmark(Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16, torch.float16]
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_SHAPES = [
        (65, 129),
        (147, 311, 566),
        (256, 513, 1024),
        (1025, 2048, 4097),
        (4096, 8192),
        (16384, 4096, 1024),
    ]
    DEFAULT_SHAPE_DESC = "packed seq lengths (B=1, Hg=16, H=32, K=V=128)"

    def set_more_shapes(self):
        return self.DEFAULT_SHAPES

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def get_input_iter(self, cur_dtype):
        for sequence_lengths in self.shapes:
            yield self._build_inputs(sequence_lengths, cur_dtype)

    def _build_inputs(self, sequence_lengths: tuple[int, ...], dtype: torch.dtype):
        device = flaggems_vllm.device
        torch.manual_seed(20260827)
        total_tokens = sum(sequence_lengths)

        q = F.normalize(
            torch.randn(
                BATCH_SIZE,
                total_tokens,
                NUM_KV_HEADS,
                KEY_DIM,
                device=device,
                dtype=torch.float32,
            ),
            p=2,
            dim=-1,
        ).to(dtype)
        k = F.normalize(
            torch.randn(
                BATCH_SIZE,
                total_tokens,
                NUM_KV_HEADS,
                KEY_DIM,
                device=device,
                dtype=torch.float32,
            ),
            p=2,
            dim=-1,
        ).to(dtype)
        v = torch.randn(
            BATCH_SIZE,
            total_tokens,
            NUM_HEADS,
            VALUE_DIM,
            device=device,
            dtype=torch.float32,
        ).to(dtype)
        # The scalar gate stays in float32 — the serving convention, The candidate
        # wrapper casts it down to the compute dtype.
        g = F.logsigmoid(
            torch.randn(
                BATCH_SIZE,
                total_tokens,
                NUM_HEADS,
                device=device,
                dtype=torch.float32,
            )
        )
        beta = torch.rand(
            BATCH_SIZE,
            total_tokens,
            NUM_HEADS,
            device=device,
            dtype=torch.float32,
        ).to(dtype)
        scale = KEY_DIM**-0.5

        boundaries = [0]
        for sequence_length in sequence_lengths:
            boundaries.append(boundaries[-1] + sequence_length)
        cu_seqlens = torch.tensor(boundaries, dtype=torch.long, device=device)
        # A non-None initial state keeps both measured implementations on the
        # same initial-state route (serving always passes one); the initial_state=None flavor is
        # covered by the accuracy tests.
        initial_state = 0.01 * torch.randn(
            len(sequence_lengths),
            NUM_HEADS,
            KEY_DIM,
            VALUE_DIM,
            device=device,
            dtype=torch.float32,
        )

        return (
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            True,
            cu_seqlens,
        )


@pytest.mark.chunk_gated_delta_rule_fwd
@pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend" or flaggems_vllm.device != "npu",
    reason="Ascend NPU backend required",
)
@pytest.mark.skipif(
    not VLLM_AVAILABLE,
    reason="requires vllm-ascend FLA ops (vllm_ascend.ops.triton.fla)",
)
def test_perf_chunk_gated_delta_rule_fwd_ascend():
    _assert_top_level_ascend_provider()
    bench = ChunkGatedDeltaRuleFwdAscendBenchmark(
        op_name="chunk_gated_delta_rule_fwd",
        torch_op=_vllm_ascend_chunk_gated_delta_rule_fwd,
    )
    bench.set_gems(_ascend_chunk_gated_delta_rule_fwd)
    bench.run()
