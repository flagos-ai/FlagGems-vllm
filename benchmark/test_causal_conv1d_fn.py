# Copyright 2026- Xcoresigma Technology Co., Ltd
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

"""Benchmark for the Ascend causal-conv1d-fn kernel.

gems op: ``flaggems_vllm.causal_conv1d_fn`` (Ascend vendor registrar entry).
torch baseline: per-sequence ``F.conv1d`` loop with the same signature.
SpeedUp = latency_torch_baseline / latency_flaggems (repo convention).
"""

import itertools

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("torch_npu")
import torch.nn.functional as F  # noqa: E402

import flaggems_vllm  # noqa: E402

from . import base  # noqa: E402

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend" or not torch.npu.is_available(),
    reason="causal_conv1d_fn requires an Ascend NPU",
)

PAD_SLOT_ID = -1
WIDTH = 4
PADDING = 3

# (batch, seqlen, dim) varlen shapes
SHAPES = [
    (4, 8, 64),
    (4, 249, 4096),
    (10, 4096, 4096),
]


class CausalConv1dBenchmark(base.GenericBenchmark):
    """Use the common benchmark runner with fixed varlen shapes."""

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = SHAPES
        self.shape_desc = "batch-seqlen-dim"


def _varlen_inputs(batch, seqlen, dim, dtype, device):
    torch.manual_seed(1000 + batch + seqlen + dim)
    padded_batch = batch + PADDING
    seq = seqlen // padded_batch
    rem = seqlen - seq * (padded_batch - 1)
    seqlens = [rem] + [seq] * (padded_batch - 1)
    query_start_loc = torch.tensor(
        [0] + list(itertools.accumulate(seqlens)), dtype=torch.int32
    ).to(device)

    # x: (1, dim, seqlen) sliced from a wider buffer, then squeezed to (dim, seqlen)
    x = torch.randn(1, seqlen, 4096 + dim + 64, device=device, dtype=dtype)
    x = x.transpose(1, 2)[:, 4096 : 4096 + dim, :]
    weight = torch.randn(dim, WIDTH, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype)
    conv_states = torch.randn(
        batch * 10, WIDTH - 1, dim, device=device, dtype=dtype
    ).transpose(1, 2)
    state_indices = torch.arange(batch, dtype=torch.int32, device=device)
    cache_indices = torch.concat(
        [
            state_indices,
            torch.full((PADDING,), PAD_SLOT_ID, dtype=torch.int32, device=device),
        ],
        dim=-1,
    )
    has_initial_state = torch.randint(
        0, 2, (padded_batch,), dtype=torch.bool, device=device
    )
    return (
        x.squeeze(0),
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        seqlens,
    )


def _causal_conv1d_torch(
    x,
    weight,
    bias,
    conv_states,
    query_start_loc,
    cache_indices,
    has_initial_state,
    seqlens,
    activation="silu",
    pad_slot_id=PAD_SLOT_ID,
):
    """Per-sequence F.conv1d loop baseline; benchmark-only reference."""
    dim, _ = x.shape
    x3 = x.unsqueeze(0)
    out_chunks = []
    for i, seqlen_i in enumerate(seqlens):
        if cache_indices[i] == pad_slot_id:
            continue
        x_s = x3[:, :, :seqlen_i]
        x3 = x3[:, :, seqlen_i:]
        initial = (
            conv_states[cache_indices[i]].unsqueeze(0) if has_initial_state[i] else None
        )
        out_s = causal_conv1d_seq_ref(x_s, weight, bias, initial, activation)
        out_chunks.append(out_s)
    return torch.cat(out_chunks, dim=2)


def causal_conv1d_seq_ref(x, weight, bias, initial_states, activation):
    """Single-sequence torch reference for one (1, dim, seqlen) chunk."""
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


def _causal_conv1d_gems(
    x,
    weight,
    bias,
    conv_states,
    query_start_loc,
    cache_indices,
    has_initial_state,
    seqlens,
    activation="silu",
    pad_slot_id=PAD_SLOT_ID,
):
    return flaggems_vllm.causal_conv1d_fn(
        x,
        weight,
        bias=bias,
        conv_states=conv_states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation=activation,
        pad_slot_id=pad_slot_id,
    )


def _conv1d_input_fn(shape, dtype, device):
    batch, seqlen, dim = shape
    yield _varlen_inputs(batch, seqlen, dim, dtype, device)


def test_causal_conv1d_fn_perf():
    bench = CausalConv1dBenchmark(
        input_fn=_conv1d_input_fn,
        op_name="causal_conv1d_fn",
        torch_op=_causal_conv1d_torch,
        gems_op=_causal_conv1d_gems,
        dtypes=[torch.bfloat16],
    )
    bench.run()
