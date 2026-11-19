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

"""Benchmark for the Ascend causal-conv1d-update (decode) kernel.

gems op: ``flaggems_vllm.causal_conv1d_update`` (Ascend vendor registrar entry).
torch baseline: batched ``F.conv1d`` reference with the same signature.
SpeedUp = latency_torch_baseline / latency_flaggems (repo convention).
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("torch_npu")
import torch.nn.functional as F  # noqa: E402

import flaggems_vllm  # noqa: E402

from . import base  # noqa: E402

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend" or not torch.npu.is_available(),
    reason="causal_conv1d_update requires an Ascend NPU",
)

PAD_SLOT_ID = -1
WIDTH = 4
PADDING = 5

# (batch, seqlen, dim) decode shapes
SHAPES = [
    (3, 1, 2048 + 16),
    (64, 1, 4096),
    (64, 3, 4096),
]


class CausalConv1dUpdateBenchmark(base.GenericBenchmark):
    """Use the common benchmark runner with fixed decode shapes."""

    def set_shapes(self, shape_file_path=None):
        _ = shape_file_path
        self.shapes = SHAPES
        self.shape_desc = "batch-seqlen-dim"


def _update_inputs(batch, seqlen, dim, dtype, device):
    torch.manual_seed(1000 + batch + seqlen + dim)
    padded_batch = batch + PADDING
    total_entries = 10 * batch

    # x: (padded_batch, dim, seqlen) channel-last, as the kernel expects
    x = torch.randn(padded_batch, seqlen, dim, device=device, dtype=dtype).transpose(
        1, 2
    )
    weight = torch.randn(dim, WIDTH, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype)
    conv_states = torch.randn(
        total_entries, WIDTH - 1, dim, device=device, dtype=dtype
    ).transpose(1, 2)
    state_indices = torch.randperm(total_entries, dtype=torch.int32, device=device)[
        :batch
    ]
    cache_indices = torch.concat(
        [
            state_indices,
            torch.full((PADDING,), PAD_SLOT_ID, dtype=torch.int32, device=device),
        ],
        dim=0,
    )
    return x, conv_states, weight, bias, cache_indices, state_indices


def _causal_conv1d_update_torch(
    x, conv_states, weight, bias, cache_indices, state_indices, activation="silu"
):
    """Batched F.conv1d reference; benchmark-only baseline."""
    batch = state_indices.size(0)
    dtype_in = x.dtype
    x_s = x[:batch]
    conv_state = conv_states[state_indices]
    x_new = torch.cat([conv_state, x_s], dim=-1).to(weight.dtype)
    conv_state.copy_(x_new[:, :, -conv_state.shape[-1] :])
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=weight.shape[0])[
        :, :, -x_s.shape[-1] :
    ]
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


def _causal_conv1d_update_gems(
    x, conv_states, weight, bias, cache_indices, state_indices, activation="silu"
):
    _ = state_indices
    return flaggems_vllm.causal_conv1d_update(
        x,
        conv_states,
        weight,
        bias,
        activation=activation,
        conv_state_indices=cache_indices,
        pad_slot_id=PAD_SLOT_ID,
    )


def _load_vllm_ascend_update():
    """vllm-ascend's native causal_conv1d_update, or None when unavailable."""
    try:
        from vllm_ascend.ops.causal_conv1d import causal_conv1d_update as native
    except Exception:  # ImportError, or an older tree that raises on import
        return None
    return native


VLLM_ASCEND_UPDATE = _load_vllm_ascend_update()


def _causal_conv1d_update_vllm_ascend(
    x, conv_states, weight, bias, cache_indices, state_indices, activation="silu"
):
    """vllm-ascend native baseline. It also updates conv_states in place.

    The native API derives the feature dim from ``x.shape[-1]``, so it wants
    channel-last input; the benchmark builds channel-first tensors.

    Note: on parts other than 310P, vllm-ascend ships this entry as a
    per-sequence torch implementation, so the SpeedUp compares against the
    vendor API's *reference* path rather than a vendor kernel.
    """
    _ = state_indices
    return VLLM_ASCEND_UPDATE(
        x.transpose(1, 2).contiguous(),
        conv_states,
        weight,
        bias,
        activation=activation,
        conv_state_indices=cache_indices,
        pad_slot_id=PAD_SLOT_ID,
    )


def _update_input_fn(shape, dtype, device):
    batch, seqlen, dim = shape
    yield _update_inputs(batch, seqlen, dim, dtype, device)


def test_causal_conv1d_update_perf():
    # Prefer vllm-ascend's native API as the baseline; fall back to the local
    # batched F.conv1d reference when it is not installed.
    baseline = (
        _causal_conv1d_update_vllm_ascend
        if VLLM_ASCEND_UPDATE is not None
        else _causal_conv1d_update_torch
    )
    bench = CausalConv1dUpdateBenchmark(
        input_fn=_update_input_fn,
        op_name="causal_conv1d_update",
        torch_op=baseline,
        gems_op=_causal_conv1d_update_gems,
        dtypes=[torch.bfloat16],
    )
    bench.run()
