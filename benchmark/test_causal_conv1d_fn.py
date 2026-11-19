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
from baseline import native_api  # noqa: E402

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

    # x follows the operator contract: channel-first [dim, seqlen], contiguous.
    # The kernel consumes it directly (fused transpose) instead of copying to a
    # token-major buffer first.
    x = torch.randn(dim, seqlen, device=device, dtype=dtype)
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
        x,
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


def _native_fn():
    """The platform's native causal_conv1d_fn, with its layout convention.

    ``native_api`` keeps the vendor split in one place: Ascend resolves the
    AscendC kernel (``fla_npu``) or vllm-ascend's entry, every other backend
    resolves vLLM's own op. Only the AscendC kernel uses kernel layouts.
    """
    return native_api.causal_conv1d_fn()


NATIVE_FN, NATIVE_FN_SOURCE = _native_fn()
_fla_npu_probe = None


def _native_available():
    """Whether the resolved native entry actually runs here.

    AscendC needs one live call: importing it proves nothing, and the
    argument layouts have to be the kernel's. The Python entries only need the
    import.
    """
    global _fla_npu_probe
    if _fla_npu_probe is None:
        if NATIVE_FN is None:
            _fla_npu_probe = (False, "no native causal_conv1d_fn on this platform")
        elif NATIVE_FN_SOURCE != "fla_npu.ops.ascendc":
            _fla_npu_probe = (True, None)
        else:
            try:
                torch.manual_seed(0)
                dev = flaggems_vllm.device
                t, dim = 64, 128
                dtype = torch.bfloat16
                x = torch.randn(t, dim, device=dev, dtype=dtype)
                weight = torch.randn(WIDTH, dim, device=dev, dtype=dtype)
                states = torch.randn(8, WIDTH - 1, dim, device=dev, dtype=dtype)
                NATIVE_FN(
                    x,
                    weight,
                    None,
                    conv_states=states,
                    query_start_loc=torch.tensor([0, t], dtype=torch.int32, device=dev),
                    cache_indices=torch.tensor([1], dtype=torch.int32, device=dev),
                    has_initial_state=torch.ones(1, dtype=torch.bool, device=dev),
                    activation="silu",
                    pad_slot_id=PAD_SLOT_ID,
                    null_block_id=PAD_SLOT_ID,
                )
                torch.npu.synchronize()
                _fla_npu_probe = (True, None)
            except Exception as exc:
                _fla_npu_probe = (False, f"{type(exc).__name__}: {exc}")
    return _fla_npu_probe


def _causal_conv1d_native(
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
    """Native baseline, adapted to the resolved entry's layouts.

    The AscendC kernel takes vLLM's kernel layouts, which differ from this
    operator's contract in three places: ``x`` is token-major
    ``[tokens, channels]``, ``weight`` is ``[width, channels]`` and
    ``conv_states`` is ``[cache, state_len, channels]``. With those
    conversions its output and updated state are bit-identical to this repo's
    kernel. Every other entry takes the operator layouts directly.
    """
    _ = seqlens
    if NATIVE_FN_SOURCE == "fla_npu.ops.ascendc":
        return NATIVE_FN(
            x.transpose(0, 1),
            weight.transpose(0, 1).contiguous(),
            bias,
            conv_states=conv_states.transpose(1, 2),
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation=activation,
            pad_slot_id=pad_slot_id,
            null_block_id=PAD_SLOT_ID,
        )
    return NATIVE_FN(
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


def _load_vllm_ascend_fn():
    """vllm-ascend's own entry: Ascend only, and only as a fallback."""
    if flaggems_vllm.vendor_name != "ascend":
        return None
    try:
        from vllm_ascend.ops.causal_conv1d import causal_conv1d_fn as native
    except Exception:  # ImportError, or an older tree that raises on import
        return None
    return native


VLLM_ASCEND_FN = _load_vllm_ascend_fn()


def _causal_conv1d_vllm_ascend(
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
    """vllm-ascend fallback. It also updates conv_states in place.

    On parts other than 310P this is a per-sequence torch implementation, so
    the SpeedUp compares against a reference path, not a kernel.
    """
    _ = seqlens
    return VLLM_ASCEND_FN(
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
    # Prefer the AscendC kernel, then vllm-ascend's own API, then the local
    # per-sequence F.conv1d loop.
    if _native_available()[0]:
        baseline = _causal_conv1d_native
    elif VLLM_ASCEND_FN is not None:
        baseline = _causal_conv1d_vllm_ascend
    else:
        baseline = _causal_conv1d_torch
    if baseline is not _causal_conv1d_native:
        print(
            "[causal_conv1d_fn] native baseline unavailable: "
            f"{_native_available()[1]}"
        )
    bench = CausalConv1dBenchmark(
        input_fn=_conv1d_input_fn,
        op_name="causal_conv1d_fn",
        torch_op=baseline,
        gems_op=_causal_conv1d_gems,
        dtypes=[torch.bfloat16],
    )
    bench.run()
