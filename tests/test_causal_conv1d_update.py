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

"""Correctness coverage for the Ascend causal-conv1d-update (decode) kernel.

The kernel under test is the Ascend TLE-DSA port living in
``flaggems_vllm/runtime/backend/_ascend/ops/causal_conv1d_update.py`` and
exported as ``flaggems_vllm.causal_conv1d_update`` via the vendor op registrar.

Torch references live in this test module only. The production wrapper keeps
its vendor-port torch compute calls (dtype cast / contiguous) and is
exercised as-is.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("torch_npu")
import torch.nn.functional as F  # noqa: E402

import flaggems_vllm  # noqa: E402

from .conftest import QUICK_MODE  # noqa: E402

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend" or not torch.npu.is_available(),
    reason="causal_conv1d_update requires an Ascend NPU",
)

device = "npu"
PAD_SLOT_ID = -1

if QUICK_MODE:
    BATCH = [3]
    WITH_PADDING = [True]
    DIM = [2048 + 16]
    SEQS = [1]
    WIDTH = [3]
    HAS_BIAS = [True]
    SILU = [True]
    ITYPES = [torch.bfloat16]
else:
    BATCH = [3, 64]
    WITH_PADDING = [True, False]
    DIM = [2048 + 16, 4096]
    SEQS = [1, 3]
    WIDTH = [3, 4]
    HAS_BIAS = [False, True]
    SILU = [True]
    ITYPES = [torch.bfloat16]


def causal_conv1d_update_ref(
    x, conv_state, weight, bias=None, activation=None, cache_seqlens=None
):
    """Per-sequence torch reference; test-only, never imported by production.

    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1; updated
    in place by taking the last state_len columns of concat(conv_state, x).
    weight: (dim, width); bias: (dim,)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        x_new = torch.cat([conv_state, x], dim=-1).to(
            weight.dtype
        )  # (batch, dim, state_len + seqlen)
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        width_idx = torch.arange(
            -(width - 1), 0, dtype=torch.long, device=x.device
        ).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = (
            torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        )
        x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
        copy_idx = torch.arange(seqlen, dtype=torch.long, device=x.device).unsqueeze(
            0
        ) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[
        :, :, -seqlen:
    ]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


def _load_vllm_ascend_ref():
    """vllm-ascend's native causal_conv1d_update, or None when unavailable."""
    try:
        from vllm_ascend.ops.causal_conv1d import causal_conv1d_update as native
    except Exception:  # ImportError, or an older tree that raises on import
        return None
    return native


VLLM_ASCEND_UPDATE = _load_vllm_ascend_ref()
REF_SOURCES = ["torch"] + (["vllm_ascend"] if VLLM_ASCEND_UPDATE is not None else [])


@pytest.mark.causal_conv1d_update
@pytest.mark.parametrize("ref_source", REF_SOURCES)
@pytest.mark.parametrize("itype", ITYPES)
@pytest.mark.parametrize("silu_activation", SILU)
@pytest.mark.parametrize("has_bias", HAS_BIAS)
@pytest.mark.parametrize("seqlen", SEQS)
@pytest.mark.parametrize("width", WIDTH)
@pytest.mark.parametrize("dim", DIM)
@pytest.mark.parametrize("with_padding", WITH_PADDING)
@pytest.mark.parametrize("batch_size", BATCH)
def test_causal_conv1d_update_with_batch_gather(
    batch_size,
    with_padding,
    dim,
    width,
    seqlen,
    has_bias,
    silu_activation,
    itype,
    ref_source,
):
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2

    torch.manual_seed(0)

    padding = 5 if with_padding else 0
    padded_batch_size = batch_size + padding
    # total_entries = number of cache line
    total_entries = 10 * batch_size

    # x will be (batch, dim, seqlen) with contiguous along dim-axis. The
    # channel-last base is kept because vllm-ascend's native API infers the
    # feature dim from x.shape[-1] and therefore wants (batch, seqlen, dim).
    x_channel_last = torch.randn(
        padded_batch_size, seqlen, dim, device=device, dtype=itype
    )
    x = x_channel_last.transpose(1, 2)

    x_ref = x[:batch_size].clone()
    conv_state_indices = torch.randperm(total_entries)[:batch_size].to(
        dtype=torch.int32, device=device
    )
    padded_state_indices = torch.concat(
        [
            conv_state_indices,
            torch.as_tensor([PAD_SLOT_ID] * padding, dtype=torch.int32, device=device),
        ],
        dim=0,
    )

    # conv_state will be (cache_lines, dim, state_len)
    # with contiguous along dim-axis
    conv_state = torch.randn(
        total_entries, width - 1, dim, device=device, dtype=itype
    ).transpose(1, 2)

    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None
    conv_state_ref = conv_state[conv_state_indices, :].detach().clone()
    activation = None if not silu_activation else "silu"
    # The kernel writes its result over x in place, so snapshot what the native
    # reference needs before the call.
    native_x = x_channel_last.clone() if ref_source == "vllm_ascend" else None
    native_state = conv_state.clone() if ref_source == "vllm_ascend" else None
    npu_out = flaggems_vllm.causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation=activation,
        conv_state_indices=padded_state_indices,
        pad_slot_id=PAD_SLOT_ID,
    )

    if ref_source == "vllm_ascend":
        out_ref = VLLM_ASCEND_UPDATE(
            native_x,
            native_state,
            weight,
            bias,
            activation=activation,
            conv_state_indices=padded_state_indices,
            pad_slot_id=PAD_SLOT_ID,
        ).transpose(1, 2)[:batch_size]
        conv_state_ref = native_state[conv_state_indices, :].detach().clone()
    else:
        out_ref = causal_conv1d_update_ref(
            x_ref, conv_state_ref, weight, bias, activation=activation
        )

    npu_out = npu_out.cpu()
    out_ref = out_ref.cpu()

    # 1. mainline output matches the reference
    assert torch.allclose(npu_out[:batch_size], out_ref, rtol=rtol, atol=atol)

    # 2. in-place conv-state update matches the reference
    assert torch.allclose(
        conv_state[conv_state_indices].cpu(),
        conv_state_ref.cpu(),
        rtol=rtol,
        atol=atol,
    )
