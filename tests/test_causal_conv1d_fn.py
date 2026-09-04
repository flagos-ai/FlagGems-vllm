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

"""Correctness coverage for the Ascend causal-conv1d-fn kernel.

The kernel under test is the Ascend TLE-DSA port living in
``flaggems_vllm/runtime/backend/_ascend/ops/causal_conv1d_fn.py`` and
exported as ``flaggems_vllm.causal_conv1d_fn`` via the vendor op registrar.

Torch references live in this test module only. The production wrapper keeps
its vendor-port torch compute calls (dtype cast / contiguous / copy_) and is
exercised as-is.
"""

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("torch_npu")
import torch.nn.functional as F  # noqa: E402

import flaggems_vllm  # noqa: E402

from .conftest import QUICK_MODE  # noqa: E402

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "ascend" or not torch.npu.is_available(),
    reason="causal_conv1d_fn requires an Ascend NPU",
)

device = "npu"
PAD_SLOT_ID = -1

if QUICK_MODE:
    BATCH = [4]
    WITH_PADDING = [True]
    DIM = [64]
    SEQS = [8]
    WIDTH = [4]
    HAS_BIAS = [True]
    SILU = [True]
    ITYPES = [torch.bfloat16]
else:
    BATCH = [4, 10]
    WITH_PADDING = [True, False]
    DIM = [64, 4096]
    SEQS = [8, 249, 4096]
    WIDTH = [4]
    HAS_BIAS = [True]
    SILU = [True]
    ITYPES = [torch.bfloat16, torch.float16]


def causal_conv1d_ref(
    x,
    weight,
    bias=None,
    initial_states=None,
    return_final_states=False,
    final_states_out=None,
    activation="silu",
):
    """Per-sequence torch reference; test-only, never imported by production.

    x: (batch, dim, seqlen); weight: (dim, width); bias: (dim,)
    initial_states / final_states_out: (batch, dim, width - 1)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
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
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


@pytest.mark.causal_conv1d_fn
@pytest.mark.parametrize("itype", ITYPES)
@pytest.mark.parametrize("silu_activation", SILU)
@pytest.mark.parametrize("has_bias", HAS_BIAS)
@pytest.mark.parametrize("width", WIDTH)
@pytest.mark.parametrize("seqlen", SEQS)
@pytest.mark.parametrize("dim", DIM)
@pytest.mark.parametrize("with_padding", WITH_PADDING)
@pytest.mark.parametrize("batch", BATCH)
def test_causal_conv1d_varlen(
    batch, with_padding, dim, seqlen, width, has_bias, silu_activation, itype
):
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    if itype == torch.float16:
        rtol, atol = 1e-2, 5e-2

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    batch_size = batch
    padding = 3 if with_padding else 0
    padded_batch_size = batch_size + padding
    nsplits = padded_batch_size - 1

    # Randomly split the total seqlen into padded_batch_size sequences.
    eos_pos = torch.randperm(seqlen - 1)[:nsplits].sort().values
    seqlens = [
        torch.diff(
            torch.cat([torch.tensor([-1]), eos_pos, torch.tensor([seqlen - 1])])
        ).tolist()
    ]
    assert sum(seqlens[-1]) == seqlen
    assert all(s > 0 for s in seqlens[-1])

    total_entries = batch_size * 10
    cumsum = torch.cumsum(torch.tensor(seqlens[0]), dim=0).to(torch.int32)
    query_start_loc = torch.concat(
        [torch.tensor([0], dtype=torch.int32), cumsum], dim=0
    )

    # x: (1, dim, seqlen), slice from a wider tensor to avoid contiguity shortcuts.
    x = torch.randn(1, seqlen, 4096 + dim + 64, device=device, dtype=itype)
    x = x.transpose(1, 2)[:, 4096 : 4096 + dim, :]

    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None
    x_ref = x.clone()
    weight_ref = weight.clone()
    bias_ref = bias.clone() if bias is not None else None
    activation = None if not silu_activation else "silu"

    # conv_states: (total_entries, dim, width - 1), updated in place by the kernel.
    final_states = torch.randn(
        total_entries, width - 1, dim, device=device, dtype=itype
    ).transpose(1, 2)
    final_states_ref = final_states.clone()
    has_initial_states = torch.randint(
        0, 2, (query_start_loc.shape[0] - 1,), dtype=torch.bool, device=device
    )
    state_indices = torch.randperm(total_entries, dtype=torch.int32, device=device)[
        :batch_size
    ]
    padded_state_indices = torch.concat(
        [
            state_indices,
            torch.as_tensor([PAD_SLOT_ID] * padding, dtype=torch.int32, device=device),
        ],
        dim=-1,
    )

    out = flaggems_vllm.causal_conv1d_fn(
        x.squeeze(0),
        weight,
        bias=bias,
        conv_states=final_states,
        query_start_loc=query_start_loc.npu(),
        cache_indices=padded_state_indices,
        has_initial_state=has_initial_states,
        activation=activation,
        pad_slot_id=PAD_SLOT_ID,
    )

    # Reference: run the first batch_size sequences one by one (skip pad slots),
    # writing final conv states back into final_states_ref in place.
    out_ref_b = []
    splits = torch.split(x_ref, seqlens[0], dim=-1)
    for i in range(len(seqlens[0])):
        x_s = splits[i]
        if padded_state_indices[i] == PAD_SLOT_ID:
            continue
        out_ref_b.append(
            causal_conv1d_ref(
                x_s,
                weight_ref,
                bias_ref,
                activation=activation,
                return_final_states=True,
                final_states_out=final_states_ref[padded_state_indices[i]].unsqueeze(0),
                initial_states=(
                    final_states_ref[padded_state_indices[i]].unsqueeze(0)
                    if has_initial_states[i]
                    else None
                ),
            )
        )
    out_ref_tensor = torch.cat([t[0] for t in out_ref_b], dim=2)

    # 1. in-place conv-state update matches the reference
    assert torch.allclose(
        final_states[state_indices],
        final_states_ref[state_indices],
        rtol=rtol,
        atol=atol,
    )

    # 2. mainline output matches the per-sequence reference concatenation
    unpadded_out = out[:, : out_ref_tensor.shape[-1]]
    assert torch.allclose(unpadded_out, out_ref_tensor, rtol=rtol, atol=atol)
