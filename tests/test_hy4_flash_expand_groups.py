# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

import pytest
import torch

import flaggems_vllm

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or flaggems_vllm.vendor_name != "nvidia",
        reason="Hy4 Flash IndexPool requires NVIDIA CUDA",
    ),
]

op = flaggems_vllm.hy4_flash_expand_groups


def make_inputs(shape):
    rows, budget, ratio = shape
    ids = torch.arange(budget, device="cuda", dtype=torch.int32).expand(rows, -1)
    positions = torch.arange(rows, device="cuda", dtype=torch.int32) + budget * ratio
    active = positions % 2 == 0
    return ids, positions, ratio, active


def reference(group_ids, positions, ratio, active):
    live = positions >= 0
    if active is not None:
        live = live & active
    slots = torch.arange(ratio, device=positions.device)
    upper = positions // ratio
    history = group_ids[:, :, None] * ratio + slots
    valid_history = (group_ids >= 0) & (group_ids < upper[:, None])
    current = upper[:, None] * ratio + slots
    values = torch.cat((history.flatten(1), current), dim=1)
    valid = torch.cat(
        (valid_history.repeat_interleave(ratio, dim=1), current <= positions[:, None]),
        dim=1,
    )
    valid = valid & live[:, None]
    collect = torch.argsort(valid.int(), dim=-1, descending=True, stable=True)
    return values.masked_fill(~valid, -1).gather(1, collect).to(torch.int32)


@pytest.mark.parametrize("shape", [(0, 3, 2), (1, 0, 4), (7, 11, 8), (7, 31, 16)])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.hy4_flash_expand_groups
def test_expansion_packs_only_live_tokens(shape, masked):
    group_ids, positions, ratio, active = make_inputs(shape)
    if positions.numel():
        positions[0] = -1
    args = group_ids, positions, ratio, active if masked else None
    torch.testing.assert_close(op(*args), reference(*args))


@pytest.mark.hy4_flash_expand_groups
def test_expansion_rejects_float_positions():
    group_ids, positions, ratio, active = make_inputs((1, 2, 4))
    with pytest.raises(ValueError, match="positions"):
        op(group_ids, positions.float(), ratio, active)
