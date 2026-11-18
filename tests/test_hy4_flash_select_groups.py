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

op = flaggems_vllm.hy4_flash_select_groups


def make_inputs(shape):
    rows, width, budget = shape
    logits = torch.randn(rows, width, device="cuda")
    upper = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    return logits, upper, budget


def reference(logits, upper, budget):
    rows, width = logits.shape
    output = torch.full((rows, budget), -1, device=logits.device, dtype=torch.int32)
    if not width:
        return output
    columns = torch.arange(width, device=logits.device)
    valid = (columns < upper[:, None]) & (logits != -torch.inf)
    values = logits.masked_fill(~valid, -torch.inf)
    ranked = torch.argsort(values, dim=-1, descending=True, stable=True)[:, :budget]
    ranked = ranked.masked_fill(~valid.gather(1, ranked), torch.iinfo(torch.int32).max)
    ordered = ranked.sort(dim=-1).values
    output[:, : ordered.shape[1]] = ordered.masked_fill(
        ordered == torch.iinfo(torch.int32).max, -1
    )
    return output


@pytest.mark.parametrize("width,budget", [(0, 4), (3, 4), (2049, 512), (8193, 512)])
@pytest.mark.hy4_flash_select_groups
def test_select_groups_handles_bounds_and_ties(width, budget):
    logits, upper, _ = make_inputs((4, width, budget))
    upper[0], upper[1] = -1, 1
    if width > 3:
        logits[:, :3] = torch.tensor([torch.nan, torch.inf, -torch.inf], device="cuda")
        logits[:, 3:] = 1
    torch.testing.assert_close(
        op(logits, upper, budget), reference(logits, upper, budget)
    )


@pytest.mark.hy4_flash_select_groups
def test_select_group_replay_updates_bounds():
    logits, upper, budget = make_inputs((4, 8193, 512))
    for _ in range(2):
        op(logits, upper, budget)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = op(logits, upper, budget)
    for bound in (8193, 7, 8193):
        upper.fill_(bound)
        graph.replay()
        torch.testing.assert_close(output, reference(logits, upper, budget))
