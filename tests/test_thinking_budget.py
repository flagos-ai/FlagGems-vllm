# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Compare public thinking-budget entry points on identical inputs."""

import pytest
import torch

_thinking_budget_mod = pytest.importorskip(
    "vllm.v1.worker.gpu.sample.thinking_budget",
    reason=(
        "vllm.v1.worker.gpu.sample.thinking_budget not available in the "
        "installed vLLM version"
    ),
)
vllm_reference = _thinking_budget_mod.apply_thinking_budget

from flaggems_vllm import apply_thinking_budget
from flaggems_vllm import device as DEVICE
from flaggems_vllm import runtime

pytestmark = [
    pytest.mark.apply_thinking_budget,
    pytest.mark.skipif(
        DEVICE == "cpu" or not runtime.torch_device_fn.is_available(),
        reason="requires the active accelerator",
    ),
]


def make_inputs(
    tokens,
    input_ids,
    budget=3,
    natural_end=(91,),
    forced_end=(91,),
    logits_kind="zeros",
):
    # Upstream RequestState allocates slot 3 first from a four-request pool.
    rows = len(input_ids)
    logits = torch.zeros((rows, 128), dtype=torch.float32, device=DEVICE)
    if logits_kind == "range":
        logits.copy_(torch.arange(128, dtype=torch.float32, device=DEVICE))
    elif logits_kind == "masked":
        logits[0, 91] = -float("inf")
    history = torch.zeros(
        (4, max(64, len(tokens) + 1)), dtype=torch.int32, device=DEVICE
    )
    history[3, : len(tokens)] = torch.tensor(tokens, dtype=torch.int32, device=DEVICE)
    budgets = torch.full((4,), -1, dtype=torch.int32, device=DEVICE)
    budgets[3] = budget
    lengths = torch.zeros(4, dtype=torch.int32, device=DEVICE)
    lengths[3] = len(tokens)
    req_ids = torch.tensor([3], dtype=torch.int32, device=DEVICE)
    # Match model_runner's shared mapping for one sampled row per request.
    mapping = (
        req_ids
        if rows == 1
        else torch.full((rows,), 3, dtype=torch.int32, device=DEVICE)
    )
    return [
        logits,
        req_ids,
        mapping,
        budgets,
        history,
        lengths,
        torch.tensor(input_ids, dtype=torch.int32, device=DEVICE),
        torch.arange(rows, dtype=torch.int32, device=DEVICE),
        torch.full((4,), -1, dtype=torch.int32, device=DEVICE),
        torch.full((4,), -1, dtype=torch.int32, device=DEVICE),
        torch.zeros(4, dtype=torch.int32, device=DEVICE),
        torch.tensor([90], dtype=torch.int32, device=DEVICE),
        torch.tensor(natural_end, dtype=torch.int32, device=DEVICE),
        torch.tensor(forced_end, dtype=torch.int32, device=DEVICE),
    ]


def check(args):
    reference_args = list(args)
    # Logits and all three caches are outputs of this public API.
    for index in (0, 8, 9, 10):
        reference_args[index] = args[index].clone()
    vllm_reference(*reference_args)
    apply_thinking_budget(*args)
    for index in (0, 8, 9, 10):
        torch.testing.assert_close(args[index], reference_args[index], rtol=0, atol=0)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            dict(tokens=[1, 90, 10, 11, 12], input_ids=[12], logits_kind="range"),
            id="budget-reached",
        ),
        pytest.param(
            dict(tokens=[1, 90, 10, 11, 12], input_ids=[12], logits_kind="masked"),
            id="restore-masked-end",
        ),
        pytest.param(dict(tokens=[1, 90, 10, 11], input_ids=[11]), id="before-budget"),
        pytest.param(
            dict(
                tokens=[1, 90, 10, 11, 12],
                input_ids=[12, 92],
                natural_end=(92, 93),
                forced_end=(92, 93),
            ),
            id="multi-token-end",
        ),
        pytest.param(
            dict(tokens=[1, 90, 10, 11, 12], input_ids=[12, 92], forced_end=(92, 93)),
            id="distinct-forced-end",
        ),
        pytest.param(
            dict(
                tokens=[1, 90, 10, 91, 20, 21, 22],
                input_ids=[22],
                forced_end=(92, 93),
            ),
            id="natural-end",
        ),
        pytest.param(
            dict(tokens=[1, 90, 10, 11, 12], input_ids=[12], budget=-1),
            id="plain-request",
        ),
        pytest.param(
            dict(tokens=[1, 90, 10, 11, 12, 91, 13], input_ids=[13]),
            id="latest-prefill-end",
        ),
        pytest.param(
            dict(tokens=[1, 90, 10, 11, 12, 91, 13, 90, 14, 15, 16], input_ids=[16]),
            id="latest-prefill-start",
        ),
        pytest.param(
            # ThinkingBudgetState clamps the upstream 2**40 input before launch.
            dict(tokens=[1, 90, 10, 11, 12], input_ids=[12], budget=2**31 - 1),
            id="clamped-budget",
        ),
        pytest.param(
            dict(
                tokens=[1, 90, 10, 11, 92],
                input_ids=[92],
                natural_end=(92, 93),
                forced_end=(92, 93),
            ),
            id="resume-end-prefix",
        ),
    ],
)
def test_thinking_budget(case):
    check(make_inputs(**case))


def test_thinking_budget_incremental_generation():
    tokens = [1, 90, *([10] * 16382)]
    args = make_inputs(tokens, [10], budget=32768)
    check(args)
    args[4][3, len(tokens)] = 10
    args[5][3] = len(tokens) + 1
    check(args)
