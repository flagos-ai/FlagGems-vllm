# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Exact mask and history-update checks for draft watermarking."""

import pytest
import torch

from flaggems_vllm import draft_watermarking_mask

try:
    from vllm.v1.worker.gpu.sample.watermark import (
        draft_watermarking_mask as vllm_reference,
    )
except (ImportError, OSError):
    vllm_reference = None

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def reference(inputs, window, include_prompt):
    tokens, mapping, prompt, total, prior, contexts, steps, enabled = (
        t.cpu() for t in inputs
    )
    prior = prior.clone()
    result = torch.zeros(len(contexts), dtype=torch.bool)
    for row, request in enumerate(mapping.tolist()):
        if request < 0:
            continue
        step = int(steps[row])
        current = tuple(contexts[row].tolist())
        sequence = tokens[
            request, 0 if include_prompt else int(prompt[request]) : int(total[request])
        ].tolist()
        first = 0 if window is None else max(len(sequence) - max(window - step, 0), 0)
        padded = [-1] * len(current) + sequence
        repeated = any(
            tuple(padded[pos : pos + len(current)]) == current
            for pos in range(first, len(sequence))
        )
        first_step = 0 if window is None else max(0, step - window)
        repeated |= any(
            tuple(prior[row, s].tolist()) == current for s in range(first_step, step)
        )
        repeated |= include_prompt and any(token < 0 for token in current)
        result[row] = bool(enabled[row]) and not repeated
        prior[row, step] = contexts[row]
    return result, prior


def make_inputs(rows, history, width, draft_steps=8, seed=17):
    generator = torch.Generator().manual_seed(seed)
    capacity = rows + 2
    tokens = torch.randint(
        0,
        32000,
        (capacity, history + width + 8),
        generator=generator,
        dtype=torch.int32,
    )
    contexts = torch.randint(
        0, 32000, (rows, width), generator=generator, dtype=torch.int64
    )
    prior = torch.randint(
        0, 32000, (rows, draft_steps, width), generator=generator, dtype=torch.int64
    )
    mapping = torch.arange(rows, dtype=torch.int32).flip(0)
    steps = torch.arange(rows, dtype=torch.int64) % draft_steps
    enabled = torch.arange(rows) % 5 != 3
    prompt = torch.full((capacity,), min(history, 3), dtype=torch.int32)
    total = torch.full((capacity,), history, dtype=torch.int32)
    for row in range(rows):
        request = int(mapping[row])
        if row % 7 == 0 and history >= width + 4:
            tokens[request, 3 : 3 + width] = contexts[row]
        if row % 7 == 1 and int(steps[row]) > 0:
            prior[row, 0] = contexts[row]
        if row % 7 == 2:
            contexts[row, 0] = -1
    mapping[4::11] = -1
    return tuple(
        t.cuda()
        for t in (tokens, mapping, prompt, total, prior, contexts, steps, enabled)
    )


def check(inputs, window, include_prompt):
    expected, expected_prior = reference(inputs, window, include_prompt)
    before = inputs[4].clone()
    pointer = inputs[4].data_ptr()
    actual = draft_watermarking_mask(*inputs, window, include_prompt)
    assert actual.dtype == torch.bool
    assert actual.shape == (len(inputs[5]),)
    assert inputs[4].data_ptr() == pointer
    assert torch.equal(actual.cpu(), expected)
    assert torch.equal(inputs[4].cpu(), expected_prior)
    if vllm_reference is not None:
        original_inputs = (*inputs[:4], before, *inputs[5:])
        original = vllm_reference(*original_inputs, window, include_prompt)
        assert torch.equal(actual, original)
        assert torch.equal(inputs[4], before)
    repeated = draft_watermarking_mask(*inputs, window, include_prompt)
    assert torch.equal(repeated, actual)
    assert torch.equal(inputs[4].cpu(), expected_prior)


@pytest.mark.draft_watermarking_mask
@pytest.mark.parametrize(
    "shape",
    [
        (0, 0, 4),
        (1, 0, 4),
        (1, 1, 1),
        (2, 3, 4),
        (5, 33, 7),
        (31, 511, 4),
        (32, 512, 4),
        (33, 513, 4),
        (127, 1023, 4),
        (128, 1024, 4),
        (129, 1025, 4),
        (8, 8191, 4),
        (8, 8192, 4),
        (8, 8193, 4),
        (5, 37, 16),
        (3, 23, 3),
    ],
)
@pytest.mark.parametrize("window", [None, 1, 8192])
@pytest.mark.parametrize("include_prompt", [False, True])
def test_mask_and_history(shape, window, include_prompt):
    check(make_inputs(*shape), window, include_prompt)


@pytest.mark.draft_watermarking_mask
@pytest.mark.parametrize(
    "layout", ["rows", "columns", "broadcast_steps", "strided_vectors"]
)
def test_strides(layout):
    inputs = list(make_inputs(17, 513, 4))
    if layout == "rows":
        for index in (0, 5):
            original = inputs[index]
            padded = torch.empty(
                (original.shape[0], original.shape[1] + 3),
                dtype=original.dtype,
                device="cuda",
            )
            padded[:, 1:-2] = original
            inputs[index] = padded[:, 1:-2]
        original = inputs[4]
        padded = torch.empty((17, 9, 7), dtype=original.dtype, device="cuda")
        padded[:, :8, 1:5] = original
        inputs[4] = padded[:, :8, 1:5]
    elif layout == "columns":
        original = inputs[5]
        padded = torch.empty((17, 8), dtype=original.dtype, device="cuda")
        padded[:, ::2] = original
        inputs[5] = padded[:, ::2]
    elif layout == "broadcast_steps":
        inputs[6] = torch.tensor(3, device="cuda", dtype=torch.int64).expand(17)
    else:
        for index in (1, 6, 7):
            original = inputs[index]
            padded = torch.empty(34, dtype=original.dtype, device="cuda")
            padded[::2] = original
            inputs[index] = padded[::2]
    check(tuple(inputs), 511, True)


@pytest.mark.draft_watermarking_mask
def test_multiple_draft_steps():
    inputs = list(make_inputs(5, 33, 4))
    inputs[6] = torch.zeros(5, dtype=torch.int64, device="cuda")
    for step in range(8):
        inputs[6].fill_(step)
        if step % 3 == 0:
            inputs[5].fill_(7)
        else:
            inputs[5].add_(1)
        check(tuple(inputs), 5, False)


@pytest.mark.draft_watermarking_mask
def test_invalid_window():
    with pytest.raises(ValueError, match="max_history"):
        draft_watermarking_mask(*make_inputs(1, 8, 4), 0, False)


@pytest.mark.draft_watermarking_mask
@pytest.mark.parametrize("matching", [False, True])
def test_frequent_anchor_tokens(matching):
    inputs = list(make_inputs(9, 8193, 4))
    inputs[0].zero_()
    inputs[5].zero_()
    if not matching:
        inputs[5][:, 0] = 17
    check(tuple(inputs), 8192, False)


@pytest.mark.draft_watermarking_mask
def test_plain_triton_path(monkeypatch):
    import importlib

    module = importlib.import_module("flaggems_vllm.ops.draft_watermarking_mask")
    monkeypatch.setattr(module, "_HAS_LAYOUT", False)
    check(make_inputs(5, 513, 4), 8192, True)


@pytest.mark.draft_watermarking_mask
def test_cpu_is_unsupported():
    inputs = tuple(t.cpu() for t in make_inputs(1, 8, 4))
    with pytest.raises(NotImplementedError, match="CUDA"):
        draft_watermarking_mask(*inputs, 8192, False)


@pytest.mark.draft_watermarking_mask
def test_short_history_after_long_history(monkeypatch):
    import importlib

    import triton

    module = importlib.import_module("flaggems_vllm.ops.draft_watermarking_mask")
    kernel = module._draft_watermarking_mask_kernel
    monkeypatch.setattr(
        kernel.fn, "configs", [triton.Config({"BLOCK": 8192}, num_warps=4)]
    )
    monkeypatch.setattr(kernel, "kernel_cache", tuple({} for _ in kernel.kernel_cache))
    inputs = list(make_inputs(32, 8193, 4))
    check(tuple(inputs), 8192, False)
    for length in (64, 512, 513, 8193):
        inputs[3].fill_(length)
        check(tuple(inputs), 8192, False)


@pytest.mark.draft_watermarking_mask
def test_repeated_request_indices():
    inputs = list(make_inputs(9, 513, 4))
    inputs[1].fill_(3)
    inputs[1][4] = -1
    check(tuple(inputs), 511, True)


@pytest.mark.draft_watermarking_mask
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_context_integer_types(dtype):
    inputs = list(make_inputs(5, 513, 4))
    inputs[4] = inputs[4].to(dtype)
    inputs[5] = inputs[5].to(dtype)
    check(tuple(inputs), 8192, False)
