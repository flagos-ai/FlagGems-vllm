# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""LoRA down-projection, including the caller's grouped metadata and FP32 buffer."""

from contextlib import contextmanager

import pytest
import torch

import flaggems_vllm as gems


@contextmanager
def _cpu_threads():
    # Tiny per-adapter products can otherwise start a full BLAS thread pool.
    previous = torch.get_num_threads()
    if previous == 1:
        yield
        return
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module", autouse=True)
def _limit_test_threads():
    with _cpu_threads():
        yield


@_cpu_threads()
def reference_output(
    inputs,
    lora_a_weights,
    output_tensor,
    token_lora_mapping,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    no_lora_flag_cpu,
    num_active_loras,
    scaling,
):
    """Compute from original token rows on CPU, independent of the grouped indices."""
    if not inputs.size(0) or no_lora_flag_cpu.item():
        return output_tensor.detach().cpu().clone()
    expected = torch.zeros(output_tensor.shape, dtype=output_tensor.dtype, device="cpu")
    if num_active_loras.item() == 0:
        return expected
    mapping = token_lora_mapping.detach().cpu()
    x = inputs.detach().cpu().float()
    for adapter in mapping.unique().tolist():
        if adapter == -1:
            continue
        rows = mapping == adapter
        for i, weight in enumerate(lora_a_weights):
            a = weight[adapter].detach().reshape(weight.shape[-2:]).cpu().float()
            expected[i, rows] = ((x[rows] @ a.T) * scaling).to(expected.dtype)
    return expected


def assert_output(actual, expected, args, *, exact=False):
    """Check the original output dtype, sentinel and untouched base-model rows."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert actual.device == args[0].device
    actual = actual.detach().cpu()
    if exact or args[8].item() or not args[9].item():
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        return
    inactive = args[3].cpu() == -1
    torch.testing.assert_close(
        actual[:, inactive], expected[:, inactive], rtol=0, atol=0
    )
    # X ~ N(0,1), A ~ N(0,1/K) keep the random projection at unit scale.
    # The FP32 allowance covers different K reduction orders. Low-precision
    # outputs can fall on opposite sides of the final rounding boundary.
    rtol, atol = {
        torch.float32: (2e-4, 2e-5),
        torch.float16: (2 * torch.finfo(torch.float16).eps, 2e-5),
        torch.bfloat16: (2 * torch.finfo(torch.bfloat16).eps, 2e-4),
    }[expected.dtype]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=True)


def _metadata(tokens, adapters, *, groups=None, active=None, no_lora=False):
    groups = sorted(set(tokens)) if groups is None else groups
    capacity = adapters + 1
    assert len(groups) <= capacity
    rows = [[i for i, adapter in enumerate(tokens) if adapter == g] for g in groups]
    order = [i for group_rows in rows for i in group_rows]
    assert sorted(order) == list(range(len(tokens)))
    counts = [len(group_rows) for group_rows in rows]
    starts = [0]
    for count in counts:
        starts.append(starts[-1] + count)
    pad = capacity - len(groups)

    def metadata_tensor(values):
        return torch.tensor(values, dtype=torch.int32, device=gems.device)

    return (
        metadata_tensor(tokens),
        metadata_tensor(order),
        metadata_tensor(counts + [0] * pad),
        metadata_tensor(starts + [0] * pad),
        metadata_tensor(groups + [-1] * pad),
        torch.tensor([no_lora], dtype=torch.bool, device="cpu"),
        torch.tensor(
            [len(groups) if active is None else active], dtype=torch.int32, device="cpu"
        ),
    )


def _case(tokens, adapters, rank, hidden, slices, dtype, out_dtype, scaling, **meta):
    gen = torch.Generator(device="cpu").manual_seed(63017)
    x = torch.randn((len(tokens), hidden), generator=gen, device="cpu")
    weights = []
    for _ in range(slices):
        a = torch.randn((adapters, rank, hidden), generator=gen, device="cpu")
        weights.append((a * (hidden**-0.5)).to(dtype).to(gems.device))
    out = torch.full(
        (slices, len(tokens), rank), 19, dtype=out_dtype, device=gems.device
    )
    return [
        x.to(dtype).to(gems.device),
        weights,
        out,
        *_metadata(tokens, adapters, **meta),
        scaling,
    ]


def _check(args, *, exact=False, expected=None):
    x, weights, out = args[:3]
    readonly = [x, *weights, *args[3:10]]
    before = [t.detach().cpu().clone() for t in readonly]
    if expected is None:
        expected = reference_output(*args)
    assert gems.lora_shrink(*args) is None
    assert_output(out, expected, args, exact=exact)
    for tensor, initial in zip(readonly, before):
        torch.testing.assert_close(
            tensor.cpu(), initial, rtol=0, atol=0, equal_nan=True
        )
    return out


@pytest.mark.lora_shrink
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "tokens,adapters,rank,hidden,slices,use_fp32,scaling",
    [
        pytest.param([0], 1, 1, 1, 1, False, 1.0, id="singleton"),
        pytest.param([1, -1, 0], 2, 4, 35, 9, True, 0.5, id="nine-weight-slices"),
        pytest.param([6, -1, 2, 6, -1], 7, 4, 31, 2, True, -0.5, id="sparse-decode"),
        pytest.param([2, 0, -1] * 5, 3, 8, 127, 3, False, 0.0, id="zero-scale"),
        pytest.param([3] * 8, 4, 16, 4096, 1, True, 2.0, id="decode-4096"),
        pytest.param(
            [4] * 127 + [1] * 2 + [-1], 5, 32, 2048, 3, True, 0.5, id="prefill-2048"
        ),
        pytest.param(
            [1, -1, 4, 1, 4, 1, 4], 5, 64, 257, 2, False, 1.0, id="rank64-tail"
        ),
        pytest.param([-1, 2, 3, 2, 0], 4, 128, 129, 1, True, 0.125, id="rank128-tail"),
    ],
)
def test_lora_shrink(tokens, adapters, rank, hidden, slices, use_fp32, scaling, dtype):
    out_dtype = torch.float32 if use_fp32 else dtype
    args = _case(tokens, adapters, rank, hidden, slices, dtype, out_dtype, scaling)
    _check(args, exact=scaling == 0)


@pytest.mark.lora_shrink
@pytest.mark.parametrize(
    "tokens,groups,active,no_lora",
    [
        pytest.param([], [], 0, False, id="empty-first-call"),
        pytest.param([-1] * 7, [-1], 0, True, id="all-base-preserves-output"),
        pytest.param([2, 2, 2], [2], 0, False, id="zero-active-clears-output"),
        pytest.param([-1] * 5, [-1], 1, False, id="only-base-group-clears-output"),
        pytest.param(
            [2, -1, 2, 0], [-1, 0, 1, 2], 6, False, id="empty-and-padded-groups"
        ),
    ],
)
def test_lora_shrink_group_controls(tokens, groups, active, no_lora):
    args = _case(
        tokens,
        5,
        8,
        33,
        2,
        torch.float16,
        torch.float32,
        1.0,
        groups=groups,
        active=active,
        no_lora=no_lora,
    )
    _check(args)


@pytest.mark.lora_shrink
def test_lora_shrink_active_prefix():
    # A well-formed suffix still has valid slots and indices. Only the first
    # two groups are scheduled by the CPU count; the suffix rows must be zero.
    args = _case(
        [2, 0, -1, 2, 0], 3, 8, 65, 2, torch.float16, torch.float32, 1.0, active=2
    )
    expected = reference_output(*args)
    expected[:, [0, 3], :] = 0
    _check(args, expected=expected)
    assert torch.count_nonzero(args[2].cpu()[:, [0, 3]]) == 0


@pytest.mark.lora_shrink
def test_lora_shrink_live_buffers():
    tokens = [2, -1, 0, 2, 0]
    args = _case(tokens, 4, 8, 67, 2, torch.bfloat16, torch.float32, 0.75, active=5)
    _check(args)
    # Keep every input address fixed while switching the live CPU controls.
    args[8].fill_(True)
    args[2].fill_(-31)
    _check(args, exact=True)
    args[8].fill_(False)
    args[9].zero_()
    _check(args, exact=True)
    new_meta = _metadata([1, 1, -1, 3, 1], 4)
    for dst, src in zip(args[3:10], new_meta):
        dst.copy_(src)
    args[1][0].fill_(1)
    args[1][1].fill_(-1)
    _check(args)
    # The old allocations stay live while the next call binds two new addresses.
    old = args[1]
    args[1] = [old[1].clone(), old[0].clone()]
    _check(args)


@pytest.mark.lora_shrink
def test_lora_shrink_output_in_unused_input_padding():
    args = _case([0, 1], 2, 1, 2, 1, torch.float16, torch.float16, 1.0)
    storage = args[0].new_full((22,), -43)
    x = storage.as_strided((2, 2), (20, 1))
    x.copy_(args[0])
    args[0] = x
    args[2] = storage[4:6].view(1, 2, 1)
    _check(args)
    assert torch.all(storage[6:20].cpu() == -43)


@pytest.mark.lora_shrink
def test_lora_shrink_layout_and_storage_offset():
    tokens = [2, -1, 0, 2, -1, 0, 0]
    args = _case(tokens, 3, 16, 33, 2, torch.float16, torch.float32, 1.0)
    x, weights, out = args[:3]
    storage = x.new_full((x.shape[0], 2 * x.shape[1] + 1), -11)
    args[0] = storage[:, 1::2]
    args[0].copy_(x)
    # The caller supplies either [L,R,K] or [L,1,R,K] per slice.
    args[1][1] = weights[1].unsqueeze(1)
    backing = out.new_full((out.numel() + 2,), -73)
    args[2] = backing[1:-1].view(out.shape)
    _check(args)
    assert backing[0].item() == backing[-1].item() == -73


@pytest.mark.lora_shrink
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("out_fp32", [True, False])
def test_lora_shrink_fp32_cancellation(dtype, out_fp32):
    # 4096 + 1 rounds to 4096 in either input dtype. With a full FP32
    # accumulation the -4096 in the K tail leaves exactly 1.
    args = _case(
        [1, -1, 0, 1], 2, 4, 2051, 2, dtype, torch.float32 if out_fp32 else dtype, -0.25
    )
    args[0].fill_(1)
    for i, weight in enumerate(args[1]):
        weight.zero_()
        weight[..., 0] = 4096
        weight[..., 1] = i + 1
        weight[..., -1] = -4096
    _check(args, exact=True)


@pytest.mark.lora_shrink
def test_lora_shrink_nonfinite():
    args = _case([0, -1, 1], 2, 4, 32, 1, torch.float16, torch.float32, 1.0)
    args[0].fill_(1)
    args[0][1].fill_(float("nan"))
    a = args[1][0]
    a.zero_()
    a[0, 0, 0], a[0, 1, 0], a[0, 2, 0] = float("inf"), -float("inf"), float("nan")
    a[1].fill_(1)
    _check(args, exact=True)


@pytest.mark.lora_shrink
def test_lora_shrink_batch_invariant(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    # Move the same three rows across adapter groups and GEMM tile boundaries.
    tokens = [3, -1, 1] + [1] * 36 + [3] * 29
    args = _case(tokens, 4, 16, 2051, 2, torch.bfloat16, torch.float32, 0.7)
    full = _check(args).cpu()
    for rows in ([0], [2], [65, 1, 2, 0, 40]):
        selected_tokens = [tokens[i] for i in rows]
        subset = [
            args[0][rows],
            args[1],
            args[2].new_full((2, len(rows), 16), 8),
            *_metadata(selected_tokens, 4),
            args[-1],
        ]
        actual = _check(subset)
        torch.testing.assert_close(actual.cpu(), full[:, rows], rtol=0, atol=0)


@pytest.mark.lora_shrink
@pytest.mark.parametrize(
    "kind",
    [
        "input_ndim",
        "input_dtype",
        "weight_hidden",
        "weight_ndim",
        "weight_dtype",
        "empty_weights",
        "slice_rank",
        "weight_layout",
        "output_shape",
        "output_layout",
        "output_dtype",
        "mapping_length",
        "indices_length",
        "counts_length",
        "starts_length",
        "ids_length",
        "metadata_dtype",
        "flag_length",
        "active_length",
        "active_negative",
        "active_overflow",
        "cpu_weight",
        "cpu_output",
        "cpu_indices",
        "device_flag",
        "device_active",
        "output_input_alias",
    ],
)
def test_lora_shrink_invalid(kind):
    args = _case([1, -1, 0], 2, 4, 32, 2, torch.float16, torch.float32, 1.0)
    x, weights, out = args[:3]
    if kind == "input_ndim":
        args[0] = x.unsqueeze(1)
    elif kind == "input_dtype":
        args[0] = x.float()
    elif kind == "weight_hidden":
        weights[0] = weights[0].new_zeros((2, 4, 33))
    elif kind == "weight_ndim":
        weights[0] = weights[0].unsqueeze(0)
    elif kind == "weight_dtype":
        weights[1] = weights[1].float()
    elif kind == "empty_weights":
        args[1] = []
    elif kind == "slice_rank":
        weights[1] = weights[1].new_zeros((2, 5, 32))
    elif kind == "weight_layout":
        weights[0] = weights[0].new_zeros((2, 4, 64))[..., ::2]
    elif kind == "output_shape":
        args[2] = out.new_full((2, 4, 4), 9)
    elif kind == "output_layout":
        args[2] = out.new_full((2, 3, 8), 9)[..., ::2]
    elif kind == "output_dtype":
        args[2] = out.int()
    elif kind in (
        "mapping_length",
        "indices_length",
        "counts_length",
        "starts_length",
        "ids_length",
    ):
        index = {
            "mapping_length": 3,
            "indices_length": 4,
            "counts_length": 5,
            "starts_length": 6,
            "ids_length": 7,
        }[kind]
        args[index] = torch.cat((args[index], args[index][-1:]))
    elif kind == "metadata_dtype":
        args[3] = args[3].float()
    elif kind == "flag_length":
        args[8] = args[8].repeat(2)
    elif kind == "active_length":
        args[9] = args[9].repeat(2)
    elif kind == "active_negative":
        args[9].fill_(-1)
    elif kind == "active_overflow":
        args[9].fill_(args[7].numel() + 1)
    elif kind == "cpu_weight":
        weights[1] = weights[1].cpu()
    elif kind == "cpu_output":
        args[2] = out.cpu()
    elif kind == "cpu_indices":
        args[4] = args[4].cpu()
    elif kind == "device_flag":
        args[8] = args[8].to(gems.device)
    elif kind == "device_active":
        args[9] = args[9].to(gems.device)
    else:
        args[2] = x.view(-1)[: out.numel()].view(out.shape)
    with pytest.raises((ValueError, TypeError)):
        gems.lora_shrink(*args)
