# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Independent LoRA-B arithmetic, original token order and caller-owned output."""

from contextlib import contextmanager

import pytest
import torch

import flaggems_vllm as gems
from baseline import lora_expand as workload


@contextmanager
def _cpu_threads():
    previous = torch.get_num_threads()
    if previous == 1:
        yield
        return
    # The CPU products are small per adapter. Avoid starting a large BLAS pool.
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module", autouse=True)
def _limit_test_threads():
    with _cpu_threads():
        yield


def _scheduled_adapters(ids, counts, flag, active, slots):
    if flag.item():
        return []
    limit = active.item()
    return [
        a
        for a, n in zip(ids[:limit].cpu().tolist(), counts[:limit].cpu().tolist())
        if n > 0 and 0 <= a < slots
    ]


@_cpu_threads()
def reference_output(
    inputs,
    lora_b_weights,
    output_tensor,
    token_lora_mapping,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    no_lora_flag_cpu,
    num_active_loras,
    offset_start=0,
    add_inputs=False,
):
    """Resolve each original token's adapter from its mapping, independently of the sorted gather."""
    expected = output_tensor.detach().cpu().clone()
    adapters = _scheduled_adapters(
        lora_ids,
        num_tokens_per_lora,
        no_lora_flag_cpu,
        num_active_loras,
        lora_b_weights[0].size(0),
    )
    if not adapters:
        return expected
    mapping = token_lora_mapping.detach().cpu()
    rows = [(adapter, torch.where(mapping == adapter)[0]) for adapter in adapters]
    start = offset_start
    for s, weight in enumerate(lora_b_weights):
        end = start + weight.size(-2)
        # The shrink buffer can be FP32. Native expands using low-precision
        # multiplicands, FP32 reduction and a second cast before residual addition.
        x = inputs[s].detach().cpu().to(weight.dtype).float()
        b = (
            weight.detach()
            .cpu()
            .reshape(weight.size(0), weight.size(-2), weight.size(-1))
            .float()
        )
        for adapter, indices in rows:
            projected = (x[indices] @ b[adapter].T).to(weight.dtype)
            if add_inputs:
                # Triton promotes a mixed FP16/BF16 addition to FP16.
                residual = expected[indices, start:end]
                if (
                    projected.dtype != residual.dtype
                    and residual.dtype != torch.float32
                ):
                    projected = projected.half() + residual.half()
                else:
                    projected = projected.float() + residual.float()
            expected[indices, start:end] = projected.to(expected.dtype)
        start = end
    return expected


@_cpu_threads()
def assert_output(actual, expected, args, *, exact=False):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert actual.device == args[0].device
    host = actual.detach().cpu()
    adapters = _scheduled_adapters(
        args[7], args[5], args[8], args[9], args[1][0].size(0)
    )
    if exact or not adapters:
        torch.testing.assert_close(host, expected, rtol=0, atol=0, equal_nan=True)
        return

    # Compare every untouched output element exactly, even on mixed-dtype rows.
    mapping = args[3].detach().cpu()
    rows = torch.zeros_like(mapping, dtype=torch.bool)
    for adapter in adapters:
        rows |= mapping == adapter
    start = args[10] if len(args) > 10 else 0
    end = start + sum(w.size(-2) for w in args[1])
    for unchanged, initial in (
        (host[~rows], expected[~rows]),
        (host[rows, :start], expected[rows, :start]),
        (host[rows, end:], expected[rows, end:]),
    ):
        torch.testing.assert_close(unchanged, initial, rtol=0, atol=0, equal_nan=True)

    # Neighboring output values can result from a dot rounding near a 16-bit
    # boundary. Allow one projected ULP, including cancellation with the 0.75 residual.
    eps = max(torch.finfo(args[1][0].dtype).eps, torch.finfo(expected.dtype).eps)
    torch.testing.assert_close(
        host[rows, start:end],
        expected[rows, start:end],
        rtol=eps,
        atol=eps,
        equal_nan=True,
    )


def _check(args, *, exact=False, defaults=False):
    x, weights, out = args[:3]
    identity = out.data_ptr(), out.storage_offset(), out.stride()
    readonly = [x, *weights, *args[3:10]]
    before = [v.detach().cpu().clone() for v in readonly]
    expected = reference_output(*args)
    result = gems.lora_expand(*(args[:10] if defaults else args))
    assert result is None
    assert args[2] is out
    assert (out.data_ptr(), out.storage_offset(), out.stride()) == identity
    assert_output(out, expected, args, exact=exact)
    for current, tensor, original in zip(
        [args[0], *args[1], *args[3:10]], readonly, before
    ):
        assert current is tensor
        torch.testing.assert_close(
            current.cpu(), original, rtol=0, atol=0, equal_nan=True
        )


def _args(binding, dtype=torch.float16):
    args, _ = workload.build_inputs(binding, dtype, gems.device)
    return list(args)


@pytest.mark.lora_expand
@pytest.mark.parametrize("binding", workload.grid(), ids=lambda b: b["case"])
@pytest.mark.parametrize("dtype", workload.DTYPES)
def test_lora_expand(binding, dtype):
    _check(
        _args(binding, dtype),
        exact=binding["case"] in ("round_fp32_out", "mixed_low_out"),
        defaults=binding["case"] == "decode_one",
    )


@pytest.mark.lora_expand
@pytest.mark.parametrize(
    "tokens,rank,widths", [(0, 8, [11]), (1, 1, [1]), (13, 7, [37, 3, 8, 5])]
)
def test_lora_expand_boundaries(tokens, rank, widths):
    b = workload.case("boundary", tokens, rank, widths, 4, offset=3, suffix=1, add=True)
    _check(_args(b))


@pytest.mark.lora_expand
@pytest.mark.parametrize(
    "mode", ["prefix", "empty_group", "out_of_range_id", "zero_active", "no_lora"]
)
def test_lora_expand_group_controls(mode):
    args = _args(workload.case("group", 17, 8, [65, 17], 5, suffix=3, padded=True))
    if mode == "prefix":
        # Valid nonempty higher-ID groups remain in the arrays beyond this prefix.
        args[9].fill_(2)
    elif mode == "empty_group":
        # Module metadata may pad the launch; an explicitly empty valid slot is also idle.
        args[7][-1] = 1
    elif mode == "out_of_range_id":
        # The candidate excludes an invalid slot while preserving valid groups.
        old = args[7][1].item()
        args[7][1] = 9
        args[3][args[3] == old] = 9
    elif mode == "zero_active":
        args[9].zero_()
    else:
        args[8].fill_(True)
    _check(args)


@pytest.mark.lora_expand
def test_lora_expand_layout_and_live_buffers():
    b = workload.case(
        "reuse",
        7,
        8,
        [41, 13],
        4,
        offset=1,
        suffix=11,
        strided=True,
        x_fp32=True,
        add=True,
    )
    args = _args(b)
    args[1][1] = args[1][1].unsqueeze(1)  # Punica's [L,1,H,R] form.
    out = args[2]
    backing = out.new_full((out.numel() + 2,), -73)
    args[2] = backing[1:-1].view_as(out)
    args[2].copy_(out)
    # Keep the same output, weights and host control addresses across calls.
    _check(args)
    args[10] += 5
    _check(args)
    args[8].fill_(True)
    _check(args, exact=True)
    args[8].fill_(False)
    args[1][1].fill_(0.125)
    args[11] = False
    _check(args)
    old = args[1]
    args[1] = [(-old[0]).clone(), (-old[1]).clone()]
    _check(args)
    assert backing[0].item() == backing[-1].item() == -73


@pytest.mark.lora_expand
@pytest.mark.parametrize("dtype", workload.DTYPES)
@pytest.mark.parametrize("tokens", [3, 17])
def test_lora_expand_fp32_accumulation(dtype, tokens):
    args = _args(
        workload.case(
            "cancel",
            tokens,
            128,
            [33, 7],
            4,
            add=True,
            offset=2,
            suffix=3,
            out_fp32=True,
        ),
        dtype,
    )
    args[0].fill_(1)
    for s, w in enumerate(args[1]):
        w.zero_()
        w[..., 0], w[..., 1], w[..., -1] = 4096, s + 1, -4096
    # FP32 accumulation preserves the small term beside 4096 in each slice.
    _check(args, exact=True)


@pytest.mark.lora_expand
def test_lora_expand_nonfinite_values():
    args = _args(workload.case("nonfinite", 7, 4, [5], 4, offset=2, suffix=2))
    args[0].fill_(1)
    args[0][:, 1].fill_(float("nan"))  # Unscheduled base-model token.
    w = args[1][0]
    w.zero_()
    w[:, 0, 0], w[:, 1, 0], w[:, 2, 0] = float("inf"), -float("inf"), float("nan")
    _check(args, exact=True)


@pytest.mark.lora_expand
@pytest.mark.parametrize(
    "kind",
    [
        "input_ndim",
        "input_dtype",
        "empty_weights",
        "weight_rank",
        "weight_slots",
        "weight_slices",
        "weight_dtype",
        "weight_layout",
        "output_tokens",
        "output_dtype",
        "output_layout",
        "negative_offset",
        "overflow_offset",
        "mapping_length",
        "sorted_length",
        "counts_length",
        "starts_length",
        "metadata_dtype",
        "active_overflow",
        "active_negative",
        "cpu_weight",
        "cpu_indices",
        "cpu_output",
        "device_flag",
        "device_active",
        "output_alias",
    ],
)
def test_lora_expand_invalid(kind):
    b = workload.case("invalid", 3, 4, [8, 3], 4)
    args = _args(b)
    x, weights, out = args[:3]
    if kind == "input_ndim":
        args[0] = x[0]
    elif kind == "input_dtype":
        args[0] = x.to(torch.bfloat16)
    elif kind == "empty_weights":
        args[1] = []
    elif kind == "weight_rank":
        weights[1] = weights[1][..., :2].contiguous()
    elif kind == "weight_slots":
        weights[1] = weights[1][:2]
    elif kind == "weight_slices":
        weights.pop()
    elif kind == "weight_dtype":
        weights[1] = weights[1].to(torch.bfloat16)
    elif kind == "weight_layout":
        weights[0] = weights[0].new_zeros((4, 8, 8))[..., ::2]
    elif kind == "output_tokens":
        args[2] = out[:-1]
    elif kind == "output_dtype":
        args[2] = out.int()
    elif kind == "output_layout":
        args[2] = out.new_empty((3, 22))[:, ::2]
    elif kind == "negative_offset":
        args[10] = -1
    elif kind == "overflow_offset":
        args[10] = 1
    elif kind in ("mapping_length", "sorted_length", "counts_length", "starts_length"):
        index = {
            "mapping_length": 3,
            "sorted_length": 4,
            "counts_length": 5,
            "starts_length": 6,
        }[kind]
        args[index] = args[index][:-1]
    elif kind == "metadata_dtype":
        args[4] = args[4].float()
    elif kind == "active_overflow":
        args[9].fill_(args[7].numel() + 1)
    elif kind == "active_negative":
        args[9].fill_(-1)
    elif kind == "cpu_weight":
        weights[0] = weights[0].cpu()
    elif kind == "cpu_indices":
        args[4] = args[4].cpu()
    elif kind == "cpu_output":
        args[2] = out.cpu()
    elif kind == "device_flag":
        args[8] = args[8].to(gems.device)
    elif kind == "device_active":
        args[9] = args[9].to(gems.device)
    else:
        args[2] = weights[0].flatten()[: out.numel()].view_as(out)
    with pytest.raises((ValueError, TypeError)):
        gems.lora_expand(*args)
