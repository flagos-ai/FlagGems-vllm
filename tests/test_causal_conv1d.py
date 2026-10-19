# SPDX-License-Identifier: Apache-2.0
"""Stateful causal convolution against an independent sequence oracle."""

import importlib
import math

import pytest
import torch

import flaggems_vllm as gems

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def _reference(sequences, state, weight, bias, slots, initial, activation):
    final = state.clone()
    outputs = []
    width = weight.shape[-1]
    for req, x in enumerate(sequences):
        if x.shape[-1] == 0:
            outputs.append(torch.empty_like(x))
            continue
        slot = slots[req]
        if slot < 0:
            outputs.append(torch.zeros_like(x))
            continue
        history = (
            state[slot, :, : width - 1].clone()
            if initial[req]
            else torch.zeros_like(state[slot, :, : width - 1])
        )
        y = torch.empty_like(x)
        for token in range(x.shape[-1]):
            # Materialize input in the state's dtype before FP32 arithmetic.
            value = x[:, token].to(state.dtype).float().unsqueeze(-1)
            window = torch.cat((history.float(), value), dim=-1)
            result = (window * weight.float()).sum(-1)
            if bias is not None:
                result += bias.float()
            if activation:
                result = result * torch.sigmoid(result)
            y[:, token] = result.to(state.dtype).to(x.dtype)
            history = window[:, 1:].to(state.dtype)
        final[slot, :, : width - 1] = history
        outputs.append(y)
    return outputs, final


def _close(actual, expected):
    eps = 2e-5 if actual.dtype == torch.float32 else 2 * torch.finfo(actual.dtype).eps
    torch.testing.assert_close(actual, expected, rtol=eps, atol=eps)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [2, 3, 4, 8])
@pytest.mark.parametrize("activation", [None, "silu"])
def test_prefill_varlen_strides_padding_and_initial_state(dtype, width, activation):
    torch.manual_seed(552)
    dim = 37
    lengths = [2, 0, 5, 3]
    starts = torch.tensor([0, 2, 2, 7, 10], dtype=torch.int32, device=gems.device)
    x = torch.randn(dim * 2, 20, dtype=dtype, device=gems.device)[::2, ::2]
    state = torch.randn(6, dim, (width + 2) * 2, dtype=dtype, device=gems.device)[
        ..., ::2
    ]
    weight = torch.randn(dim, width * 2, dtype=dtype, device=gems.device)[:, ::2]
    bias = torch.randn(dim, dtype=dtype, device=gems.device) * 0.1
    slots = torch.tensor([4, 1, -1, 3], dtype=torch.int32, device=gems.device)
    initial = torch.tensor([True, True, False, False], device=gems.device)
    sequences = list(x.split(lengths, dim=-1))
    expected, final = _reference(
        sequences,
        state,
        weight,
        bias,
        [4, 1, -1, 3],
        [True, True, False, False],
        activation is not None,
    )
    actual = gems.causal_conv1d_fn(
        x,
        weight,
        bias,
        state,
        starts,
        cache_indices=slots,
        has_initial_state=initial,
        activation=activation,
    )
    _close(actual, torch.cat(expected, dim=-1))
    torch.testing.assert_close(state, final, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens", [0, 1, 5])
def test_decode_regular_and_varlen_state_equality(dtype, tokens):
    torch.manual_seed(91)
    batch, dim, width = 3, 7, 4
    x = torch.randn(batch, dim, tokens, dtype=dtype, device=gems.device)
    state = torch.randn(5, dim, 6, dtype=dtype, device=gems.device)
    state_varlen = state.clone()
    weight = torch.randn(dim, width, dtype=dtype, device=gems.device)
    slots = torch.tensor([3, -1, 1], dtype=torch.int64, device=gems.device)
    expected, final = _reference(
        list(x.unbind()), state, weight, None, [3, -1, 1], [True] * 3, True
    )
    regular = gems.causal_conv1d_update(
        x, state, weight, activation="swish", conv_state_indices=slots
    )
    _close(regular, torch.stack(expected))
    torch.testing.assert_close(state, final, rtol=0, atol=0)
    flat = x.transpose(1, 2).reshape(-1, dim)
    starts = torch.tensor(
        [0, tokens, 2 * tokens, 3 * tokens], dtype=torch.int32, device=gems.device
    )
    varlen = gems.causal_conv1d_update(
        flat,
        state_varlen,
        weight,
        activation="silu",
        conv_state_indices=slots,
        query_start_loc=starts,
    )
    _close(varlen, torch.stack(expected).transpose(1, 2).reshape(-1, dim))
    torch.testing.assert_close(state_varlen, final, rtol=0, atol=0)


def test_single_token_decode_preserves_rank_and_state_dtype_rounding():
    x = torch.tensor([[0.3333333, 0.6666666, 1.000001]], device=gems.device)
    state = torch.randn(2, 3, 4, dtype=torch.bfloat16, device=gems.device)
    weight = torch.randn(3, 3, device=gems.device)
    slots = torch.tensor([1], dtype=torch.int32, device=gems.device)
    expected, final = _reference([x.t()], state, weight, None, [1], [True], False)
    actual = gems.causal_conv1d_update(x, state, weight, conv_state_indices=slots)
    assert actual.shape == x.shape and actual.dtype == x.dtype
    _close(actual, expected[0].t())
    torch.testing.assert_close(state, final, rtol=0, atol=0)


def _run_layout(mode, sequences, state, weight, bias, slots, activation, initial=None):
    """Return all layouts as [dim, total_tokens] for hand-calculated checks."""
    indices = torch.tensor(slots, dtype=torch.int32, device=gems.device)
    if mode == "regular":
        output = gems.causal_conv1d_update(
            torch.stack(sequences),
            state,
            weight,
            bias,
            activation=activation,
            conv_state_indices=indices,
        )
        return torch.cat(list(output.unbind()), dim=-1)
    boundaries = [0]
    for sequence in sequences:
        boundaries.append(boundaries[-1] + sequence.shape[-1])
    starts = torch.tensor(boundaries, dtype=torch.int32, device=gems.device)
    x = torch.cat(sequences, dim=-1)
    if mode == "prefill":
        return gems.causal_conv1d_fn(
            x,
            weight,
            bias,
            state,
            starts,
            cache_indices=indices,
            has_initial_state=initial,
            activation=activation,
        )
    return gems.causal_conv1d_update(
        x.t(),
        state,
        weight,
        bias,
        activation=activation,
        conv_state_indices=indices,
        query_start_loc=starts,
    ).t()


@pytest.mark.parametrize("mode", ["prefill", "regular", "varlen"])
def test_history_prefix_and_extra_storage_against_hand_calculation(mode):
    # The suffix is deliberately very different from the first width-1 values.
    # It belongs to extra cache storage, not to this non-speculative history.
    state = torch.tensor(
        [
            [[1, 2, 3, 701, 702, 703, 704], [4, 5, 6, 711, 712, 713, 714]],
            [[101, 102, 103, 104, 105, 106, 107]] * 2,
            [[7, 8, 9, 721, 722, 723, 724], [10, 11, 12, 731, 732, 733, 734]],
        ],
        dtype=torch.float32,
        device=gems.device,
    )
    expected_state = state.clone()
    expected_state[2, :, :3] = torch.tensor(
        [[9, 10, 11], [12, 13, 14]], device=gems.device
    )
    expected_state[0, :, :3] = torch.tensor([[3, 4, 5], [6, 7, 8]], device=gems.device)
    sequences = [
        torch.tensor([[10, 11], [13, 14]], dtype=torch.float32, device=gems.device),
        torch.tensor([[4, 5], [7, 8]], dtype=torch.float32, device=gems.device),
    ]
    weight = torch.tensor([[1, 10, 100, 1000], [0.5, -1, 2, -3]], device=gems.device)
    expected = torch.tensor(
        [[10987, 12098, 4321, 5432], [-21, -22.5, -12, -13.5]],
        device=gems.device,
    )
    actual = _run_layout(mode, sequences, state, weight, None, [2, 0], None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["prefill", "regular", "varlen"])
@pytest.mark.parametrize("state_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation", [None, "silu"])
def test_fp32_output_materializes_activated_result_in_state_dtype(
    mode, state_dtype, activation
):
    state = torch.full((3, 3, 5), -77, dtype=state_dtype, device=gems.device)
    prefix = [0.25, 0.5, 0.75]
    state[1, :, 0] = torch.tensor(prefix, dtype=state_dtype, device=gems.device)
    expected_state = state.clone()
    values = [0.3333333, 0.6666666, 1.000001]
    x = torch.tensor(values, dtype=torch.float32, device=gems.device).unsqueeze(-1)
    rounded_x = torch.tensor(values, dtype=state_dtype).float().tolist()
    # One width-2 step: history * 1/2 + materialized_input * 1/4 + 1/8.
    # These binary-exact coefficients isolate the required output rounding.
    results = [old * 0.5 + new * 0.25 + 0.125 for old, new in zip(prefix, rounded_x)]
    if activation:
        results = [value / (1 + math.exp(-value)) for value in results]
    expected = torch.tensor(results, dtype=state_dtype).float().to(gems.device)
    expected = expected.unsqueeze(-1)
    unrounded = torch.tensor(results, dtype=torch.float32)
    assert not torch.equal(unrounded, expected.cpu().squeeze(-1))
    expected_state[1, :, 0] = x[:, 0].to(state_dtype)
    weight = torch.tensor([[0.5, 0.25]] * 3, device=gems.device)
    bias = torch.full((3,), 0.125, device=gems.device)
    actual = _run_layout(mode, [x], state, weight, bias, [1], activation)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)


def _empty_request_data(all_empty):
    state = torch.tensor(
        [
            [[1, 2, 51, 52, 53], [7, 8, 54, 55, 56]],
            [[11, 12, 61, 62, 63], [17, 18, 64, 65, 66]],
            [[21, 22, 71, 72, 73], [27, 28, 74, 75, 76]],
        ],
        dtype=torch.float32,
        device=gems.device,
    )
    empty = torch.empty(2, 0, device=gems.device)
    tokens = torch.tensor([[3, 4], [5, 6]], dtype=torch.float32, device=gems.device)
    sequences = [empty, empty if all_empty else tokens, empty]
    weight = torch.tensor([[1, 2, 3]] * 2, dtype=torch.float32, device=gems.device)
    expected_state = state.clone()
    if not all_empty:
        expected_state[0, :, :2] = tokens
    return sequences, state, expected_state, weight


@pytest.mark.parametrize("all_empty", [False, True])
@pytest.mark.parametrize("has_initial_state", [False, True])
def test_prefill_empty_requests_preserve_cache(all_empty, has_initial_state):
    sequences, state, expected_state, weight = _empty_request_data(all_empty)
    initial = torch.full((3,), has_initial_state, device=gems.device)
    actual = _run_layout(
        "prefill", sequences, state, weight, None, [2, 0, 1], None, initial
    )
    if all_empty:
        expected = torch.empty(2, 0, device=gems.device)
    else:
        values = [[14, 20], [38, 36]] if has_initial_state else [[9, 18], [15, 28]]
        expected = torch.tensor(values, dtype=torch.float32, device=gems.device)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("mode", "all_empty"), [("regular", True), ("varlen", False), ("varlen", True)]
)
def test_decode_empty_requests_preserve_cache(mode, all_empty):
    sequences, state, expected_state, weight = _empty_request_data(all_empty)
    actual = _run_layout(mode, sequences, state, weight, None, [2, 0, 1], None)
    expected = (
        torch.empty(2, 0, device=gems.device)
        if all_empty
        else torch.tensor([[14, 20], [38, 36]], dtype=torch.float32, device=gems.device)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)


@pytest.mark.parametrize(
    "invalid",
    [
        "bias_shape",
        "state_width",
        "device",
        "slots_shape",
        "x_channels",
        "activation",
        "apc",
        "bias_stride",
        "slots_dtype",
        "slots_stride",
    ],
)
def test_conv_preflight_preserves_state_and_does_not_launch(invalid, monkeypatch):
    module = importlib.import_module("flaggems_vllm.ops.causal_conv1d")
    calls = []

    class RejectLaunch:
        def __getitem__(self, grid):
            calls.append(grid)
            raise AssertionError("invalid convolution launched")

    monkeypatch.setattr(module, "_conv_kernel", RejectLaunch())
    x = torch.randn(2, 7, device=gems.device)
    state = torch.randn(3, 7, 4, device=gems.device)
    weight = torch.randn(7, 4, device=gems.device)
    bias = torch.randn(7, device=gems.device)
    slots = torch.tensor([0, 2], dtype=torch.int32, device=gems.device)
    extra = {}
    if invalid == "bias_shape":
        bias = bias[:6]
    elif invalid == "state_width":
        state = state[..., :2]
    elif invalid == "device":
        bias = bias.cpu()
    elif invalid == "slots_shape":
        slots = slots[:1]
    elif invalid == "x_channels":
        x = x[:, :6]
    elif invalid == "bias_stride":
        bias = torch.randn(14, device=gems.device)[::2]
    elif invalid == "slots_dtype":
        slots = slots.float()
    elif invalid == "slots_stride":
        slots = torch.tensor([0, 1, 2, 1], dtype=torch.int32, device=gems.device)[::2]
    elif invalid == "activation":
        extra["activation"] = "relu"
    else:
        extra["num_accepted_tokens"] = torch.ones(
            2, dtype=torch.int32, device=gems.device
        )
    before = state.clone()
    with pytest.raises((ValueError, NotImplementedError)):
        gems.causal_conv1d_update(
            x, state, weight, bias, conv_state_indices=slots, **extra
        )
    assert not calls
    torch.testing.assert_close(state, before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "invalid", ["starts_dtype", "starts_stride", "initial_dtype", "initial_stride"]
)
def test_prefill_metadata_preflight_does_not_write_state(invalid, monkeypatch):
    module = importlib.import_module("flaggems_vllm.ops.causal_conv1d")
    calls = []

    class RejectLaunch:
        def __getitem__(self, grid):
            calls.append(grid)
            raise AssertionError("invalid prefill metadata launched")

    monkeypatch.setattr(module, "_conv_kernel", RejectLaunch())
    x = torch.randn(7, 3, device=gems.device)
    state = torch.randn(3, 7, 4, device=gems.device)
    weight = torch.randn(7, 4, device=gems.device)
    starts = torch.tensor([0, 2, 3], dtype=torch.int32, device=gems.device)
    slots = torch.tensor([0, 2], dtype=torch.int32, device=gems.device)
    initial = torch.tensor([True, False], device=gems.device)
    if invalid == "starts_dtype":
        starts = starts.float()
    elif invalid == "starts_stride":
        starts = torch.tensor([0, 0, 2, 0, 3], dtype=torch.int32, device=gems.device)[
            ::2
        ]
    elif invalid == "initial_dtype":
        initial = initial.float()
    else:
        initial = torch.tensor([True, False, False, True], device=gems.device)[::2]
    before = state.clone()
    with pytest.raises(ValueError):
        gems.causal_conv1d_fn(
            x,
            weight,
            None,
            state,
            starts,
            cache_indices=slots,
            has_initial_state=initial,
        )
    assert not calls
    torch.testing.assert_close(state, before, rtol=0, atol=0)
