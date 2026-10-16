# SPDX-License-Identifier: Apache-2.0
"""Numerical/state contracts for the PLE path extracted from the plugin."""

import pytest
import torch
import torch.nn.functional as F

from flaggems_vllm import ple_short_conv_

pytestmark = pytest.mark.ple_fusion


def reference(
    x,
    output,
    state,
    weight,
    indices,
    *,
    dilation,
    query_start_loc=None,
    has_initial_state=None,
    accepted_tokens=None,
    max_query_len=1,
    null_block_id=-1,
):
    """Independent per-request native conv oracle, including cache mutation."""
    state_len = (weight.shape[1] - 1) * dilation
    output.zero_()
    if indices is None:
        requests = [(0, x.shape[0], None)]
    else:
        starts = (
            query_start_loc.tolist()
            if query_start_loc is not None
            else list(range(x.shape[0] + 1))
        )
        requests = [
            (starts[r], starts[r + 1], i) for r, i in enumerate(indices.tolist())
        ]
    for r, (start, end, index) in enumerate(requests):
        if (
            end == start
            or index == null_block_id
            or (index is not None and not 0 <= index < state.shape[0])
        ):
            continue
        initial = x.new_zeros((x.shape[1], state_len))
        rollback = 0 if accepted_tokens is None else int(accepted_tokens[r]) - 1
        has_history = (
            index is not None
            and (has_initial_state is None or bool(has_initial_state[r]))
            and 0 <= rollback < max_query_len
        )
        if has_history:
            initial = state[index, :, rollback : rollback + state_len].to(x.dtype)
        history = torch.cat((initial, x[start:end].T), dim=1)
        conv = F.conv1d(
            history[None], weight[:, None], groups=x.shape[1], dilation=dilation
        )
        output[start:end].copy_(F.silu(conv[0]).T)
        if index is not None and state_len:
            if accepted_tokens is None:
                state[index, :, :state_len].copy_(history[:, -state_len:])
            else:
                count = state_len + end - start - 1
                state[index, :, :count].copy_(history[:, 1 : count + 1])


def inputs(mode, dtype, channels=33, rows=4, device="cuda"):
    dilation, width, max_query = 3, 4, 4
    state = torch.randn(rows + 2, 12, channels, dtype=dtype, device=device).transpose(
        1, 2
    )
    weights = torch.randn(channels, width, dtype=dtype, device=device) * 0.1
    lengths = (
        [1] * rows
        if mode == "decode"
        else ([4] * rows if mode == "spec" else [3] * rows)
    )
    starts = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device=device, dtype=torch.int32
    )
    if mode == "prefill":
        starts[1] = starts[0]  # Empty first request, then a longer second request.
    tokens = int(starts[-1]) + (2 if mode in ("prefill", "spec") else 0)
    x = torch.randn(tokens, channels, dtype=dtype, device=device)
    output = torch.empty_like(x)
    indices = torch.arange(1, rows + 1, dtype=torch.int32, device=device)
    indices[-1] = -1
    kwargs = {"dilation": dilation}
    if mode == "profile":
        indices = None
    elif mode == "decode":
        flags = torch.ones(rows, dtype=torch.bool, device=device)
        flags[0] = False
        kwargs["has_initial_state"] = flags
    else:
        kwargs["query_start_loc"] = starts
        if mode == "spec":
            kwargs.update(
                accepted_tokens=torch.tensor(
                    [1, 2, 0, 4], device=device, dtype=torch.int32
                ),
                max_query_len=max_query,
            )
        else:
            kwargs["has_initial_state"] = torch.tensor(
                [False, True, False, True], device=device
            )
    return (x, output, state, weights, indices), kwargs


def check(args, kwargs):
    x, out, state, weight, indices = args
    expected_state = state.clone()
    expected_out = torch.empty_like(out)
    reference(x, expected_out, expected_state, weight, indices, **kwargs)
    ple_short_conv_(*args, **kwargs)
    # Existing PLE precision contract: do not weaken the migrated tolerance.
    tol = {torch.bfloat16: 2e-2, torch.float16: 3e-3, torch.float32: 2e-5}[x.dtype]
    torch.testing.assert_close(out, expected_out, atol=tol, rtol=tol)
    torch.testing.assert_close(state, expected_state, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("mode", ["decode", "prefill", "spec", "profile"])
def test_ple_conv_and_strided_cache_match_native(mode, dtype):
    torch.manual_seed(73)
    args, kwargs = inputs(mode, dtype)
    check(args, kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["decode", "prefill", "spec"])
def test_ple_conv_graph_refreshes_inputs_and_state(mode):
    args, kwargs = inputs(mode, torch.bfloat16)
    check(args, kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ple_short_conv_(*args, **kwargs)
    x, output, state, weight, indices = args
    x.normal_()
    state.normal_()
    output.fill_(float("nan"))
    indices[0] = 2
    indices[1] = 1
    if mode != "decode":
        kwargs["query_start_loc"][1] = 2
    if mode == "spec":
        kwargs["query_start_loc"].copy_(
            torch.tensor([0, 2, 6, 10, 14], device=x.device, dtype=torch.int32)
        )
        kwargs["accepted_tokens"][0] = 3
    expected_state = state.clone()
    expected = torch.empty_like(output)
    reference(x, expected, expected_state, weight, indices, **kwargs)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(state, expected_state, atol=0, rtol=0)


def test_ple_conv_rejects_cpu_before_launch():
    args, kwargs = inputs("decode", torch.float32, device="cpu")
    with pytest.raises(NotImplementedError, match="accelerator"):
        ple_short_conv_(*args, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["decode", "prefill", "spec"])
def test_ple_conv_production_channels(mode):
    args, kwargs = inputs(mode, torch.bfloat16, channels=10240)
    check(args, kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ple_conv_empty_and_single_tap_preserve_cache():
    args, kwargs = inputs("decode", torch.float32)
    x, output, state, weight, indices = args
    original = state.clone()
    ple_short_conv_(x[:0], output[:0], state, weight, indices[:0], dilation=3)
    torch.testing.assert_close(state, original, atol=0, rtol=0)
    # Noncontiguous input/output views and a single tap need no state update.
    backing = torch.randn(x.shape[0], x.shape[1] * 2, device=x.device)
    x = backing[:, ::2]
    output = torch.empty_like(backing)[:, ::2]
    check((x, output, state, weight[:, :1], indices), kwargs)
