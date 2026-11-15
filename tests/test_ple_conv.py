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


# Large-stride address arithmetic and cache canaries.
def _reference(x, weight, histories, indices, lengths, initial, accepted, rows):
    channels, width = weight.shape
    state_len = (width - 1) * 3
    out = torch.zeros_like(x)
    expected = {}
    begin = 0
    for q, (row, length) in enumerate(zip(indices, lengths)):
        if row <= 0 or row >= rows or length == 0:
            begin += length
            continue
        history = histories[row]
        rollback = 0 if accepted is None else accepted[q] - 1
        has_history = initial[q] and (accepted is None or 0 <= rollback < 3)
        for j in range(length):
            acc = torch.zeros(channels)
            for tap in range(width):
                pos = j - (width - 1 - tap) * 3
                value = torch.zeros(channels)
                if pos >= 0:
                    value = x[begin + pos].float()
                elif has_history and rollback + state_len + pos >= 0:
                    value = history[:, rollback + state_len + pos].float()
                acc += value * weight[:, tap].float()
            rounded = acc.to(x.dtype).float()
            out[begin + j] = (rounded * torch.sigmoid(rounded)).to(x.dtype)
        updated = history.clone()
        count = state_len if accepted is None else state_len + length - 1
        for s in range(count):
            pos = s + length - state_len if accepted is None else s + 1 - state_len
            value = torch.zeros(channels, dtype=x.dtype)
            if 0 <= pos < length:
                value = x[begin + pos]
            elif pos < 0 and has_history and rollback + state_len + pos >= 0:
                value = history[:, rollback + state_len + pos]
            updated[:, s] = value
        expected[row] = updated
        begin += length
    return out, expected


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("mode", ["decode", "prefill", "spec"])
def test_ple_large_state_rows_and_changed_index_graph(dtype, mode):
    if not torch.cuda.is_available():
        pytest.skip("requires a Triton accelerator")
    rows, channels, stride, guard = 12003, 129, 204800, 128
    capacity = 11 if mode == "spec" else 9
    n = (rows - 1) * stride + channels * capacity + 2 * guard
    required = n * torch.empty((), dtype=dtype).element_size()
    if torch.cuda.mem_get_info()[0] < required + 1024**3:
        pytest.skip("large-row regression needs space for its real strided allocation")
    backing = torch.full((n,), -17.0, device="cuda", dtype=dtype)
    state = torch.as_strided(
        backing, (rows, channels, capacity), (stride, 1, channels), guard
    )
    lengths = [1] * 8 if mode == "decode" else [1, 3, 0, 2, 1, 3, 1, 1]
    starts_cpu = [0]
    for length in lengths:
        starts_cpu.append(starts_cpu[-1] + length)
    x_cpu = (
        ((torch.arange(starts_cpu[-1] * channels) % 17 - 8) / 16)
        .reshape(-1, channels)
        .to(dtype)
    )
    weight_cpu = torch.full((channels, 4), 0.25, dtype=dtype)
    x, weight = x_cpu.cuda(), weight_cpu.cuda()
    out = torch.empty_like(x)
    initial_cpu = [True, True, True, True, True, True, False, True]
    initial = torch.tensor(initial_cpu, device="cuda", dtype=torch.bool)
    accepted_cpu = [1, 2, 1, 1, 1, 1, 3, 2] if mode == "spec" else None
    accepted = (
        torch.tensor(accepted_cpu, device="cuda", dtype=torch.int32)
        if accepted_cpu is not None
        else None
    )
    starts = torch.tensor(starts_cpu, device="cuda", dtype=torch.int32)
    # int32 is intentional: it is the production index dtype, not pointer width.
    indices = torch.empty(8, device="cuda", dtype=torch.int32)
    batches = [
        [10485, 10486, 11242, 0, -1, rows, 12000, 7],
        [12000, 11242, 10486, 0, -1, rows, 10485, 8],
        [11242, 12000, 10485, 0, -1, rows, 10486, 9],
    ]
    histories = {
        row: ((torch.arange(channels * capacity) % 11 - 5) / 8)
        .reshape(channels, capacity)
        .to(dtype)
        for row in {r for batch in batches for r in batch if 0 < r < rows}
    }

    def reset(batch):
        backing.fill_(-17)
        for row, value in histories.items():
            state[row].copy_(value)
        indices.copy_(torch.tensor(batch, device="cuda", dtype=torch.int32))

    def invoke():
        ple_short_conv_(
            x,
            out,
            state,
            weight,
            indices,
            dilation=3,
            query_start_loc=None if mode == "decode" else starts,
            has_initial_state=initial,
            accepted_tokens=accepted,
            max_query_len=3 if mode == "spec" else 1,
            null_block_id=0,
        )

    def verify(batch):
        expected_out, updates = _reference(
            x_cpu,
            weight_cpu,
            histories,
            batch,
            lengths,
            initial_cpu,
            accepted_cpu,
            rows,
        )
        torch.testing.assert_close(out.cpu(), expected_out, atol=0.004, rtol=0.004)
        for row, old in histories.items():
            torch.testing.assert_close(
                state[row].cpu(), updates.get(row, old), atol=0, rtol=0
            )
        # NULL row and padding are untouched, including regions adjacent to writes.
        null_row = state[0].cpu()
        torch.testing.assert_close(
            null_row, torch.full_like(null_row, -17), atol=0, rtol=0
        )
        for row in histories:
            end = guard + row * stride + channels * capacity
            assert bool((backing[end : end + guard] == -17).all())
        assert bool((backing[:guard] == -17).all())
        assert bool((backing[-guard:] == -17).all())

    reset(batches[0])
    invoke()
    torch.cuda.synchronize()
    verify(batches[0])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()
    for batch in batches:
        reset(batch)
        graph.replay()
        torch.cuda.synchronize()
        verify(batch)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_ple_empty_input_does_not_write_state(index_dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires a Triton accelerator")
    state = torch.full((4, 17, 9), 3, device="cuda", dtype=torch.bfloat16)
    x = torch.empty((0, 17), device="cuda", dtype=torch.bfloat16)
    weight = torch.ones((17, 4), device="cuda", dtype=torch.bfloat16)
    ple_short_conv_(
        x,
        torch.empty_like(x),
        state,
        weight,
        torch.empty(0, device="cuda", dtype=index_dtype),
        dilation=3,
    )
    assert bool((state == 3).all())


# Geometry, exact in-place state, and changed-input graph regressions.
DTYPES = [torch.bfloat16, torch.float16, torch.float32]
MODES = ["decode", "prefill", "spec"]


def oracle(x, initial, weight, ids, kw):
    output, updated = torch.zeros_like(x), initial.clone()
    width, dilation = weight.shape[1], kw["dilation"]
    history = (width - 1) * dilation
    starts = kw.get("query_start_loc")
    starts = starts.tolist() if starts is not None else list(range(len(ids) + 1))
    flags, accepted = kw.get("has_initial_state"), kw.get("accepted_tokens")
    for req, row in enumerate(ids):
        begin, end = starts[req : req + 2]
        if (
            row == kw.get("null_block_id", -1)
            or not 0 <= row < initial.shape[0]
            or begin == end
        ):
            continue
        rollback = 0 if accepted is None else int(accepted[req]) - 1
        has = (flags is None or bool(flags[req])) and (
            accepted is None or 0 <= rollback < kw["max_query_len"]
        )
        for t in range(end - begin):
            acc = torch.zeros(x.shape[1], dtype=torch.float32)
            for tap in range(width):
                pos = t - (width - 1 - tap) * dilation
                value = torch.zeros_like(acc)
                if pos >= 0:
                    value = x[begin + pos].float()
                elif has and rollback + history + pos >= 0:
                    value = (
                        initial[row, :, rollback + history + pos].to(x.dtype).float()
                    )
                acc += value * weight[:, tap].float()
            rounded = acc.to(x.dtype).float()
            output[begin + t] = (rounded * torch.sigmoid(rounded)).to(x.dtype)
        if not history:
            continue
        count = history if accepted is None else history + end - begin - 1
        for s in range(count):
            pos = s + end - begin - history if accepted is None else s + 1 - history
            value = torch.zeros(x.shape[1], dtype=x.dtype)
            if 0 <= pos < end - begin:
                value = x[begin + pos]
            elif pos < 0 and has and rollback + history + pos >= 0:
                value = initial[row, :, rollback + history + pos].to(x.dtype)
            updated[row, :, s] = value.to(initial.dtype)
    return output, updated


def state_on_device(initial, layout):
    rows, channels, capacity = initial.shape
    if layout == "channel-contiguous":
        state = torch.empty(
            (rows, capacity, channels), device="cuda", dtype=initial.dtype
        ).transpose(1, 2)
    else:
        state = torch.empty_like(
            initial, device="cuda", memory_format=torch.contiguous_format
        )
    state.copy_(initial)
    return state


def execute(x_cpu, initial_cpu, weight_cpu, ids_cpu, kw_cpu, layout, eager_repeats):
    x, weight = x_cpu.cuda(), weight_cpu.cuda()
    state = state_on_device(initial_cpu, layout)
    pristine = state.clone()
    indices = torch.tensor(
        ids_cpu, device="cuda", dtype=kw_cpu.pop("index_dtype", torch.int32)
    )
    kw = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in kw_cpu.items()}
    output = torch.empty_like(x)
    tol = {torch.bfloat16: 2e-2, torch.float16: 3e-3, torch.float32: 2e-5}[x.dtype]

    def reset(phase):
        state.copy_(pristine)
        x.copy_(x_cpu * ([1, 0.5, -0.5, 1][phase]))
        valid = sorted({r for r in ids_cpu if r > 0 and r < initial_cpu.shape[0]})
        mapping = {r: valid[(i + phase) % len(valid)] for i, r in enumerate(valid)}
        ids = [mapping.get(r, r) for r in ids_cpu]
        indices.copy_(torch.tensor(ids, device="cuda", dtype=indices.dtype))
        flags = kw_cpu.get("has_initial_state")
        if flags is not None:
            kw["has_initial_state"].copy_(~flags if phase % 2 else flags)
        output.fill_(float("nan"))
        return ids

    def invoke():
        ple_short_conv_(x, output, state, weight, indices, **kw)

    def verify(ids):
        torch.cuda.synchronize()
        current_kw = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in kw.items()
        }
        expected_output, expected_state = oracle(
            x.cpu(), initial_cpu, weight_cpu, ids, current_kw
        )
        torch.testing.assert_close(output.cpu(), expected_output, atol=tol, rtol=tol)
        torch.testing.assert_close(state.cpu(), expected_state, atol=0, rtol=0)

    for _ in range(eager_repeats):
        ids = reset(0)
        invoke()
        verify(ids)
    reset(0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()
    for phase in (1, 2, 3):
        ids = reset(phase)
        graph.replay()
        verify(ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton accelerator required")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("width", range(1, 9))
@pytest.mark.parametrize("dilation", [1, 3])
def test_all_widths_dilations_dtype_modes(dtype, mode, width, dilation):
    channels = [1, 33, 129][width % 3]
    history = (width - 1) * dilation
    capacity = max(1, history + (3 if mode == "spec" else 0))
    lengths = (
        [1] * 8
        if mode == "decode"
        else ([1, 3, 0, 2, 1, 3, 1, 2] if mode == "spec" else [0, 1, 9, 2, 3, 0, 1, 4])
    )
    starts = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
    )
    tokens = int(starts[-1]) + (0 if mode == "decode" else 2)
    x = (
        ((torch.arange(tokens * channels) % 17 - 8) / 16)
        .reshape(tokens, channels)
        .to(dtype)
    )
    weight = torch.full((channels, width), 0.25, dtype=dtype)
    state = (
        ((torch.arange(8 * channels * capacity) % 13 - 6) / 8)
        .reshape(8, channels, capacity)
        .to(dtype)
    )
    kw = {
        "dilation": dilation,
        "null_block_id": 0,
        "has_initial_state": torch.tensor(
            [True, False, True, True, False, True, True, False]
        ),
    }
    if mode != "decode":
        kw["query_start_loc"] = starts
    if mode == "spec":
        kw.update(
            accepted_tokens=torch.tensor([1, 2, 0, 4, 5, -1, 4, 1], dtype=torch.int32),
            max_query_len=4,
        )
    execute(
        x,
        state,
        weight,
        [1, 2, 3, 0, -1, 8, 4, 5],
        kw,
        "channel-contiguous" if width % 2 else "contiguous",
        1,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton accelerator required")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("layout", ["contiguous", "channel-contiguous"])
def test_production_channels_history20_and_graph(dtype, mode, index_dtype, layout):
    # Reuse the maintained synthetic fixture; no saved model inputs are needed.
    torch.manual_seed(73049)
    args, kw = inputs(mode, torch.bfloat16, channels=10240)
    x, _, old, weight, indices = args
    old = old.cpu().to(dtype)
    state = torch.full((old.shape[0], 10240, 20), -17, dtype=dtype)
    state[:, :, : old.shape[2]] = old
    kwargs = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in kw.items()}
    kwargs["index_dtype"] = index_dtype
    execute(
        x.cpu().to(dtype),
        state,
        weight.cpu().to(dtype),
        indices.cpu().tolist(),
        kwargs,
        layout,
        3,
    )
