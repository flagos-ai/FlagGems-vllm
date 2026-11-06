# SPDX-License-Identifier: Apache-2.0
"""GLM numeric active set. Latency is microseconds; SpeedUp=Torch/Gems.

Pure operators honor the repository kernel/cudagraph helpers. Stateful cases
report a distinct event metric with resets outside every interval, including
warmup. Reset/copy time is excluded and state never accumulates across samples.
Use --level core for the small and hot cases; comprehensive also runs odd and
large cases. --warmup/--iter retain the repository's millisecond budget units.
"""

import json
import math
import statistics
import time

import pytest
import torch
import torch.nn.functional as F
import triton

import flaggems_vllm as gems

from . import conftest

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

# Every family has a fixed small / hot / odd / large active set. No shape is
# removed in response to a slow or inaccurate result.
ACTIVE = {
    "copy": [(1, 128), (512, 1280), (17, 257), (4096, 4096)],
    "gather": [(1, 128), (512, 1280), (17, 257), (4096, 4096)],
    "scatter": [(1, 128), (512, 1280), (17, 257), (4096, 4096)],
    "concat": [(1, 128), (512, 512), (17, 257), (4096, 512)],
    "concat_zero_rope": [(1, 128), (512, 512), (17, 257), (4096, 512)],
    "pad_heads": [(1, 128), (512, 512), (17, 257), (4096, 512)],
    "mhc_rms_norm": [(1, 128), (512, 1280), (17, 257), (4096, 4096)],
    "mhc_pre_with_norm": [(1, 128), (512, 1280), (17, 257), (4096, 1280)],
    "mhc_fused_post_pre_with_norm": [(1, 128), (512, 1280), (17, 257), (4096, 1280)],
    "hadamard128": [(1, 128), (512, 128), (17, 128), (4096, 128)],
    "fwht128_quant_fp8": [(1, 128), (512, 128), (17, 128), (4096, 128)],
    "causal_conv1d_update": [(1, 128), (64, 1280), (7, 257), (256, 4096)],
}


def _hadamard_reference(x):
    out = x.float()
    size = 1
    while size < 128:
        pair = out.reshape(*x.shape[:-1], -1, 2, size)
        a, b = pair[..., 0, :], pair[..., 1, :]
        out = torch.stack((a + b, a - b), dim=-2).reshape(x.shape)
        size *= 2
    return (out * (1 / math.sqrt(128))).to(x.dtype)


def _make_case(family, rows, dim):
    torch.manual_seed(20261008 + list(ACTIVE).index(family))
    device = gems.device
    reset = lambda: None
    x = torch.randn(rows, dim, dtype=torch.bfloat16, device=device)
    if family == "copy":
        x = x.t()
        return (
            lambda: x.contiguous().float(),
            lambda: gems.contiguous_copy(x, torch.float32),
            reset,
        )
    if family == "gather":
        idx = torch.arange(rows - 1, -1, -1, device=device, dtype=torch.int64)
        return lambda: x[idx], lambda: gems.gather_rows(x, idx), reset
    if family == "scatter":
        requests = max(1, rows // 4)
        lmax = math.ceil(rows / requests) + 1
        index = torch.arange(rows, dtype=torch.int32, device=device)
        req, pos = index % requests, index // requests

        def scatter_ref():
            out = torch.full((requests, lmax, dim), -3, dtype=x.dtype, device=device)
            out[req.long(), pos.long()] = x
            return out

        return (
            scatter_ref,
            lambda: gems.scatter_decode_tokens(x, -3, requests, lmax, (req, pos)),
            reset,
        )
    if family in {"concat", "concat_zero_rope"}:
        a = x.view(rows, 1, dim)
        b = torch.randn(
            rows,
            1,
            0 if family == "concat_zero_rope" else 64,
            dtype=x.dtype,
            device=device,
        )
        return lambda: torch.cat((a, b), dim=-1), lambda: gems.concat_query(a, b), reset
    if family == "pad_heads":
        a = x.view(rows, 1, dim).expand(rows, 2, dim)
        return (
            lambda: F.pad(a, (0, 0, 0, 2)),
            lambda: gems.pad_attention_heads(a, 4),
            reset,
        )
    if family == "mhc_rms_norm":
        weight = torch.randn(dim, dtype=x.dtype, device=device)

        def norm_ref():
            value = x.float()
            return (
                value
                * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
                * weight.float()
            ).to(x.dtype)

        return norm_ref, lambda: gems.mhc_rms_norm(x, weight, 1e-6), reset
    if family in {"mhc_pre_with_norm", "mhc_fused_post_pre_with_norm"}:
        residual = torch.randn(rows, 4, dim, dtype=torch.bfloat16, device=device)
        fn = torch.randn(24, 4 * dim, dtype=torch.float32, device=device) * 0.003
        scale = torch.tensor([0.1, 0.3, 0.2], device=device)
        base = torch.randn(24, device=device) * 0.1
        norm = torch.randn(dim, dtype=torch.bfloat16, device=device)

        def pre_ref(residual=residual):
            flat = residual.flatten(-2)
            projection = (flat @ fn.bfloat16().t()).bfloat16().float()
            expanded = torch.cat(
                (scale[0].expand(4), scale[1].expand(4), scale[2].expand(16))
            )
            mix = (
                projection
                * torch.rsqrt(flat.float().square().mean(-1, keepdim=True) + 1e-6)
                * expanded
                + base
            )
            pre = mix[:, :4].sigmoid().unsqueeze(-1) + 1e-6
            post = mix[:, 4:8].sigmoid().unsqueeze(-1)
            comb = mix[:, 8:].view(-1, 4, 4).softmax(-1) + 1e-6
            comb /= comb.sum(-2, keepdim=True) + 1e-6
            for _ in range(4):
                comb /= comb.sum(-1, keepdim=True) + 1e-6
                comb /= comb.sum(-2, keepdim=True) + 1e-6
            value = (residual.float() * pre).sum(-2).bfloat16().float()
            normalized = (
                value
                * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
                * norm.float()
            ).bfloat16()
            return post, comb, normalized

        if family == "mhc_fused_post_pre_with_norm":
            layer = torch.randn(rows, dim, dtype=torch.bfloat16, device=device)
            post_mix = torch.randn(rows, 4, 1, device=device) * 0.3
            comb_mix = torch.randn(rows, 4, 4, device=device) * 0.2

            def fused_ref():
                # Common unchanged post stage isolates the new pre/norm ABI
                # from alternate FP32 summation orders at its BF16 boundary.
                current = gems.mhc_post(layer, residual, post_mix, comb_mix)
                return (current, *pre_ref(current))

            return (
                fused_ref,
                lambda: gems.mhc_fused_post_pre_with_norm(
                    layer,
                    residual,
                    post_mix,
                    comb_mix,
                    fn,
                    scale,
                    base,
                    1e-6,
                    1e-6,
                    1e-6,
                    1.0,
                    5,
                    norm_weight=norm,
                    norm_eps=1e-6,
                ),
                reset,
            )
        return (
            pre_ref,
            lambda: gems.mhc_pre_with_norm(
                residual,
                fn,
                scale,
                base,
                1e-6,
                1e-6,
                1e-6,
                1.0,
                5,
                norm_weight=norm,
                norm_eps=1e-6,
            ),
            reset,
        )
    if family == "hadamard128":
        return lambda: _hadamard_reference(x), lambda: gems.hadamard128(x), reset
    if family == "fwht128_quant_fp8":

        def quant_ref():
            rotated = _hadamard_reference(x).float()
            scale = torch.pow(
                2.0,
                torch.ceil(
                    torch.log2(
                        rotated.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448.0
                    )
                ),
            )
            return (rotated / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale

        return quant_ref, lambda: gems.fwht128_quant_fp8(x), reset
    if family == "causal_conv1d_update":
        state_initial = torch.randn(rows, dim, 3, dtype=x.dtype, device=device)
        state = state_initial.clone()
        weight = torch.randn(dim, 4, dtype=x.dtype, device=device)
        bias = torch.randn(dim, dtype=x.dtype, device=device)

        def conv_ref():
            window = torch.cat(
                (state[..., :3].float(), x.to(state.dtype).float().unsqueeze(-1)),
                dim=-1,
            )
            value = (window * weight.float().unsqueeze(0)).sum(-1) + bias.float()
            state[..., :3].copy_(window[..., 1:].to(state.dtype))
            return F.silu(value).to(state.dtype).to(x.dtype)

        def reset_state():
            state.copy_(state_initial)

        reset_state.mutates = True
        return (
            conv_ref,
            lambda: gems.causal_conv1d_update(
                x, state, weight, bias, activation="silu"
            ),
            reset_state,
        )
    raise AssertionError(f"Unknown benchmark family: {family}")


def _event_latency_us(fn, reset, warmup_ms, budget_ms):
    deadline = time.perf_counter() + warmup_ms / 1000.0
    while time.perf_counter() < deadline:
        reset()
        fn()
    torch.cuda.synchronize()
    samples = []
    deadline = time.perf_counter() + budget_ms / 1000.0
    # At least five independent samples even for a 1 ms smoke budget. Each
    # reset is submitted before start and therefore outside the GPU interval.
    while len(samples) < 5 or time.perf_counter() < deadline:
        reset()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


def _latency_us(fn, reset, warmup_ms, budget_ms, mode):
    mutable = getattr(reset, "mutates", False)
    if mode == "kernel" and not mutable:
        return (
            triton.testing.do_bench(
                fn, warmup=warmup_ms, rep=budget_ms, return_mode="median"
            )
            * 1000.0
        )
    if mode == "cudagraph" and not mutable:
        return (
            triton.testing.do_bench_cudagraph(fn, rep=budget_ms, return_mode="median")
            * 1000.0
        )
    if mode in {"kernel", "cudagraph"}:
        # Triton's ordinary helper has no per-sample prepare hook. Calling
        # reset inside fn would count reset time or let state accumulate.
        # Stateful cases therefore report this distinct event-based metric.
        if mode == "cudagraph":
            reset()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                retained_output = fn()
            fn = graph.replay
            # Keep all graph-owned result addresses live until measurement.
            assert retained_output is not None
        return _event_latency_us(fn, reset, warmup_ms, budget_ms)
    samples = []
    deadline = time.perf_counter() + warmup_ms / 1000.0
    while time.perf_counter() < deadline:
        reset()
        fn()
    torch.cuda.synchronize()
    deadline = time.perf_counter() + budget_ms / 1000.0
    while len(samples) < 5 or time.perf_counter() < deadline:
        reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if mode == "operator":
            torch.cuda.synchronize()
        elif mode != "wrapper":
            raise ValueError(f"Unsupported benchmark mode: {mode}")
        samples.append((time.perf_counter() - start) * 1e6)
        torch.cuda.synchronize()
    return statistics.median(samples)


def _effective_mode(mode, reset):
    if getattr(reset, "mutates", False) and mode in {"kernel", "cudagraph"}:
        return f"{mode}_event_with_external_state_reset"
    return mode


@torch.no_grad()
@pytest.mark.parametrize("family", tuple(ACTIVE))
@pytest.mark.parametrize("shape_class", ["small", "hot", "odd", "large"])
def test_glm5_numeric_performance(family, shape_class, request, record_property):
    if request.config.getoption("level") == "core" and shape_class in {"odd", "large"}:
        pytest.skip("comprehensive active set case")
    shape = ACTIVE[family][["small", "hot", "odd", "large"].index(shape_class)]
    baseline, candidate, reset = _make_case(family, *shape)
    reset()
    expected = baseline()
    reset()
    actual = candidate()
    if family == "fwht128_quant_fp8":
        # FP8 codec must preserve bytes; scale is exact power-of-two FP32.
        torch.testing.assert_close(
            actual[0].view(torch.uint8), expected[0].view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    else:
        tolerance = (
            0.01
            if family
            in {
                "mhc_rms_norm",
                "mhc_pre_with_norm",
                "mhc_fused_post_pre_with_norm",
                "causal_conv1d_update",
            }
            else 0
        )
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    warmup = float(request.config.getoption("warmup"))
    budget = float(request.config.getoption("iter"))
    mode = request.config.getoption("mode")
    base_us = _latency_us(baseline, reset, warmup, budget, mode)
    kernel_us = _latency_us(candidate, reset, warmup, budget, mode)
    result = {
        "family": family,
        "baseline": (
            "existing_post+torch_pre_norm"
            if family == "mhc_fused_post_pre_with_norm"
            else "torch"
        ),
        "shape_class": shape_class,
        "shape": shape,
        "dtype": "bfloat16",
        "requested_mode": mode,
        "effective_mode": _effective_mode(mode, reset),
        "latency_base_us": base_us,
        "latency_us": kernel_us,
        "SpeedUp": base_us / kernel_us,
        "aggregation": "median",
        "direction": "higher SpeedUp is better",
    }
    record_property("glm5_perf", json.dumps(result))
    conftest.update_result(family, result)
    print(json.dumps(result))
