# SPDX-License-Identifier: Apache-2.0
"""KDA64 operator baselines with state preparation outside timed regions.

The active set covers small, model-sized, and odd token counts at BF16 K,V=128.
The baseline is the original vLLM 64-token kernel chain; gate cumsum retains
its original FP32 dot reference. Kernel mode times warmed CUDA Graph replay,
while operator/wrapper mode times eager submission. Report median microseconds
and baseline/candidate SpeedUp per shape; aggregate shapes geometrically.
"""

import json
import statistics
import time

import pytest
import torch
import triton
import triton.language as tl

from flaggems_vllm.ops import kda_bounded as bounded

from . import conftest

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
ACTIVE_SET = {
    "gate64": ((1, 2, 128), (8192, 4, 128), (65, 3, 128)),
    "prefill64": ((1, 2, 128), (1024, 4, 128), (65, 3, 128)),
    "recurrent": ((1, 2, 128), (16, 4, 128), (7, 3, 128)),
}


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["g_bias"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BD": bd}, num_warps=nw) for bd in (32, 64) for nw in (2, 4, 8)
    ],
    key=["H", "D", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def _original_gate_cumsum_kernel(
    g,
    A,
    y,
    g_bias,
    cu_seqlens,
    chunk_indices,
    cumsum_scale,
    lower_bound: tl.constexpr,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos = i_b * T

    p_g = tl.make_block_ptr(
        g + (bos * H + i_h) * D,
        (T, D),
        (H * D, 1),
        (i_t * BT, i_d * BD),
        (BT, BD),
        (1, 0),
    )
    p_y = tl.make_block_ptr(
        y + (bos * H + i_h) * D,
        (T, D),
        (H * D, 1),
        (i_t * BT, i_d * BD),
        (BT, BD),
        (1, 0),
    )

    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    if HAS_BIAS:
        o_d = i_d * BD + tl.arange(0, BD)
        b_bias = tl.load(g_bias + i_h * D + o_d, mask=o_d < D, other=0.0).to(tl.float32)
        b_g += b_bias[None, :]

    b_a = tl.exp(tl.load(A + i_h).to(tl.float32))
    b_gate = lower_bound / (1.0 + tl.exp(-(b_a * b_g)))

    # Chunk-local inclusive cumsum, stored in log2 units for the downstream
    # exp2-based KDA core.
    o_t = tl.arange(0, BT)
    m_cumsum = tl.where(o_t[:, None] >= o_t[None, :], 1.0, 0.0)
    b_y = tl.dot(m_cumsum, b_gate, allow_tf32=False) * cumsum_scale
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


def _original_gate_cumsum(raw, a, bias):
    # Original PR host and tuning domain; no candidate op is used by baseline.
    b, t, h, d = raw.shape
    out = torch.empty_like(raw, dtype=torch.float32)
    _original_gate_cumsum_kernel[
        lambda meta: (triton.cdiv(d, meta["BD"]), triton.cdiv(t, 64), b * h)
    ](
        g=raw,
        A=a.reshape(-1),
        y=out,
        g_bias=None if bias is None else bias.reshape(-1),
        cu_seqlens=None,
        chunk_indices=None,
        cumsum_scale=1.4426950408889634,
        lower_bound=-5.0,
        T=t,
        H=h,
        D=d,
        BT=64,
    )
    return out


def _latency_us(call, reset):
    config = conftest.Config
    mode = config.mode.value
    with torch.no_grad():
        warm_start, warm_calls = time.perf_counter(), 0
        while warm_calls < 3 or (time.perf_counter() - warm_start) * 1000 < float(
            config.warm_up
        ):
            reset()
            call()
            torch.cuda.synchronize()
            warm_calls += 1
        if mode in ("kernel", "cudagraph"):
            graph = torch.cuda.CUDAGraph()
            reset()
            with torch.cuda.graph(graph):
                call()
            invoke = graph.replay
        else:
            invoke = call
        samples = []
        duration_ms = max(float(config.repetition), 1.0)
        while len(samples) < 5 or sum(samples) < duration_ms:
            # Copy state/input seeds before the start event. The reset kernels
            # remain outside both the captured graph and measured interval.
            reset()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            invoke()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
    return statistics.median(samples) * 1000


@pytest.mark.parametrize("family", ACTIVE_SET)
@pytest.mark.parametrize("shape_index", range(3))
@torch.no_grad()
def test_kda_bounded_perf(family, shape_index, record_property):
    from vllm.model_executor.layers.fla.ops import kda

    tokens, heads, dim = ACTIVE_SET[family][shape_index]
    torch.manual_seed(454 + shape_index)
    dtype = torch.bfloat16
    q = torch.randn((1, tokens, heads, dim), dtype=dtype, device="cuda")
    k, raw, value_seed = torch.randn_like(q), torch.randn_like(q), torch.randn_like(q)
    value = torch.empty_like(value_seed)
    beta = torch.rand((1, tokens, heads), dtype=torch.float32, device="cuda")
    a = torch.randn(heads, device="cuda")
    bias = torch.randn(heads * dim, dtype=dtype, device="cuda")
    initial = torch.randn((1, heads, dim, dim), device="cuda") * 0.1
    if family == "gate64":
        baseline = lambda: (_original_gate_cumsum(raw, a, bias), None)
        candidate = lambda: (bounded.safe_kda_gate_chunk_cumsum(raw, a, bias), None)
        reset = lambda: None
    elif family == "prefill64":

        def baseline():
            q_norm, k_norm = kda.l2norm_fwd(q), kda.l2norm_fwd(k)
            gate = _original_gate_cumsum(raw, a, bias)
            return kda._chunk_kda_fwd_with_cumulative_g(
                q=q_norm,
                k=k_norm,
                v=value,
                g=gate,
                beta=beta,
                scale=dim**-0.5,
                initial_state=initial,
                output_final_state=True,
                chunk_size=64,
            )

        candidate = lambda: bounded.chunk_kda_with_safe_gate(
            q,
            k,
            value,
            raw,
            beta,
            a,
            bias,
            initial_state=initial,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        reset = lambda: value.copy_(value_seed)
    else:
        cu = torch.arange(tokens + 1, dtype=torch.int32, device="cuda")
        slots = torch.arange(1, tokens + 1, dtype=torch.int32, device="cuda")
        state_seed = torch.randn((tokens + 1, heads, dim, dim), device="cuda") * 0.1
        state = torch.empty_like(state_seed)
        gate = bounded.safe_kda_gate(
            raw.view(tokens, heads * dim), a, dim, bias
        ).unsqueeze(0)
        baseline = lambda: kda.fused_recurrent_kda(
            q,
            k,
            value_seed,
            gate,
            beta,
            initial_state=state,
            cu_seqlens=cu,
            ssm_state_indices=slots,
        )
        candidate = lambda: bounded.fused_recurrent_kda(
            q,
            k,
            value_seed,
            gate,
            beta,
            initial_state=state,
            cu_seqlens=cu,
            ssm_state_indices=slots,
        )
        reset = lambda: state.copy_(state_seed)
    reset()
    expected, expected_state = baseline()
    expected = expected.clone()
    expected_state = None if expected_state is None else expected_state.clone()
    reset()
    actual, actual_state = candidate()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if expected_state is not None:
        torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
    baseline_us = _latency_us(baseline, reset)
    candidate_us = _latency_us(candidate, reset)
    result = {
        "op": f"kda_bounded_{family}",
        "shape": [tokens, heads, dim],
        "dtype": "bfloat16",
        "requested_mode": conftest.Config.mode.value,
        "effective_mode": (
            "cudagraph_event_with_external_reset"
            if conftest.Config.mode.value in ("kernel", "cudagraph")
            else "eager_event_with_external_reset"
        ),
        "baseline_us": baseline_us,
        "candidate_us": candidate_us,
        "speedup": baseline_us / candidate_us,
    }
    result.update(
        family=f"kda_{family}",
        shape_class=("small", "hot", "odd")[shape_index],
        latency_base_us=baseline_us,
        latency_us=candidate_us,
        SpeedUp=baseline_us / candidate_us,
        aggregation="median",
    )
    record_property("glm5_perf", json.dumps(result))
    conftest.update_result(f"kda_{family}", result)
    print(json.dumps(result, sort_keys=True))
