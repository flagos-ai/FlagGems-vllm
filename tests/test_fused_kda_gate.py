# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""KDA gate tests and an unchanged, pinned GLM5Next vLLM reference.

Reference: vllm-project/vllm@a686c8e1b7e5870eadfc9f47447de2b8a26211fc
vllm/models/glm5next/nvidia/ops/third_party/kda/kernels.py
The two reference functions below are copied verbatim, including autotuning.
Source SHA256: 6f5865164682e2ea475995fbdeb1df6942baecfb06abeee3a4468c7c6344775e
SPDX-FileCopyrightText: Copyright contributors to the vLLM project
SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
Original FLA copyright (c) 2023-2025, Songlin Yang, Yu Zhang.
"""

import inspect
import json
import os
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl
from triton import cdiv, next_power_of_2

import flaggems_vllm
from flaggems_vllm import fused_kda_gate as gems_fused_kda_gate

from . import accuracy_utils as utils
from . import conftest as cfg

if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":
    from triton.language.extra.libdevice import fast_logf as log
else:
    log = tl.log

BT_LIST_AUTOTUNE = [32, 64, 128]
NUM_WARPS_AUTOTUNE = [4, 8, 16, 32]


@triton.autotune(
    configs=[
        triton.Config({"BT": bt}, num_warps=nw, num_stages=ns)
        for bt in BT_LIST_AUTOTUNE
        for nw in NUM_WARPS_AUTOTUNE
        for ns in [2, 3]
    ],
    key=["H", "D"],
)
@triton.jit
def kda_gate_fwd_kernel(
    g,
    A,
    y,
    g_bias,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    SAFE_GATE: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    T,
    H,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t, i_h = tl.program_id(0).to(tl.int64), tl.program_id(1)
    n_t = i_t * BT

    b_a = tl.load(A + i_h).to(tl.float32)
    b_a = tl.exp(b_a) if SAFE_GATE else -tl.exp(b_a)

    stride_row = H * D
    stride_col = 1

    o_t = n_t + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_t = o_t < T
    m_g = m_t[:, None] & (o_d[None, :] < D)
    g_ptr = g + i_h * D + o_t[:, None] * stride_row + o_d[None, :] * stride_col
    y_ptr = y + i_h * D + o_t[:, None] * stride_row + o_d[None, :] * stride_col

    b_g = tl.load(g_ptr, mask=m_g, other=0.0).to(tl.float32)

    if HAS_BIAS:
        n_d = tl.arange(0, BD)
        bias_mask = n_d < D
        b_bias = tl.load(g_bias + i_h * D + n_d, mask=bias_mask, other=0.0).to(
            tl.float32
        )
        b_g = b_g + b_bias[None, :]

    if SAFE_GATE:
        # y = lower_bound * sigmoid(exp(A) * (g + g_bias)), bounded to
        # (lower_bound, 0) for safe-gate checkpoints.
        b_y = LOWER_BOUND / (1.0 + tl.exp(-(b_a * b_g)))
    else:
        # softplus(x, beta) = (1/beta) * log(1 + exp(beta * x))
        # When beta * x > threshold, use linear approximation x
        # Use threshold to switch to linear when beta*x > threshold
        g_scaled = b_g * beta
        use_linear = g_scaled > threshold
        sp = tl.where(use_linear, b_g, (1.0 / beta) * log(1.0 + tl.exp(g_scaled)))
        b_y = b_a * sp

    tl.store(y_ptr, b_y.to(y.dtype.element_ty), mask=m_g)


def fused_kda_gate(
    g: torch.Tensor,
    A: torch.Tensor,
    head_k_dim: int,
    g_bias: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
    safe_gate: bool = False,
    lower_bound: float | None = -5.0,
) -> torch.Tensor:
    """Forward pass for KDA gate:
    input g: [..., H*D]
    param A: [H] or [1, 1, H, 1]
    beta: softplus beta parameter (softplus branch only)
    threshold: softplus threshold parameter (softplus branch only)
    safe_gate: when False (default) compute y = -exp(A)*softplus(g+g_bias);
      when True compute the bounded y = lower_bound*sigmoid(exp(A)*(g+g_bias))
    lower_bound: floor for the safe_gate branch (default -5.0)
    return  : [..., H, D]
    """
    orig_shape = g.shape[:-1]

    g = g.view(-1, g.shape[-1])
    T = g.shape[0]
    HD = g.shape[1]
    H = A.numel()
    assert H * head_k_dim == HD

    y = torch.empty_like(g, dtype=torch.float32)

    def grid(meta):
        return (cdiv(T, meta["BT"]), H)

    kda_gate_fwd_kernel[grid](
        g,
        A,
        y,
        g_bias,
        beta,
        threshold,
        safe_gate,
        lower_bound if lower_bound is not None else -5.0,
        T,
        H,
        head_k_dim,
        BD=next_power_of_2(head_k_dim),
        HAS_BIAS=g_bias is not None,
    )

    y = y.view(*orig_shape, H, head_k_dim)
    return y


ref_fused_kda_gate = fused_kda_gate
pytestmark = [
    pytest.mark.fused_kda_gate,
    pytest.mark.skipif(
        flaggems_vllm.vendor_name != "nvidia" or not torch.cuda.is_available(),
        reason="requires NVIDIA CUDA",
    ),
]
DTYPES = (
    [torch.bfloat16]
    if cfg.QUICK_MODE
    else [torch.float16, torch.bfloat16, torch.float32]
)
SHAPES = (
    [
        (1, 8, 64),
        (8, 8, 128),
        (64, 96, 128),
    ]
    if cfg.QUICK_MODE
    else [
        (1, 8, 64),
        (8, 8, 128),
        (64, 96, 128),
        (300, 96, 64),
        (37, 12, 80),
        (1024, 8, 128),
    ]
)


def torch_fused_kda_gate(
    g,
    A,
    head_k_dim,
    g_bias=None,
    beta=1.0,
    threshold=20.0,
    safe_gate=False,
    lower_bound=-5.0,
):
    # Do not mutate g: .float() can alias an FP32 input.
    value = g.float().reshape(-1, A.numel(), head_k_dim)
    if g_bias is not None:
        value = value + g_bias.float().reshape(A.numel(), head_k_dim)
    scale = A.float().reshape(-1, 1).exp()
    if safe_gate:
        lower_bound = -5.0 if lower_bound is None else lower_bound
        result = lower_bound / (1.0 + torch.exp(-(scale * value)))
    else:
        scaled = value * beta
        sp = torch.where(
            scaled > threshold,
            value,
            (1.0 / beta) * torch.log(1.0 + torch.exp(scaled)),
        )
        result = -scale * sp
    return result.reshape(*g.shape[:-1], A.numel(), head_k_dim)


def assert_output(g, A, D, bias=None, **kwargs):
    out = gems_fused_kda_gate(g, A, D, bias, **kwargs)
    ref = torch_fused_kda_gate(
        utils.to_reference(g),
        utils.to_reference(A),
        D,
        utils.to_reference(bias) if bias is not None else None,
        **kwargs,
    )
    assert out.shape == (*g.shape[:-1], A.numel(), D)
    assert out.dtype == torch.float32 and out.device == g.device
    assert not out.requires_grad
    utils.gems_assert_close(out, ref, torch.float32, equal_nan=True)
    return out


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("safe_gate", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
def test_fused_kda_gate(shape, dtype, safe_gate, has_bias):
    utils.init_seed(42)
    T, H, D = shape
    g = torch.randn(T, H * D, dtype=dtype, device=flaggems_vllm.device)
    A = torch.randn(H, device=g.device) * 0.5
    bias = torch.randn(H, D, device=g.device) * 0.1 if has_bias else None
    assert_output(g, A, D, bias, safe_gate=safe_gate)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("safe_gate", [False, True])
def test_against_pinned_vllm(dtype, safe_gate):
    g = torch.randn(8, 8 * 128, device=flaggems_vllm.device, dtype=dtype)
    A = torch.randn(1, 1, 8, 1, device=g.device)
    bias = torch.randn(8, 128, device=g.device)
    kwargs = dict(beta=0.5, threshold=10.0, safe_gate=safe_gate, lower_bound=None)
    ref = ref_fused_kda_gate(g, A, 128, bias, **kwargs)
    out = gems_fused_kda_gate(g, A, 128, bias, **kwargs)
    utils.gems_assert_close(out, utils.to_reference(ref), torch.float32)


@pytest.mark.parametrize("shape", [(16,), (1, 16), (2, 3, 16), (1, 2, 3, 16)])
@pytest.mark.parametrize("safe_gate", [False, True])
def test_rank_and_offset(shape, safe_gate):
    count = 1
    for dim in shape:
        count *= dim
    g = torch.randn(count + 1, device=flaggems_vllm.device)[1:].view(shape)
    A = torch.randn(3, device=g.device)[1:]
    bias = torch.randn(17, device=g.device)[1:].view(2, 8)
    before = g.clone()
    out = assert_output(g, A, 8, bias, safe_gate=safe_gate)
    second = gems_fused_kda_gate(g, A, 8, bias, safe_gate=safe_gate)
    assert out.data_ptr() != second.data_ptr()
    torch.testing.assert_close(g, before, rtol=0, atol=0)


@pytest.mark.parametrize("safe_gate", [False, True])
@pytest.mark.parametrize("lower_bound", [None, -2.0, 0.0])
@pytest.mark.parametrize("beta,threshold", [(0.5, 5.0), (2.0, 10.0)])
def test_scalar_parameters(safe_gate, lower_bound, beta, threshold):
    g = torch.linspace(-30, 30, 48, device=flaggems_vllm.device).reshape(3, 16)
    A = torch.tensor([-0.5, 0.5], device=g.device)
    assert_output(
        g,
        A,
        8,
        beta=beta,
        threshold=threshold,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
    )


@pytest.mark.parametrize("safe_gate", [False, True])
def test_nonfinite_and_threshold(safe_gate):
    g = torch.tensor(
        [
            float("-inf"),
            -0.0,
            0.0,
            float("inf"),
            float("nan"),
            -100.0,
            -20.0,
            19.99,
            20.0,
            20.01,
            100.0,
            -1.0,
        ],
        device=flaggems_vllm.device,
    ).reshape(1, 12)
    A = torch.tensor([0.0], device=g.device)
    assert_output(g, A, 12, safe_gate=safe_gate)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_mixed_parameter_dtypes(dtype):
    g = torch.randn(3, 16, dtype=torch.float32, device=flaggems_vllm.device)
    A = torch.randn(2, dtype=dtype, device=g.device)
    bias = torch.randn(16, dtype=dtype, device=g.device)
    assert_output(g, A, 8, bias, safe_gate=True)
    g = g.to(dtype)
    assert_output(g, A.float(), 8, safe_gate=False)


@pytest.mark.parametrize("shape", [(0, 16), (2, 0, 16)])
def test_empty(shape):
    g = torch.empty(shape, device=flaggems_vllm.device)
    A = torch.empty(2, device=g.device)
    assert_output(g, A, 8)


@pytest.mark.parametrize("T", [4097, 16384] if not cfg.QUICK_MODE else [37])
def test_prefill(T):
    g = torch.randn(T, 8 * 128, device=flaggems_vllm.device, dtype=torch.bfloat16)
    A = torch.randn(8, device=g.device)
    assert_output(g, A, 128, safe_gate=True)


@pytest.mark.parametrize(
    "case",
    [
        "head_dim",
        "hidden",
        "bias_size",
        "g_layout",
        "A_layout",
        "bias_layout",
        "dtype",
        "device",
    ],
)
def test_invalid_input(case):
    g = torch.randn(2, 16, device=flaggems_vllm.device)
    A = torch.randn(2, device=g.device)
    bias = torch.randn(16, device=g.device)
    D, error = 8, ValueError
    if case == "head_dim":
        D = 0
    elif case == "hidden":
        g = g[:, :8].contiguous()
    elif case == "bias_size":
        bias = bias[:8]
    elif case == "g_layout":
        g = torch.randn(2, 32, device=g.device)[:, ::2]
        error = NotImplementedError
    elif case == "A_layout":
        A = torch.randn(4, device=g.device)[::2]
        error = NotImplementedError
    elif case == "bias_layout":
        bias = torch.randn(32, device=g.device)[::2]
        error = NotImplementedError
    elif case == "dtype":
        g = g.to(torch.int32)
        error = NotImplementedError
    elif case == "device":
        A = A.cpu()
    with pytest.raises(error):
        gems_fused_kda_gate(g, A, D, bias)


def test_public_export_and_signature():
    assert flaggems_vllm.ops.fused_kda_gate is gems_fused_kda_gate
    assert flaggems_vllm.ops.FLA.fused_kda_gate is gems_fused_kda_gate
    assert ("fused_kda_gate", gems_fused_kda_gate) in flaggems_vllm._FULL_CONFIG
    assert inspect.signature(gems_fused_kda_gate) == inspect.signature(
        ref_fused_kda_gate
    )


@pytest.mark.parametrize("safe_gate", [False, True])
def test_repeated_metadata_reads_current_values(safe_gate):
    device = flaggems_vllm.device
    g = torch.randn(8, 1024, device=device, dtype=torch.bfloat16)
    A = torch.randn(8, device=device)
    bias = torch.randn(8, 128, device=device)
    assert_output(g, A, 128, bias, safe_gate=safe_gate)
    g.add_(0.5)
    A.mul_(0.5)
    bias.sub_(0.25)
    # The same metadata must launch with the new values, on the current stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        assert_output(g, A, 128, bias, safe_gate=safe_gate)
        replacement = torch.randn_like(g)
        assert_output(replacement, A.clone(), 128, bias.clone(), safe_gate=safe_gate)
    torch.cuda.current_stream().wait_stream(stream)


def test_cuda_graph_replay_reads_current_values():
    g = torch.randn(8, 1024, device=flaggems_vllm.device, dtype=torch.bfloat16)
    A = torch.randn(8, device=g.device)
    bias = torch.randn(1024, device=g.device)
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        gems_fused_kda_gate(g, A, 128, bias, safe_gate=True)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = gems_fused_kda_gate(g, A, 128, bias, safe_gate=True)
    g.add_(0.25)
    graph.replay()
    ref = torch_fused_kda_gate(g, A, 128, bias, safe_gate=True)
    utils.gems_assert_close(out, utils.to_reference(ref), torch.float32)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_device_guard():
    original = torch.cuda.current_device()
    other = (original + 1) % torch.cuda.device_count()
    g = torch.randn(8, 1024, device=f"cuda:{other}", dtype=torch.bfloat16)
    A = torch.randn(8, device=g.device)
    assert_output(g, A, 128, safe_gate=True)
    assert_output(g, A, 128, safe_gate=True)
    assert torch.cuda.current_device() == original


@pytest.mark.parametrize("safe_gate", [False, True])
def test_large_head_count(safe_gate):
    T, H, D = 16385, 96, 128
    g = torch.randn(T, H * D, device=flaggems_vllm.device, dtype=torch.bfloat16)
    A = torch.randn(H, device=g.device)
    bias = torch.randn(H, D, device=g.device)
    out = gems_fused_kda_gate(g, A, D, bias, safe_gate=safe_gate)
    # Check the head mapping at distant rows and the partially filled last block.
    rows = torch.tensor([0, 31, 32, T - 2, T - 1], device=g.device)
    ref = torch_fused_kda_gate(g.index_select(0, rows), A, D, bias, safe_gate=safe_gate)
    utils.gems_assert_close(
        out.index_select(0, rows), utils.to_reference(ref), torch.float32
    )


@pytest.mark.parametrize("safe_gate", [False, True])
@pytest.mark.parametrize("dtype", DTYPES + ([] if cfg.QUICK_MODE else [torch.float64]))
def test_tiled_dtypes_and_tail(dtype, safe_gate):
    T, H, D = 5003, 12, 80
    g = torch.randn(T, H * D, device=flaggems_vllm.device, dtype=dtype)
    A = torch.randn(1, 1, H, 1, device=g.device)
    bias = torch.randn(H, D, device=g.device)
    assert_output(g, A, D, bias, safe_gate=safe_gate)


def captured_gate_cases(payload):
    """Deduplicate executed contracts without losing dtype, stride or scalars."""
    if payload.get("status") != "shapes_collected":
        raise ValueError("model shape collection must have completed successfully")
    cases = {}
    fields = (
        "operator_g",
        "A",
        "g_bias",
        "D",
        "beta",
        "threshold",
        "safe_gate",
        "lower_bound",
    )
    for workload in payload["cases"]:
        for worker in workload["workers"]:
            for row in worker["shapes"]:
                case = {field: row[field] for field in fields}
                cases[json.dumps(case, sort_keys=True)] = case
    if not cases:
        raise ValueError("no executed gate shapes found")
    return [cases[key] for key in sorted(cases)]


def captured_gate_inputs(case, device):
    def tensor(metadata):
        if metadata is None:
            return None
        dtype = getattr(torch, metadata["dtype"].removeprefix("torch."))
        return torch.empty_strided(
            metadata["shape"], metadata["stride"], dtype=dtype, device=device
        ).normal_(std=0.5)

    return (
        tensor(case["operator_g"]),
        tensor(case["A"]),
        case["D"],
        tensor(case["g_bias"]),
        case["beta"],
        case["threshold"],
        case["safe_gate"],
        case["lower_bound"],
    )


_capture_path = os.environ.get("FUSED_KDA_GATE_SHAPES")
_captured_cases = (
    captured_gate_cases(json.loads(Path(_capture_path).read_text(encoding="utf-8")))
    if _capture_path
    else []
)


@pytest.mark.parametrize("case", _captured_cases)
@pytest.mark.parametrize("execution", ["eager", "cudagraph"])
def test_captured_model_shapes(case, execution):
    utils.init_seed(42)
    args = captured_gate_inputs(case, flaggems_vllm.device)
    if execution == "cudagraph":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            gems_fused_kda_gate(*args)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = gems_fused_kda_gate(*args)
        # Replay must read current data rather than the values used for capture.
        args[0].add_(0.25)
        graph.replay()
    else:
        out = gems_fused_kda_gate(*args)
    native = ref_fused_kda_gate(*args)
    torch_ref = torch_fused_kda_gate(*args)
    assert out.shape == (*args[0].shape[:-1], args[1].numel(), args[2])
    assert out.dtype == torch.float32
    utils.gems_assert_close(out, utils.to_reference(native), torch.float32)
    utils.gems_assert_close(out, utils.to_reference(torch_ref), torch.float32)
