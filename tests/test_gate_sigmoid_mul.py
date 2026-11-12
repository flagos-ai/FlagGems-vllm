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

"""Frozen mathematical coverage, independent of optional installed vLLM native."""

import math

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.runtime import torch_device_fn

DTYPES = (torch.float16, torch.bfloat16, torch.float32)
DTYPE_PAIRS = [(a, g) for a in DTYPES for g in DTYPES]
# Fixed before candidate execution. The CPU oracle rounds only the final result.
TOLERANCES = {
    torch.float16: (2e-3, 2e-5),
    torch.bfloat16: (1.6e-2, 2e-4),
    torch.float32: (2e-6, 2e-7),
}
CASES = (
    "scalar",
    "singleton",
    "empty",
    "empty_broadcast",
    "nonpower",
    "matrix",
    "rank3",
    "rank8",
    "row_gate",
    "column_gate",
    "row_a",
    "scalar_gate",
    "scalar_a",
    "transpose",
    "stepped",
    "expanded",
    "both_broadcast",
    "split_rowgap2112",
)
# Rank-8, split, and all dense and boundary cases retain their full matrix.
# Each remaining Ascend layout still checks three output dtypes and both
# directions of one mixed input pair. Rotate pairs across the layouts.
_ASCEND_LAYOUT_MIXED_PAIRS = {
    "row_gate": (torch.float16, torch.float32),
    "column_gate": (torch.bfloat16, torch.float32),
    "row_a": (torch.float16, torch.bfloat16),
    "scalar_gate": (torch.float16, torch.bfloat16),
    "scalar_a": (torch.bfloat16, torch.float32),
    "transpose": (torch.float16, torch.float32),
    "stepped": (torch.bfloat16, torch.float32),
    "expanded": (torch.float16, torch.bfloat16),
    "both_broadcast": (torch.float16, torch.float32),
}


def _layout_dtype_cases():
    for case in CASES:
        mixed = (
            _ASCEND_LAYOUT_MIXED_PAIRS.get(case)
            if flaggems_vllm.vendor_name == "ascend"
            else None
        )
        for index, pair in enumerate(DTYPE_PAIRS):
            if mixed is not None and pair not in (
                (torch.float16, torch.float16),
                (torch.bfloat16, torch.bfloat16),
                mixed,
                mixed[::-1],
            ):
                continue
            yield pytest.param(case, *pair, id=f"{case}-a_dtype{index}-g_dtype{index}")


@pytest.fixture
def device():
    selected = torch.device(flaggems_vllm.device)
    assert selected.type not in ("cpu", "meta"), "accelerator runtime required"
    return selected


def _view(shape, dtype, device, seed, strides=None, offset=3):
    """Keep prefix, suffix and strided gaps as input-write canaries."""
    if strides is None:
        strides = []
        stride = 1
        for size in reversed(shape):
            strides.insert(0, stride)
            stride *= size
    extent = (
        0
        if 0 in shape
        else 1 + sum((size - 1) * stride for size, stride in zip(shape, strides))
    )
    owner = torch.full((offset + extent + 7,), -91.0, dtype=dtype)
    view = owner.as_strided(shape, strides, offset)
    generator = torch.Generator().manual_seed(seed)
    # Distinct a and g distributions catch accidental silu(a) implementations.
    values = torch.randn(shape, generator=generator) * (0.7 if seed == 13 else 3)
    view.copy_(values.to(dtype))
    owner = owner.to(device)
    return owner.as_strided(shape, strides, offset), owner


def _inputs(case, a_dtype, g_dtype, device):
    shapes = {
        "scalar": ((), ()),
        "singleton": ((1,), (1,)),
        "empty": ((0,), (0,)),
        "empty_broadcast": ((2, 0, 7), (1, 0, 1)),
        "nonpower": ((1031,), (1031,)),
        "matrix": ((17, 129), (17, 129)),
        "rank3": ((3, 5, 19), (3, 5, 19)),
        "rank8": ((2, 1, 2, 1, 2, 1, 2, 3), (1, 1, 2, 1, 1, 1, 2, 3)),
        "row_gate": ((7, 131), (1, 131)),
        "column_gate": ((7, 131), (7, 1)),
        "row_a": ((1, 131), (7, 131)),
        "scalar_gate": ((3, 17), ()),
        "scalar_a": ((), (3, 17)),
        "transpose": ((11, 17), (11, 17)),
        "stepped": ((5, 37), (5, 37)),
        "expanded": ((1, 37), (5, 1)),
        "both_broadcast": ((2, 1, 19), (1, 3, 1)),
        "split_rowgap2112": ((5, 1024), (5, 1024)),
    }
    a_shape, g_shape = shapes[case]
    strides = None
    if case == "transpose":
        strides = (1, 11)
    elif case == "stepped":
        strides = (83, 2)
    a, a_owner = _view(a_shape, a_dtype, device, 13, strides)
    g, g_owner = _view(
        g_shape,
        g_dtype,
        device,
        29,
        (3136, 1) if case == "split_rowgap2112" else strides,
        offset=2112 if case == "split_rowgap2112" else 5,
    )
    if case == "expanded":
        a, g = a.expand(5, 37), g.expand(5, 37)
    return a, g, (a_owner, g_owner)


def _oracle(a, g):
    a64, g64 = a.cpu().double(), g.cpu().double()
    exp_abs = torch.exp(-torch.abs(g64))
    sigmoid = torch.where(g64 >= 0, 1 / (1 + exp_abs), exp_abs / (1 + exp_abs))
    return (a64 * sigmoid).to(torch.result_type(a, g))


def _bytes(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).clone()


def _check(a, g, owners, expected=None):
    snapshots = [_bytes(owner) for owner in owners]
    if expected is None:
        expected = _oracle(a, g)
    result = flaggems_vllm.gate_sigmoid_mul(a, g)
    actual = result.cpu()
    assert result.shape == torch.broadcast_shapes(a.shape, g.shape)
    assert result.dtype == torch.result_type(a, g)
    assert result.device == a.device
    assert result.is_contiguous()
    for owner in owners:
        assert result.untyped_storage().data_ptr() != owner.untyped_storage().data_ptr()
    rtol, atol = TOLERANCES[result.dtype]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    for owner, before in zip(owners, snapshots):
        assert torch.equal(_bytes(owner), before), "input storage or canary was mutated"
    return actual


@pytest.mark.gate_sigmoid_mul
@pytest.mark.parametrize("case,a_dtype,g_dtype", list(_layout_dtype_cases()))
def test_gate_sigmoid_mul_math_and_layout(case, a_dtype, g_dtype, device):
    _check(*_inputs(case, a_dtype, g_dtype, device))


@pytest.mark.gate_sigmoid_mul
@pytest.mark.parametrize(
    "dtype,width,tokens",
    [
        (dtype, width, tokens)
        for dtype in (torch.float16, torch.float32)
        for width, tokens in [(1536, 1), (3072, 17)]
    ]
    + [
        (torch.bfloat16, width, tokens)
        for width in (1536, 3072, 6144, 12288)
        for tokens in (1, 17, 512)
    ]
    + [(torch.bfloat16, 1536, 4096)],
)
def test_gate_sigmoid_mul_hotspots(dtype, width, tokens, device):
    a, a_owner = _view((tokens, width), dtype, device, 13)
    g, g_owner = _view((tokens, width), dtype, device, 29)
    _check(a, g, (a_owner, g_owner))


@pytest.mark.gate_sigmoid_mul
@pytest.mark.parametrize("dtype", DTYPES)
def test_gate_sigmoid_mul_special_values(dtype, device):
    # Use -inf, rather than a finite underflow threshold, for the Inf * 0 NaN.
    a = torch.tensor(
        [1, -1, 0, math.inf, -math.inf, math.inf, 2, 0, -0.0],
        dtype=dtype,
    ).to(device)
    g = torch.tensor(
        [-100, 100, math.nan, -math.inf, -math.inf, math.inf, math.nan, 100, -100],
        dtype=dtype,
    ).to(device)
    expected = _oracle(a, g)
    _check(a, g, (a, g), expected)


@pytest.mark.gate_sigmoid_mul
@pytest.mark.parametrize("dtype", DTYPES)
def test_gate_sigmoid_mul_compiled_fp32_exp_boundary(dtype, device):
    # Compiled-expression boundary, deliberately separate from the FP64 oracle:
    # exp(100) overflows FP32, so sigmoid(-100) becomes zero before multiplying.
    # finfo.max makes this distinction observable even with the fixed tolerances.
    limit = torch.finfo(dtype).max
    a = torch.tensor([limit, -limit, limit, -limit], dtype=dtype).to(device)
    g = torch.tensor([-100, -100, 100, 100], dtype=dtype).to(device)
    expected = torch.tensor([0.0, -0.0, limit, -limit], dtype=dtype)
    _check(a, g, (a, g), expected)


@pytest.mark.gate_sigmoid_mul
@pytest.mark.parametrize("dtype", DTYPES)
def test_gate_sigmoid_mul_signed_zero(dtype, device):
    a = torch.tensor([0.0, -0.0] * 4, dtype=dtype).to(device)
    g = torch.tensor([-10, -10, 0, 0, 10, 10, 100, 100], dtype=dtype).to(device)
    actual = _check(a, g, (a, g))
    assert torch.equal(torch.signbit(actual), torch.signbit(a.cpu()))


@pytest.mark.gate_sigmoid_mul
@pytest.mark.parametrize("shape_a,shape_g", [((2, 3), (2, 4)), ((0, 3), (2, 3))])
def test_gate_sigmoid_mul_invalid_broadcast(shape_a, shape_g, device):
    a = torch.empty(shape_a, device=device)
    g = torch.empty(shape_g, device=device)
    with pytest.raises(ValueError):
        flaggems_vllm.gate_sigmoid_mul(a, g)


@pytest.mark.gate_sigmoid_mul
@pytest.mark.parametrize(
    "dtype", [torch.bool, torch.int32, torch.float64, torch.complex64]
)
@pytest.mark.parametrize("bad_argument", [0, 1])
def test_gate_sigmoid_mul_unsupported_dtype(dtype, bad_argument, device):
    inputs = [torch.empty(3, device=device), torch.empty(3, device=device)]
    inputs[bad_argument] = torch.empty(3, dtype=dtype, device=device)
    with pytest.raises(NotImplementedError):
        flaggems_vllm.gate_sigmoid_mul(*inputs)


@pytest.mark.gate_sigmoid_mul
@pytest.mark.parametrize("host_device", ["cpu", "meta"])
def test_gate_sigmoid_mul_unsupported_device(host_device):
    a = torch.empty(3, device=host_device)
    with pytest.raises(NotImplementedError):
        flaggems_vllm.gate_sigmoid_mul(a, a)


@pytest.mark.gate_sigmoid_mul
@pytest.mark.parametrize("bad_argument", [0, 1])
def test_gate_sigmoid_mul_non_tensor(bad_argument, device):
    inputs = [torch.empty(3, device=device), torch.empty(3, device=device)]
    inputs[bad_argument] = 1.0
    with pytest.raises(TypeError):
        flaggems_vllm.gate_sigmoid_mul(*inputs)


@pytest.mark.gate_sigmoid_mul
def test_gate_sigmoid_mul_device_mismatch(device):
    a = torch.empty(3, device=device)
    g = torch.empty(3, device="cpu")
    with pytest.raises(ValueError):
        flaggems_vllm.gate_sigmoid_mul(a, g)


@pytest.mark.gate_sigmoid_mul
def test_gate_sigmoid_mul_cross_accelerator(device):
    if torch_device_fn.device_count() < 2:
        pytest.skip("cross-device rejection requires two visible accelerator devices")
    current = (
        device.index if device.index is not None else torch_device_fn.current_device()
    )
    other = (current + 1) % torch_device_fn.device_count()
    a = torch.empty(3, device=torch.device(device.type, current))
    g = torch.empty(3, device=torch.device(device.type, other))
    with pytest.raises(ValueError):
        flaggems_vllm.gate_sigmoid_mul(a, g)


@pytest.mark.gate_sigmoid_mul
def test_gate_sigmoid_mul_unsupported_rank(device):
    a = torch.empty((1,) * 9, device=device)
    with pytest.raises(NotImplementedError):
        flaggems_vllm.gate_sigmoid_mul(a, a)


@pytest.mark.gate_sigmoid_mul
def test_gate_sigmoid_mul_unsupported_layout(device):
    try:
        indices = torch.tensor([[0, 1]], device=device)
        values = torch.ones(2, device=device)
        a = torch.sparse_coo_tensor(indices, values, (3,), device=device)
    except (NotImplementedError, RuntimeError) as exc:
        reason = str(exc).lower()
        if isinstance(exc, NotImplementedError) or any(
            token in reason
            for token in (
                "not implemented",
                "not supported",
                "unsupported",
                "could not run",
            )
        ):
            pytest.skip(
                f"{device.type} cannot construct sparse COO input: {type(exc).__name__}: {exc}"
            )
        raise
    g = torch.ones(3, device=device)
    with pytest.raises(NotImplementedError):
        flaggems_vllm.gate_sigmoid_mul(a, g)
