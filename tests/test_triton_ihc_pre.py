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

"""Independent CPU-oracle coverage for the public iHC pre entry point.

FP64 reductions/projection avoid reproducing the kernel's partition or reduction
order. Gate arithmetic and the channel sum use FP32, followed by an explicit
rounding to x.dtype. Large prefills are checked in bounded CPU chunks.
"""

import pytest
import torch

import flaggems_vllm
from baseline import triton_ihc_pre as workloads

from . import conftest as cfg

DEVICE = flaggems_vllm.device
DTYPES = (torch.float16, torch.bfloat16)
pytestmark = pytest.mark.usefixtures("_ihc_pre_cpu_threads")


@pytest.fixture
def _ihc_pre_cpu_threads():
    # Each pytest worker owns its process-local Torch thread pool. Restrict only
    # this operator's accuracy tests, including input snapshots and comparisons.
    with workloads._cpu_single_thread():
        yield


@workloads._cpu_single_thread()
def ihc_pre_reference(x, weight, scale, base, magnitude, hc_eps, norm_eps):
    """Return CPU (y, post), independently of production/native implementations."""
    n, hc, d = x.shape
    # Inputs may already be CPU snapshots when called by a benchmark.
    weight64 = weight.detach().to(device="cpu", dtype=torch.float64)
    scale32 = scale.detach().to(device="cpu", dtype=torch.float32)
    base32 = base.detach().to(device="cpu", dtype=torch.float32)
    y = torch.empty((n, d), dtype=x.dtype, device="cpu")
    post = torch.empty((n, hc), dtype=torch.float32, device="cpu")
    for start in range(0, n, 32):
        part = slice(start, min(start + 32, n))
        x32 = x[part].detach().to(device="cpu", dtype=torch.float32)
        flat64 = x32.reshape(-1, hc * d).double()
        sumsq32 = flat64.square().sum(dim=1, keepdim=True).float()
        rms32 = torch.rsqrt(sumsq32 / (hc * d) + norm_eps)
        mix32 = (flat64 @ weight64.T).float()
        pre32 = torch.sigmoid(mix32[:, :hc] * rms32 * scale32[0] + base32[:hc])
        pre32 = pre32 + hc_eps
        post[part] = (
            magnitude * torch.sigmoid(mix32[:, hc:] * rms32 * scale32[1] + base32[hc:])
            + hc_eps
        )
        # Sum in channel order in FP32, then round exactly once to the ABI dtype.
        reduced32 = torch.zeros((x32.shape[0], d), dtype=torch.float32)
        for channel in range(hc):
            reduced32 = reduced32 + pre32[:, channel, None] * x32[:, channel, :]
        y[part] = reduced32.to(dtype=x.dtype)
    return y, post


def _inputs(n, hc, d, dtype):
    generator = torch.Generator(device="cpu").manual_seed(42)
    x = torch.randn((n, hc, d), generator=generator).to(dtype=dtype, device=DEVICE)
    weight = torch.randn((2 * hc, hc * d), generator=generator, device="cpu")
    scale = torch.tensor([0.01, -0.02], dtype=torch.float32)
    base = torch.randn((2 * hc,), generator=generator, device="cpu")
    return [x, weight.to(DEVICE), scale.to(DEVICE), base.to(DEVICE)]


def _check(inputs, scalars=(2.0, 1e-6, 1e-5), op=None):
    if op is None:
        op = flaggems_vllm.triton_ihc_pre
    before = [t.detach().cpu().clone() for t in inputs]
    outputs = op(*inputs, *scalars)
    assert isinstance(outputs, tuple) and len(outputs) == 2
    y, post = outputs
    n, hc, d = inputs[0].shape
    assert y.shape == (n, d) and post.shape == (n, hc)
    assert y.dtype == inputs[0].dtype and post.dtype == torch.float32
    assert y.device == inputs[0].device and post.device == inputs[0].device
    expected_y, expected_post = ihc_pre_reference(*before, *scalars)
    torch.testing.assert_close(
        y.cpu(), expected_y, rtol=1e-2, atol=2e-2, equal_nan=True
    )
    torch.testing.assert_close(
        post.cpu(), expected_post, rtol=1e-5, atol=2e-5, equal_nan=True
    )
    for original, snapshot in zip(inputs, before):
        assert torch.equal(
            original.detach().cpu().contiguous().view(torch.uint8),
            snapshot.contiguous().view(torch.uint8),
        )
    return y, post


# Paired, rather than Cartesian, coverage: every required N/HC/D is represented.
# The legacy test_ihc.py retains all its original HC=4 model-sized pre cases.
QUICK_SHAPES = [
    (0, 1, 1),
    (1, 1, 1),
    (3, 3, 7),
    (7, 4, 33),
    (255, 4, 17),
    (256, 4, 17),
    (257, 4, 17),
    (1, 4, 4096),
]
FULL_SHAPES = QUICK_SHAPES + [
    (31, 1, 63),
    (63, 3, 129),
    (7, 4, 6144),
    (255, 4, 4096),
    (256, 4, 4096),
    (257, 4, 4096),
    (255, 4, 6144),
    (256, 4, 6144),
    (257, 4, 6144),
    (4096, 4, 4096),
    (4096, 4, 6144),
    (8192, 4, 4096),
    (8192, 4, 6144),
]


@pytest.mark.triton_ihc_pre
@pytest.mark.parametrize("shape", QUICK_SHAPES if cfg.QUICK_MODE else FULL_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_triton_ihc_pre_shapes(shape, dtype):
    _check(_inputs(*shape, dtype))


BOUNDARIES = [
    "zero_weight",
    "zero_scale",
    "zero_base",
    "all_zero",
    "negative_scale",
    "saturated_base",
    "magnitude_zero",
    "magnitude_one",
    "hc_eps_zero",
    "hc_eps_large",
    "norm_eps_large",
    "x_zero",
    "x_tiny",
    "x_large",
    "x_cancellation",
    "x_nan",
    "x_pos_inf",
    "x_neg_inf",
    "x_signed_zero",
]


@pytest.mark.triton_ihc_pre
@pytest.mark.parametrize("case", BOUNDARIES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_triton_ihc_pre_boundaries(case, dtype):
    inputs = _inputs(3, 3, 17, dtype)
    x, weight, scale, base = inputs
    magnitude, hc_eps, norm_eps = 2.0, 1e-6, 1e-5
    if case in ("zero_weight", "all_zero"):
        weight.zero_()
    if case in ("zero_scale", "all_zero"):
        scale.zero_()
    if case in ("zero_base", "all_zero"):
        base.zero_()
    if case == "negative_scale":
        scale.copy_(torch.tensor([-0.25, -0.5], device=DEVICE))
    elif case == "saturated_base":
        base.copy_(torch.tensor([-100, 100, -100, 100, -100, 100], device=DEVICE))
    elif case == "magnitude_zero":
        magnitude = 0.0
    elif case == "magnitude_one":
        magnitude = 1.0
    elif case == "hc_eps_zero":
        hc_eps = 0.0
    elif case == "hc_eps_large":
        hc_eps = 0.125
    elif case == "norm_eps_large":
        norm_eps = 0.25
    elif case == "x_zero":
        x.zero_()
    elif case == "x_tiny":
        x.fill_(torch.finfo(dtype).tiny / 2)
    elif case == "x_large":
        x.mul_(256)
    elif case == "x_cancellation":
        pattern = (torch.arange(17, device=DEVICE) % 2) * 2 - 1
        x.copy_(pattern[None, None, :])
        weight.fill_(0.03125)
    elif case in ("x_nan", "x_pos_inf", "x_neg_inf"):
        x[0, 0, 0] = {
            "x_nan": float("nan"),
            "x_pos_inf": float("inf"),
            "x_neg_inf": -float("inf"),
        }[case]
    elif case == "x_signed_zero":
        x.zero_()
        x[:, ::2, :] = -0.0
    _check(inputs, (magnitude, hc_eps, norm_eps))


def _strided_copy(tensor, axis):
    shape = list(tensor.shape)
    shape[axis] *= 2
    padded = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
    slices = [slice(None)] * tensor.ndim
    slices[axis] = slice(None, None, 2)
    view = padded[tuple(slices)]
    view.copy_(tensor)
    return view


LAYOUTS = [
    "x_hidden_stride",
    "x_channel_stride",
    "x_token_stride",
    "x_transpose",
    "x_expand",
    "scale_stride",
    "base_stride",
    "all_strided",
    "weight_storage_offset",
]


@pytest.mark.triton_ihc_pre
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_triton_ihc_pre_layouts(layout, dtype):
    inputs = _inputs(7, 3, 17, dtype)
    if layout in ("x_hidden_stride", "all_strided"):
        inputs[0] = _strided_copy(inputs[0], 2)
    elif layout == "x_channel_stride":
        inputs[0] = _strided_copy(inputs[0], 1)
    elif layout == "x_token_stride":
        inputs[0] = _strided_copy(inputs[0], 0)
    elif layout == "x_transpose":
        inputs[0] = inputs[0].transpose(1, 2).contiguous().transpose(1, 2)
    elif layout == "x_expand":
        inputs[0] = inputs[0][:1].expand(7, -1, -1)
    if layout in ("scale_stride", "all_strided"):
        inputs[2] = _strided_copy(inputs[2], 0)
    if layout in ("base_stride", "all_strided"):
        inputs[3] = _strided_copy(inputs[3], 0)
    if layout == "weight_storage_offset":
        padded = torch.empty((8, 51), dtype=torch.float32, device=DEVICE)
        inputs[1] = padded[1:7]
        inputs[1].copy_(_inputs(7, 3, 17, dtype)[1])
        assert inputs[1].is_contiguous() and inputs[1].storage_offset() > 0
    _check(inputs)


@pytest.mark.triton_ihc_pre
@pytest.mark.parametrize("dtype", DTYPES)
def test_triton_ihc_pre_legacy_compatibility(dtype):
    inputs = _inputs(7, 3, 17, dtype)
    scalars = (1.0, 0.125, 0.25)
    _check(inputs, scalars)
    _check(inputs, scalars, op=flaggems_vllm.ihc_pre)


@pytest.mark.triton_ihc_pre
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d", [4096, 6144])
def test_triton_ihc_pre_same_token_across_dispatch_threshold(dtype, d):
    """The same tokens retain both outputs across N=255/256/257 dispatches."""
    inputs = _inputs(257, 4, d, dtype)
    full_y, full_post = _check(inputs)
    for n in (1, 255, 256):
        subset = [inputs[0][:n], *inputs[1:]]
        y, post = _check(subset)
        torch.testing.assert_close(y, full_y[:n], rtol=1e-2, atol=2e-2)
        torch.testing.assert_close(post, full_post[:n], rtol=1e-5, atol=2e-5)
        # Repeated launches use the same established numerical tolerances.
        again_y, again_post = flaggems_vllm.triton_ihc_pre(*subset, 2.0, 1e-6, 1e-5)
        torch.testing.assert_close(y, again_y, rtol=1e-2, atol=2e-2)
        torch.testing.assert_close(post, again_post, rtol=1e-5, atol=2e-5)


INVALID_CASES = [
    "x_rank2",
    "x_rank4",
    "hc_zero",
    "d_zero",
    "weight_rank1",
    "weight_rows",
    "weight_columns",
    "scale_rank2",
    "scale_length",
    "base_rank2",
    "base_length",
    "weight_strided",
    "x_float32",
    "x_int32",
    "weight_float16",
    "scale_float16",
    "base_float16",
    "x_not_tensor",
    "weight_not_tensor",
    "scale_not_tensor",
    "base_not_tensor",
    "magnitude_tensor",
    "hc_eps_complex",
    "norm_eps_string",
]


@pytest.mark.triton_ihc_pre
@pytest.mark.parametrize("case", INVALID_CASES)
@pytest.mark.parametrize("n", [0, 3])
def test_triton_ihc_pre_rejects_invalid_inputs(case, n):
    inputs = _inputs(n, 3, 17, torch.float16)
    scalars = [2.0, 1e-6, 1e-5]
    expected = ValueError
    if case == "x_rank2":
        inputs[0] = inputs[0].reshape(n, 51)
    elif case == "x_rank4":
        inputs[0] = inputs[0].reshape(n, 3, 1, 17)
    elif case == "hc_zero":
        inputs[0] = torch.empty((n, 0, 17), dtype=torch.float16, device=DEVICE)
    elif case == "d_zero":
        inputs[0] = torch.empty((n, 3, 0), dtype=torch.float16, device=DEVICE)
    elif case == "weight_rank1":
        inputs[1] = inputs[1].flatten()
    elif case == "weight_rows":
        inputs[1] = inputs[1][:-1]
    elif case == "weight_columns":
        inputs[1] = inputs[1][:, :-1].contiguous()
    elif case == "scale_rank2":
        inputs[2] = inputs[2][None, :]
    elif case == "scale_length":
        inputs[2] = inputs[2][:1]
    elif case == "base_rank2":
        inputs[3] = inputs[3][None, :]
    elif case == "base_length":
        inputs[3] = inputs[3][:-1]
    elif case == "weight_strided":
        inputs[1] = _strided_copy(inputs[1], 1)
    elif case in ("x_float32", "x_int32"):
        inputs[0] = inputs[0].to(torch.float32 if case == "x_float32" else torch.int32)
        expected = TypeError
    elif case.endswith("_float16"):
        index = {"weight_float16": 1, "scale_float16": 2, "base_float16": 3}[case]
        inputs[index] = inputs[index].half()
        expected = TypeError
    elif case.endswith("_not_tensor"):
        index = {
            "x_not_tensor": 0,
            "weight_not_tensor": 1,
            "scale_not_tensor": 2,
            "base_not_tensor": 3,
        }[case]
        inputs[index] = None
        expected = TypeError
    elif case == "magnitude_tensor":
        scalars[0] = torch.tensor(2.0)
        expected = TypeError
    elif case == "hc_eps_complex":
        scalars[1] = 1j
        expected = TypeError
    elif case == "norm_eps_string":
        scalars[2] = "1e-5"
        expected = TypeError
    with pytest.raises(expected):
        flaggems_vllm.triton_ihc_pre(*inputs, *scalars)


@pytest.mark.triton_ihc_pre
@pytest.mark.parametrize("index", [0, 1, 2, 3, None])
@pytest.mark.parametrize("n", [0, 3])
def test_triton_ihc_pre_rejects_cpu_or_mixed_device_inputs(index, n):
    inputs = _inputs(n, 3, 17, torch.float16)
    if index is None:
        inputs = [t.cpu() for t in inputs]
    else:
        inputs[index] = inputs[index].cpu()
    with pytest.raises(ValueError):
        flaggems_vllm.triton_ihc_pre(*inputs, 2.0, 1e-6, 1e-5)
