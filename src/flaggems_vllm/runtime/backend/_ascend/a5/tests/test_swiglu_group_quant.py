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

"""Vendor-aligned accuracy tests for the Ascend A5 SwiGLU group-quant op.

Run on an idle 950DT, for example:

    ASCEND_RT_VISIBLE_DEVICES=7 PYTHONPATH=src pytest -q \
        src/flaggems_vllm/runtime/backend/_ascend/a5/tests/test_swiglu_group_quant.py
"""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
pytest.importorskip("cann_ops_nn")

from flaggems_vllm.runtime.backend._ascend.a5.ops.swiglu_group_quant import (  # noqa: E402
    SwigluGroupQuantPlan,
    scale_shape,
    swiglu_group_quant,
)

_DST_CODE = {torch.float8_e4m3fn: 292, torch.float8_e5m2: 291}
_TIE_RATE = 2e-4
_SCALE_ULP = 8


def _vendor(x, weight=None, group_index=None, **kwargs):
    dst = kwargs.pop("dst_type", torch.float8_e4m3fn)
    return torch.ops.cann_ops_nn.swiglu_group_quant(
        x,
        weight,
        group_index,
        None,
        dst_type=_DST_CODE[dst],
        dst_type_max=15.0,
        **kwargs,
    )


def _bytes(t):
    return t.cpu().view(torch.uint8)


def _check_bytes(name, actual, expected, fp8=False):
    actual, expected = _bytes(actual), _bytes(expected)
    assert actual.shape == expected.shape, name
    assert actual.dtype == expected.dtype, name
    mismatch = actual != expected
    count = int(mismatch.sum())
    if fp8 and count <= max(1, int(_TIE_RATE * actual.numel())):
        # The separate SwiGLU rounding can select an adjacent FP8 value at an
        # exact tie. Bound the rate globally, not with a loose value tolerance.
        return
    assert count == 0, f"{name}: {count}/{actual.numel()} bytes differ"


def _check_scale(actual, expected, plain_fp32=False):
    if not plain_fp32:
        _check_bytes("scale", actual, expected)
        return
    delta = (actual.view(torch.int32).long() - expected.view(torch.int32).long()).abs()
    # aclnnExp/Div and Triton can differ by a few ULP on this unrounded path.
    # Every other scale is compared byte-exactly.
    assert delta.numel() == 0 or int(delta.max()) <= _SCALE_ULP


def _check_case(
    T,
    D,
    *,
    quant_mode=1,
    round_scale=None,
    dtype=torch.bfloat16,
    dst_type=torch.float8_e4m3fn,
    weight=False,
    clamp_limit=-1.0,
    output_origin=False,
    zeros=False,
):
    if round_scale is None:
        round_scale = quant_mode == 1
    torch.manual_seed(0)
    H = D // 2
    if zeros:
        x = torch.zeros(T, D, dtype=dtype)
        x[:, 0::32], x[:, H + 0 :: 32] = 0.0, -1.0  # exact -0.0
        x[:, 1::32], x[:, H + 1 :: 32] = -0.0, 1.0  # exact -0.0
        x[:, 2::32], x[:, H + 2 :: 32] = -0.0, -1.0  # exact +0.0
        x[:, 3::32], x[:, H + 3 :: 32] = 1e-20, -1.0  # negative underflow
        x[:, 4::32], x[:, H + 4 :: 32] = 10.0, 0.4
        x = x.npu()
    else:
        x = torch.randn(T, D, device="npu", dtype=dtype) * 3
    w = torch.randn(T, device="npu", dtype=torch.float32) if weight else None
    kwargs = dict(
        dst_type=dst_type,
        quant_mode=quant_mode,
        block_size=0,
        round_scale=round_scale,
        clamp_limit=clamp_limit,
        output_origin=output_origin,
    )
    actual = swiglu_group_quant(x, w, **kwargs)
    expected = _vendor(x, w, **kwargs)
    _check_bytes("y", actual[0], expected[0], fp8=True)
    _check_scale(actual[1], expected[1], quant_mode == 0 and not round_scale)
    if output_origin:
        _check_bytes("y_origin", actual[2], expected[2])
    else:
        assert actual[2] is None
    assert actual[0].shape == (T, H)
    assert actual[1].shape == scale_shape((T, D), quant_mode)
    assert actual[1].dtype == expected[1].dtype


@pytest.mark.parametrize("T", [1, 7, 64, 256, 1023])
def test_mx_basic_and_row_tail(T):
    _check_case(T, 4096)


@pytest.mark.parametrize("D", [256, 512, 1536, 2560, 3584, 8192, 14336])
def test_hidden_sizes_and_overlapping_columns(D):
    _check_case(64, D)


@pytest.mark.parametrize("mode,rounded", [(1, True), (0, True), (0, False)])
@pytest.mark.parametrize("dst_type", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_quant_modes_and_dtypes(mode, rounded, dst_type, dtype):
    _check_case(
        64, 2048, quant_mode=mode, round_scale=rounded, dst_type=dst_type, dtype=dtype
    )


@pytest.mark.parametrize("mode", [0, 1])
def test_optional_inputs(mode):
    _check_case(
        100, 2560, quant_mode=mode, weight=True, clamp_limit=2.5, output_origin=True
    )


@pytest.mark.parametrize("counts", [[70], [17, 18, 17, 18]])
@pytest.mark.parametrize("mode", [0, 1])
def test_group_index_tail_is_untouched(counts, mode):
    T, D = 100, 2560
    x = torch.randn(T, D, device="npu", dtype=torch.bfloat16)
    gi = torch.tensor(counts, device="npu", dtype=torch.int64)
    y0 = torch.empty((T, D // 2), dtype=torch.float8_e4m3fn, device="npu")
    y0.view(torch.uint8).random_(0, 256)
    scale0 = torch.full(scale_shape((T, D), mode), 3, dtype=torch.uint8, device="npu")
    if mode == 1:
        scale0 = scale0.view(torch.float8_e8m0fnu)
    else:
        scale0 = scale0.float()
    kwargs = dict(quant_mode=mode, round_scale=True, dst_type=torch.float8_e4m3fn)
    actual = swiglu_group_quant(
        x, group_index=gi, out=y0.clone(), scale=scale0.clone(), **kwargs
    )
    expected = _vendor(x, group_index=gi, **kwargs)
    real = sum(counts)
    _check_bytes("y", actual[0][:real], expected[0][:real])
    _check_bytes("scale", actual[1][:real], expected[1][:real])
    _check_bytes("untouched y tail", actual[0][real:], y0[real:])
    _check_bytes("untouched scale tail", actual[1][real:], scale0[real:])


@pytest.mark.parametrize("mode,rounded", [(1, True), (0, True), (0, False)])
def test_signed_zero(mode, rounded):
    _check_case(64, 4096, quant_mode=mode, round_scale=rounded, zeros=True)


def test_multiple_specializations_and_reused_plan():
    # A cached compile must not bind another mode's binary.
    x = torch.randn(64, 1024, device="npu", dtype=torch.float16)
    for mode, rounded in [(1, True), (0, True), (0, False)]:
        plan = SwigluGroupQuantPlan(x, quant_mode=mode, round_scale=rounded)
        kwargs = dict(quant_mode=mode, round_scale=rounded)
        for _ in range(3):
            actual = plan(x)
            expected = _vendor(x, **kwargs)
            _check_bytes("y", actual[0], expected[0], fp8=True)
            _check_scale(actual[1], expected[1], mode == 0 and not rounded)


def test_higher_rank_scale_shape():
    x = torch.randn(2, 3, 4, 1024, device="npu", dtype=torch.bfloat16)
    # Unlike this op's MX default, the vendor defaults to block mode 0.
    actual = swiglu_group_quant(x)
    expected = _vendor(x, quant_mode=1, round_scale=True)
    _check_bytes("y", actual[0], expected[0], fp8=True)
    _check_bytes("scale", actual[1], expected[1])
    assert actual[1].shape == (2, 3, 4, 8, 2)
    assert actual[1].dtype == expected[1].dtype


def test_mx_uint8_scale_buffer_is_preserved():
    x = torch.randn(4, 1024, device="npu", dtype=torch.bfloat16)
    scale = torch.empty(4, 8, 2, dtype=torch.uint8, device="npu")
    actual = swiglu_group_quant(x, scale=scale)
    expected = _vendor(x, quant_mode=1, round_scale=True)
    assert actual[1].dtype == torch.uint8
    _check_bytes("y", actual[0], expected[0], fp8=True)
    _check_bytes("scale", actual[1], expected[1])


def test_invalid_input_raises():
    x = torch.randn(4, 512, device="npu", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="quant_mode 2"):
        swiglu_group_quant(x, quant_mode=2)
    with pytest.raises(ValueError, match="multiple of 256"):
        swiglu_group_quant(torch.randn(4, 258, device="npu", dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="weight has"):
        swiglu_group_quant(x, weight=torch.empty(3, dtype=torch.float32, device="npu"))
    with pytest.raises(ValueError, match="plan is for"):
        fresh = torch.randn(5, 512, device="npu", dtype=torch.bfloat16)
        SwigluGroupQuantPlan(x)(fresh)
