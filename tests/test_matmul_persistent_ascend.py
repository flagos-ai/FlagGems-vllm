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

"""Test the package-level Ascend matmul specialization."""

import pytest
import torch

import flaggems_vllm as gems
from baseline import matmul_persistent as workload
from tests import accuracy_utils as utils
from tests import conftest as cfg

ASCEND_MODULE = "flaggems_vllm.runtime.backend._ascend.ops.matmul_persistent"


def build_ascend_inputs(binding, dtype, device):
    generator = torch.Generator(device="cpu").manual_seed(42)
    a, b = (
        torch.rand(binding[key], dtype=dtype, generator=generator).to(device)
        for key in ("a_shape", "b_shape")
    )
    return a, b


def assert_ascend_output(result, expected, a):
    assert result.shape == expected.shape
    assert result.dtype == a.dtype
    assert result.device == a.device
    tolerance = 0.1 if a.dtype == torch.bfloat16 else 0.01
    utils.gems_assert_close(result.cpu(), expected, a.dtype, atol=tolerance)


@pytest.fixture(scope="module")
def npu():
    if gems.vendor_name != "ascend" or not torch.npu.is_available():
        pytest.skip("Ascend NPU required")
    return torch.device("npu", torch.npu.current_device())


@pytest.mark.matmul_persistent
@pytest.mark.parametrize("dtype", workload.DTYPES)
@pytest.mark.parametrize("binding", workload.grid())
def test_ascend_matmul_persistent(binding, dtype, npu, caplog):
    if cfg.QUICK_MODE and binding not in workload.grid()[:2]:
        pytest.skip("Outside the quick shape set")
    entry = gems.matmul_persistent
    assert entry.__module__ == ASCEND_MODULE
    a, b = build_ascend_inputs(binding, dtype, npu)
    a_cpu, b_cpu = a.cpu(), b.cpu()
    with caplog.at_level("DEBUG", logger=utils.gems_log_logger(entry)):
        result = entry(a, b)
    assert f"{utils.gems_log_prefix(entry)} MATMUL_PERSISTENT" in caplog.text
    assert_ascend_output(result, a_cpu.double() @ b_cpu.double(), a)
    utils.gems_assert_equal(a.cpu(), a_cpu)
    utils.gems_assert_equal(b.cpu(), b_cpu)


@pytest.mark.matmul_persistent
@pytest.mark.parametrize("dtype", workload.DTYPES)
@pytest.mark.parametrize("layout", ["offset", "strided", "row_padding"])
def test_ascend_matmul_tail_layout_and_out(dtype, layout, npu):
    def view_with_layout(shape):
        batches, rows, columns = shape
        step = 2 if layout == "strided" else 1
        row_stride = columns * step + (5 if layout == "row_padding" else 0)
        strides = (rows * row_stride, row_stride, step)
        offset = 1
        length = offset + 1 + sum((size - 1) * s for size, s in zip(shape, strides))
        storage = torch.full((length,), float("nan"), device=npu, dtype=dtype)
        view = storage.as_strided(shape, strides, storage_offset=offset)
        signed = torch.randn(shape, dtype=dtype, generator=generator)
        view.copy_(signed)
        return view

    generator = torch.Generator(device="cpu").manual_seed(42)
    a = view_with_layout((2, 65, 129))
    b = view_with_layout((2, 129, 33))
    expected = a.cpu().double() @ b.cpu().double()
    storage = torch.full((expected.numel() + 32,), -11, device=npu, dtype=dtype)
    out = storage[16:-16].view(expected.shape)
    actual = gems.matmul_persistent(a, b, out=out)
    assert actual is out
    assert_ascend_output(actual, expected, a)
    for guard in (storage[:16], storage[-16:]):
        utils.gems_assert_equal(guard, torch.full_like(guard, -11))
    single = gems.matmul_persistent(a[0:1], b[0:1])
    utils.gems_assert_equal(single, actual[0:1])


@pytest.mark.matmul_persistent
def test_ascend_matmul_contract(npu):
    entry = gems.matmul_persistent
    a = torch.empty((2, 3), device=npu, dtype=torch.float16)
    b = torch.empty((3, 3), device=npu, dtype=torch.float16)
    with pytest.raises(ValueError, match="input storage"):
        entry(a, b, out=a)

    strided = torch.empty((2, 9), device=npu, dtype=a.dtype)[:, ::3]
    with pytest.raises(NotImplementedError, match="packing strides"):
        entry(strided, b)

    for left, right, shape in (
        ((0, 3), (3, 2), (0, 2)),
        ((2, 0), (0, 3), (2, 3)),
    ):
        x = torch.empty(left, device=npu, dtype=a.dtype)
        y = torch.empty(right, device=npu, dtype=a.dtype)
        result = entry(x, y)
        utils.gems_assert_equal(result, torch.zeros(shape, device=npu, dtype=a.dtype))
