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

import pytest
import torch

import flaggems_vllm

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

BINCOUNT_SIZES = [16, 100, 1024, 10000] if not QUICK_MODE else [100, 1024]
BINCOUNT_MAXVALS = [10, 100, 1000] if not QUICK_MODE else [100]


def _half_ulp(values, dtype):
    """ULP of non-negative fp64 values in a half-precision dtype."""
    finfo = torch.finfo(dtype)
    rounded = values.clamp(min=0.0, max=finfo.max / 2).to(dtype)
    upper = torch.nextafter(
        rounded, torch.full_like(rounded, float("inf"), dtype=dtype)
    )
    return upper.to(torch.float64) - rounded.to(torch.float64)


def _assert_bincount_weights_close(res, ref, ref_inp, ref_weights):
    """Compare weighted bincount outputs across accumulation domains.

    FlagGems' bincount follows the official generic semantics on every
    backend: weights are upcast to fp32, accumulated, and rounded to the
    weights dtype exactly once at the end. torch's own bincount accumulates
    fp16/bf16 weights in a different domain: aten (CPU and CUDA) converts
    every non-fp32 weights tensor to double before accumulating
    (SummaryOps dispatches through ``.to(kDouble)``), while the GCU native
    op (topsatenBincount) atomically adds in the half domain itself. The
    two sides disagree by cross-domain rounding noise that grows with the
    per-bin accumulation length, which the plain ``gems_assert_close``
    tolerance (atol=1e-4 + dtype rtol) does not model.

    """
    dtype = ref_weights.dtype
    assert res.dtype == dtype
    if dtype not in (torch.float16, torch.bfloat16):
        utils.gems_assert_close(res, ref, dtype)
        return

    inp_cpu = ref_inp.cpu()
    weights_cpu = ref_weights.cpu()
    counts = torch.bincount(inp_cpu).to(torch.float64)
    l1 = torch.bincount(inp_cpu, weights=weights_cpu.to(torch.float64).abs())
    tol = (counts + 1.0) * 0.5 * _half_ulp(l1, dtype) + counts * l1 * 2.0**-23

    diff = (res.cpu().to(torch.float64) - ref.cpu().to(torch.float64)).abs()
    if tol.numel() < diff.numel():
        pad = torch.zeros(diff.numel() - tol.numel(), dtype=torch.float64)
        tol = torch.cat([tol, pad])
    tol = tol[: diff.numel()]
    ok = diff <= tol
    assert bool(ok.all()), (
        f"weighted bincount mismatch beyond cross-accumulation bound: "
        f"max|res-ref|={diff.max().item():.6g}, "
        f"bound={tol.max().item():.6g}, dtype={dtype}, n={ref_inp.numel()}"
    )


@pytest.mark.bincount
@pytest.mark.parametrize("size", BINCOUNT_SIZES)
@pytest.mark.parametrize("max_val", BINCOUNT_MAXVALS)
def test_accuracy_bincount(size, max_val):
    """Test bincount without weights."""
    inp = torch.randint(
        0, max_val, (size,), dtype=torch.int64, device=flaggems_vllm.device
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.bincount(ref_inp)
    res_out = flaggems_vllm.bincount(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bincount
@pytest.mark.parametrize("size", BINCOUNT_SIZES)
@pytest.mark.parametrize("max_val", BINCOUNT_MAXVALS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_bincount_with_weights(size, max_val, dtype):
    """Test bincount with weights."""
    inp = torch.randint(
        0, max_val, (size,), dtype=torch.int64, device=flaggems_vllm.device
    )
    weights = torch.randn(size, dtype=dtype, device=flaggems_vllm.device)
    ref_inp = utils.to_reference(inp)
    ref_weights = utils.to_reference(weights)

    ref_out = torch.bincount(ref_inp, weights=ref_weights)
    res_out = flaggems_vllm.bincount(inp, weights=weights)

    if flaggems_vllm.vendor_name == "enflame":
        # enflame-only gate: this backend's fp32-accumulation
        # semantics diverge from native half-domain accumulation; use the
        # noise-model tolerance + CPU-copy compare instead of the default.
        _assert_bincount_weights_close(res_out, ref_out, ref_inp, ref_weights)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.bincount
@pytest.mark.parametrize("size", BINCOUNT_SIZES)
@pytest.mark.parametrize("max_val", BINCOUNT_MAXVALS)
@pytest.mark.parametrize("minlength", [0, 50, 2000])
def test_accuracy_bincount_with_minlength(size, max_val, minlength):
    """Test bincount with minlength parameter."""
    inp = torch.randint(
        0, max_val, (size,), dtype=torch.int64, device=flaggems_vllm.device
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.bincount(ref_inp, minlength=minlength)
    res_out = flaggems_vllm.bincount(inp, minlength=minlength)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bincount
def test_accuracy_bincount_empty():
    """Test bincount with empty input."""
    inp = torch.tensor([], dtype=torch.int64, device=flaggems_vllm.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.bincount(ref_inp)
    res_out = flaggems_vllm.bincount(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bincount
def test_accuracy_bincount_single():
    """Test bincount with single element."""
    inp = torch.tensor([5], dtype=torch.int64, device=flaggems_vllm.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.bincount(ref_inp)
    res_out = flaggems_vllm.bincount(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bincount
def test_accuracy_bincount_all_zeros():
    """Test bincount with all zeros."""
    inp = torch.zeros(100, dtype=torch.int64, device=flaggems_vllm.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.bincount(ref_inp)
    res_out = flaggems_vllm.bincount(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bincount
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_bincount_weights_edge_cases(dtype):
    """Test bincount with edge case weights."""
    inp = torch.tensor([0, 1, 2, 1, 0], dtype=torch.int64, device=flaggems_vllm.device)
    weights = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype, device=flaggems_vllm.device
    )
    ref_inp = utils.to_reference(inp)
    ref_weights = utils.to_reference(weights)

    ref_out = torch.bincount(ref_inp, weights=ref_weights)
    res_out = flaggems_vllm.bincount(inp, weights=weights)

    if flaggems_vllm.vendor_name == "enflame":
        # enflame-only gate: this backend's fp32-accumulation
        # semantics diverge from native half-domain accumulation; use the
        # noise-model tolerance + CPU-copy compare instead of the default.
        _assert_bincount_weights_close(res_out, ref_out, ref_inp, ref_weights)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)
