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


def _f32_to_fp8_e4m3fn(y):
    """Bit-exact f32 -> e4m3fn conversion (RNE); `y` must be finite and
    pre-clamped to [-448, 448].

    torch_npu cannot cast fp32 -> float8_e4m3fn on device, so this mirrors
    in torch the branchless integer sequence the Ascend Triton kernel uses
    (`_f32_to_fp8_e4m3fn` in runtime/backend/_ascend/ops/
    per_token_group_quant_fp8.py) and returns the fp8 tensor via a uint8
    bitcast.  torch uses int32 where the kernel uses uint32; every lane that
    survives the final `where` is non-negative, so the arithmetic shifts
    here agree with the kernel's logical shifts on those lanes.
    """
    b = y.view(torch.int32)
    a = b & 0x7FFFFFFF
    t = a - 0x3C000000
    t += 0x0007FFFF + ((t >> 20) & 1)
    r_norm = t >> 20
    r_sub = (a.view(torch.float32) * 512.0 + 8388608.0).view(torch.int32) - 0x4B000000
    r = torch.where(a >= 0x3C800000, r_norm, r_sub)
    return (r | ((b >> 24) & 0x80)).to(torch.uint8).view(torch.float8_e4m3fn)


def _fp8_to_float32(t):
    """Upcast fp8 for comparison; torch_npu has no on-device fp8 -> fp32
    cast either, so route the cast through the host."""
    if t.dtype == torch.float8_e4m3fn and t.device.type == "npu":
        return t.cpu().to(torch.float32)
    return t.to(torch.float32)


def native_per_token_group_quant_fp8(
    x, group_size, eps=1e-10, dtype=None, scale_ue8m0=False
):
    if dtype is None:
        dtype = flaggems_vllm.per_token_group_quant_fp8.__globals__[
            "SUPPORTED_FP8_DTYPE"
        ]

    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps).to(torch.float32)
    x_s = amax * torch.tensor(1.0 / fp8_max, dtype=torch.float32, device=x.device)
    if scale_ue8m0:
        min_val = torch.tensor(1e-10, dtype=x_s.dtype, device=x_s.device)
        x_s = torch.exp2(torch.ceil(torch.log2(torch.maximum(x_s.abs(), min_val))))
    x_q = (x_ / x_s).clamp(min=fp8_min, max=fp8_max)
    if dtype == torch.float8_e4m3fn and x_q.device.type == "npu":
        # torch_npu cannot cast fp32 -> float8_e4m3fn on device; use the
        # same bit-exact integer conversion as the Ascend Triton kernel.
        x_q = _f32_to_fp8_e4m3fn(x_q)
    else:
        x_q = x_q.to(dtype)
    x_q = x_q.reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))

    return x_q, x_s


@pytest.mark.per_token_group_quant_fp8
@pytest.mark.parametrize("seed", utils.FP8_QUANT_SHAPES["SEEDS"])
@pytest.mark.parametrize("group_size", utils.FP8_QUANT_SHAPES["GROUP_SIZE"])
@pytest.mark.parametrize("dtype", utils.FP8_QUANT_SHAPES["DTYPES"])
@pytest.mark.parametrize("d", utils.FP8_QUANT_SHAPES["D"])
@pytest.mark.parametrize("num_tokens", utils.FP8_QUANT_SHAPES["NUM_TOKENS"])
@pytest.mark.parametrize("scale_ue8m0", [True, False])
def test_per_token_group_quant_fp8(num_tokens, d, dtype, group_size, seed, scale_ue8m0):
    torch.manual_seed(seed)

    x = torch.rand(num_tokens, d, dtype=dtype, device=flaggems_vllm.device)
    ref_x = utils.to_reference(x)

    ref_out, ref_scale = native_per_token_group_quant_fp8(
        ref_x, group_size, scale_ue8m0=scale_ue8m0
    )
    with flaggems_vllm.use_gems():
        out, scale = flaggems_vllm.per_token_group_quant_fp8(
            x, group_size, scale_ue8m0=scale_ue8m0
        )

    utils.gems_assert_close(scale, ref_scale, dtype=torch.float32)

    out_fp32 = _fp8_to_float32(utils.to_cpu(out, ref_out))
    ref_out_fp32 = _fp8_to_float32(ref_out)

    assert torch.allclose(out_fp32, ref_out_fp32, rtol=0.15)


def _require_generic_group_quant_fp8(*, require_fp8=False):
    import importlib

    op = importlib.import_module("flaggems_vllm.ops.per_token_group_quant_fp8")
    if flaggems_vllm.per_token_group_quant_fp8 is not op.per_token_group_quant_fp8:
        pytest.skip("backend uses its own per-token group quantization implementation")
    if require_fp8 and op.SUPPORTED_FP8_DTYPE != torch.float8_e4m3fn:
        pytest.skip("generic FP8 e4m3fn conversion is unavailable")
    return op


@pytest.mark.per_token_group_quant_fp8
def test_per_token_group_quant_fp8_generic_contract_guard(monkeypatch):
    import importlib

    op = importlib.import_module("flaggems_vllm.ops.per_token_group_quant_fp8")
    monkeypatch.setattr(
        flaggems_vllm, "per_token_group_quant_fp8", op.per_token_group_quant_fp8
    )
    monkeypatch.setattr(op, "SUPPORTED_FP8_DTYPE", torch.float8_e4m3fn)
    assert _require_generic_group_quant_fp8(require_fp8=True) is op

    monkeypatch.setattr(op, "SUPPORTED_FP8_DTYPE", torch.float32)
    assert _require_generic_group_quant_fp8() is op
    with pytest.raises(pytest.skip.Exception, match="conversion is unavailable"):
        _require_generic_group_quant_fp8(require_fp8=True)

    monkeypatch.setattr(
        flaggems_vllm, "per_token_group_quant_fp8", lambda *a, **k: None
    )
    for require_fp8 in (False, True):
        with pytest.raises(pytest.skip.Exception, match="backend uses its own"):
            _require_generic_group_quant_fp8(require_fp8=require_fp8)


@pytest.mark.per_token_group_quant_fp8
@pytest.mark.parametrize("has_reorder", [False, True])
def test_per_token_group_quant_fp8_reorder_schedule(monkeypatch, has_reorder):
    import importlib

    op = importlib.import_module("flaggems_vllm.ops.per_token_group_quant_fp8")
    monkeypatch.setattr(op, "_HAS_LOAD_REORDER", has_reorder)
    for total, group, ue in [
        (7, 512, False),
        (112, 256, True),
        (664, 64, False),
        (32768, 256, True),
        (55296, 512, False),
        (55296, 512, True),
    ]:
        groups, steps, warps = op._contiguous_group_config(total, group, ue)
        assert groups > 0 and warps in (1, 4)
        assert steps in ((1, 2, 4) if has_reorder else (1,))
        if has_reorder and not (group == 256 and ue and total >= 128):
            assert steps > 1


@pytest.mark.per_token_group_quant_fp8
def test_per_token_group_quant_fp8_without_reorder(monkeypatch):
    import triton.language as tl

    op = _require_generic_group_quant_fp8(require_fp8=True)
    monkeypatch.setattr(op, "_HAS_LOAD_REORDER", False)
    monkeypatch.setattr(op, "tle", tl)
    torch.manual_seed(0)
    x = torch.rand((7, 512), dtype=torch.bfloat16, device=flaggems_vllm.device)
    expected_q, expected_s = native_per_token_group_quant_fp8(x, 512)
    q, s = flaggems_vllm.per_token_group_quant_fp8(x, 512)
    utils.gems_assert_close(s, expected_s, dtype=torch.float32)
    assert torch.allclose(_fp8_to_float32(q), _fp8_to_float32(expected_q), rtol=0.15)


@pytest.mark.per_token_group_quant_fp8
@pytest.mark.parametrize(
    "shape,group_size",
    [((7, 512), 512), ((83, 512), 64), ((257, 512), 256), ((129, 3584), 512)],
)
@pytest.mark.parametrize("scale_ue8m0", [False, True])
def test_per_token_group_quant_fp8_reorder_special(shape, group_size, scale_ue8m0):
    import triton

    op = _require_generic_group_quant_fp8(require_fp8=True)
    if not op._HAS_LOAD_REORDER:
        pytest.skip("compiler does not expose load reorder")
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16, device=flaggems_vllm.device)
    flat = x.view(-1)
    flat[0] = float("nan")
    flat[group_size : 2 * group_size] = float("inf")
    flat[2 * group_size : 3 * group_size] = -float("inf")
    flat[3 * group_size : 4 * group_size] = 0.0
    expected_q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    expected_s = torch.empty(
        (shape[0], shape[1] // group_size), device=x.device, dtype=torch.float32
    )
    # This scalar kernel is unchanged from upstream, including dynamic clamp
    # bounds. It is an independent device oracle for special values and tails.
    op._per_token_group_quant_fp8[(x.numel() // group_size,)](
        x,
        expected_q,
        expected_s,
        group_size,
        shape[1],
        x.stride(0),
        1e-10,
        -448.0,
        448.0,
        scale_ue8m0,
        BLOCK=triton.next_power_of_2(group_size),
        num_warps=min(max(triton.next_power_of_2(group_size) // 256, 1), 8),
        num_stages=1,
    )
    q, s = flaggems_vllm.per_token_group_quant_fp8(
        x, group_size, scale_ue8m0=scale_ue8m0
    )
    assert torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))
    torch.testing.assert_close(s, expected_s, rtol=0, atol=0, equal_nan=True)


@pytest.mark.per_token_group_quant_fp8
@pytest.mark.parametrize("shape", [(0, 512), (7, 0)])
def test_per_token_group_quant_fp8_empty(shape):
    _require_generic_group_quant_fp8()
    x = torch.empty(shape, dtype=torch.bfloat16, device=flaggems_vllm.device)
    q, s = flaggems_vllm.per_token_group_quant_fp8(x, 512)
    assert q.shape == x.shape and q.device == x.device and q.numel() == 0
    assert s.shape == (shape[0], shape[1] // 512) and s.dtype == torch.float32
    assert s.device == x.device and s.numel() == 0
