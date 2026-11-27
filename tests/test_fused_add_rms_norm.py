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

import warnings

import pytest
import torch

import flaggems_vllm

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES


def _detect_vllm_fused_add_rms_norm():
    """Return vllm._custom_ops if its fused_add_rms_norm is really callable.

    1. vllm._custom_ops must import and expose the Python wrapper.
    2. vllm._C (compiled extension) must be importable.
    3. On CUDA devices the dispatcher must have a CUDA kernel registered.
    No kernel is executed and no device is touched at import time.
    """
    try:
        from vllm import _custom_ops as ops
    except Exception:
        return None

    if not callable(getattr(ops, "fused_add_rms_norm", None)):
        return None

    try:
        import vllm._C  # noqa: F401
    except Exception as e:
        warnings.warn(
            f"vllm._C is unavailable ({type(e).__name__}: {e}); "
            "using the torch reference instead of vLLM native."
        )
        return None

    try:
        dev_type = torch.device(flaggems_vllm.device).type
    except Exception:
        dev_type = "cuda"

    if dev_type == "cuda":
        try:
            has_kernel = torch._C._dispatch_has_kernel_for_dispatch_key(
                "_C::fused_add_rms_norm", "CUDA"
            )
        except Exception:
            has_kernel = True  # internal API changed; trust steps 1-2
        if not has_kernel:
            warnings.warn(
                "_C::fused_add_rms_norm has no CUDA kernel registered; "
                "using the torch reference instead of vLLM native."
            )
            return None

    return ops


_vllm_ops = _detect_vllm_fused_add_rms_norm()
_HAS_VLLM_FUSED_ADD_RMS_NORM = _vllm_ops is not None


def _vllm_fused_add_rms_norm_ref(x, residual, weight, eps):
    """In-place vLLM baseline; returns normalized output for comparison."""
    _vllm_ops.fused_add_rms_norm(x, residual, weight, eps)
    return x, residual


@pytest.mark.fused_add_rms_norm
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_fused_add_rms_norm(shape, dtype):
    N = shape[1]
    layer_shape = [
        N,
    ]
    inp = torch.randn(shape[:2], dtype=dtype, device=flaggems_vllm.device)
    residual = torch.randn(shape[:2], dtype=dtype, device=flaggems_vllm.device)
    weight = torch.randn(layer_shape, dtype=dtype, device=flaggems_vllm.device)
    eps = 1e-5

    # Prefer vLLM's native fused_add_rms_norm as the reference when available;
    # fall back to the torch composition otherwise.
    if _HAS_VLLM_FUSED_ADD_RMS_NORM:
        ref_inp = inp.clone()
        ref_res = residual.clone()
        ref_out, ref_new_residual = _vllm_fused_add_rms_norm_ref(
            ref_inp, ref_res, weight, eps
        )
        # The vLLM baseline runs on-device (CUDA/MUSA/etc.) and leaves its
        # outputs there, unlike the torch-reference branch below, whose
        # inputs are already normalized to the reference device/dtype via
        # utils.to_reference() *before* the computation. Without this step,
        # gems_assert_close -> accuracy_utils.to_cpu()'s device assertion
        # fails on any vendor where TO_CPU is set (e.g. MThreads/MUSA).
        ref_out = utils.to_reference(ref_out, True)
        ref_new_residual = utils.to_reference(ref_new_residual, True)
    else:
        ref_inp = utils.to_reference(inp, True)
        ref_residual = utils.to_reference(residual, True)
        ref_weight = utils.to_reference(weight, True)

        def _torch_fused_add_rms_norm(x, residual, weight, eps):
            x = x + residual
            variance = x.pow(2).mean(-1, keepdim=True)
            hidden_states = x * torch.rsqrt(variance + eps)
            return weight * hidden_states, x

        ref_out, ref_new_residual = _torch_fused_add_rms_norm(
            ref_inp,
            ref_residual,
            weight=ref_weight,
            eps=eps,
        )

    res_out, res_new_residual = flaggems_vllm.ops.fused_add_rms_norm(
        inp, residual, list(layer_shape), weight=weight, eps=eps
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(res_new_residual, ref_new_residual, dtype)
