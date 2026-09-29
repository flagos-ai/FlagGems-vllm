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

from . import base, consts

VENDOR = flaggems_vllm.vendor_name

# NVIDIA, Hygon, and T-Head expose fused_add_rms_norm through
# vLLM custom ops.
VLLM_NATIVE_VENDORS = {
    "nvidia",
    "hygon",
    "thead",
}


# -----------------------------------------------------------------------------
# Vendor-specific baseline import
# -----------------------------------------------------------------------------

vendor_ops = None
iluvatar_fused_add_rms_norm_perf = None

if VENDOR in VLLM_NATIVE_VENDORS:
    from vllm import _custom_ops as vendor_ops

elif VENDOR == "mthreads":
    from vllm_musa import _custom_ops as vendor_ops

elif VENDOR == "ascend":
    import torch_npu
    from vllm_ascend.utils import enable_custom_op

elif VENDOR == "iluvatar":
    # Import vLLM first so its platform/plugin initialization completes
    # before importing vllm_iluvatar custom kernels. Importing the
    # vllm_iluvatar submodule first causes a circular import through
    # vllm_iluvatar.util.logger -> vllm -> vllm_iluvatar.platform.
    import vllm  # noqa: F401

    # Use the Iluvatar native perf backend directly instead of the unified
    # dispatcher. This guarantees that the benchmark baseline is the native
    # vLLM-Iluvatar kernel and cannot silently fall back to the torch backend.
    from vllm_iluvatar.custom_kernels.fused_add_rms_norm.perf_impl import (
        fused_add_rms_norm_perf as iluvatar_fused_add_rms_norm_perf,
    )


# -----------------------------------------------------------------------------
# Vendor capabilities
# -----------------------------------------------------------------------------


def _mthreads_shape_supported(shape):
    """Return whether vLLM-MUSA fused_add_rms_norm supports the shape."""
    if len(shape) != 2:
        return False

    hidden_size = shape[-1]
    return hidden_size > 0 and hidden_size % 8 == 0 and hidden_size <= 16384


def _iluvatar_shape_supported(shape):
    """Return whether vLLM-Iluvatar perf fused_add_rms_norm supports the shape."""
    hidden_size = shape[-1]

    # The native perf kernel requires an even hidden size.
    if hidden_size <= 0 or hidden_size % 2 != 0:
        return False

    # The current vLLM-Iluvatar native perf kernel explicitly rejects
    # hidden_size=65536 ("unsupported hidden_size: 65536").
    #
    # Keep the benchmark comparison restricted to shapes supported by both
    # vLLM-Iluvatar and FlagGems-vllm.
    if hidden_size == 65536:
        return False

    return True


def _get_supported_dtypes():
    """Return dtypes supported by both FlagGems and the selected baseline."""
    if VENDOR in ("mthreads", "iluvatar"):
        # vLLM-MUSA and vLLM-Iluvatar native fused_add_rms_norm
        # support FP16/BF16 only.
        return [
            dtype
            for dtype in consts.FLOAT_DTYPES
            if dtype in (torch.float16, torch.bfloat16)
        ]

    # Ascend 910B / A2 AddRmsNorm supports FP16, BF16 and FP32.
    #
    # NVIDIA/Hygon/T-Head native vLLM baseline and the generic PyTorch
    # reference use the normal benchmark floating-point dtype set.
    return consts.FLOAT_DTYPES


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------


def _input_fn(shape, dtype, device):
    inp = torch.randn(
        shape,
        dtype=dtype,
        device=device,
    )

    residual = torch.randn(
        shape,
        dtype=dtype,
        device=device,
    )

    layer_shape = (shape[-1],)
    weight = torch.randn(
        layer_shape,
        dtype=dtype,
        device=device,
    )

    yield inp, residual, layer_shape, weight, 1e-5


# -----------------------------------------------------------------------------
# Baselines
# -----------------------------------------------------------------------------


def _torch_reference_op(x, residual, layer_shape, weight, eps):
    """Generic PyTorch reference for vendors without a native baseline."""
    del layer_shape

    x = x + residual
    variance = x.pow(2).mean(-1, keepdim=True)
    hidden_states = x * torch.rsqrt(variance + eps)

    return weight * hidden_states


def _vllm_native_op(x, residual, layer_shape, weight, eps):
    """vLLM native fused_add_rms_norm baseline.

    Used by NVIDIA, Hygon, and T-Head.
    """
    del layer_shape

    vendor_ops.fused_add_rms_norm(
        x,
        residual,
        weight,
        eps,
    )

    # vLLM fused_add_rms_norm updates x/residual in-place.
    return x


def _mthreads_vllm_op(x, residual, layer_shape, weight, eps):
    """vLLM-MUSA fused_add_rms_norm baseline."""
    del layer_shape

    vendor_ops.musa_fused_add_rms_norm(
        x,
        residual,
        weight,
        eps,
        block_x=0,
    )

    # vLLM-MUSA updates input/residual in-place.
    return x


def _ascend_vllm_op(x, residual, layer_shape, weight, eps):
    """vLLM-Ascend fused add RMSNorm baseline.

    This mirrors the residual path used by
    vllm_ascend.ops.layernorm.AscendRMSNorm.forward_oot().

    vLLM-Ascend 0.23.0 uses:
      1. torch.ops._C_ascend.npu_add_rms_norm_bias
         when the vLLM-Ascend custom-op library is available;

      2. torch_npu.npu_add_rms_norm
         as the fallback path.

    Both implementations return the normalized output and the updated
    residual instead of modifying the original input buffers in-place.
    """
    del layer_shape

    if enable_custom_op():
        out, _, residual_out = torch.ops._C_ascend.npu_add_rms_norm_bias(
            x,
            residual,
            weight,
            None,
            eps,
        )
    else:
        out, _, residual_out = torch_npu.npu_add_rms_norm(
            x,
            residual,
            weight,
            eps,
        )

    # Return the normalized output. The benchmark only needs the callable
    # to execute the same fused operator used by vLLM-Ascend.
    del residual_out
    return out


def _iluvatar_vllm_op(x, residual, layer_shape, weight, eps):
    """vLLM-Iluvatar native perf fused_add_rms_norm baseline."""
    del layer_shape

    assert iluvatar_fused_add_rms_norm_perf is not None

    iluvatar_fused_add_rms_norm_perf(
        x,
        residual,
        weight,
        eps,
    )

    # vLLM-Iluvatar native perf fused_add_rms_norm updates input/residual
    # in-place, matching the semantics expected by this benchmark.
    return x


def _get_baseline_op():
    """Select the baseline implementation for the current vendor."""
    if VENDOR in VLLM_NATIVE_VENDORS:
        return _vllm_native_op

    if VENDOR == "mthreads":
        return _mthreads_vllm_op

    if VENDOR == "ascend":
        return _ascend_vllm_op

    if VENDOR == "iluvatar":
        return _iluvatar_vllm_op

    return _torch_reference_op


# -----------------------------------------------------------------------------
# Benchmark
# -----------------------------------------------------------------------------


class FusedAddRmsNormBenchmark(base.GenericBenchmarkExcluse1D):
    """Benchmark FlagGems-vllm fused_add_rms_norm.

    NVIDIA:
        vLLM native vs FlagGems-vllm

    Hygon:
        vLLM native vs FlagGems-vllm

    T-Head:
        vLLM native PPU kernel vs FlagGems-vllm

    MThreads:
        vLLM-MUSA vs FlagGems-vllm

    Ascend:
        vLLM-Ascend vs FlagGems-vllm

    Iluvatar:
        vLLM-Iluvatar native perf kernel vs FlagGems-vllm

    Other vendors:
        PyTorch reference vs FlagGems-vllm
    """

    def get_latency(self, op, *args, **kwargs):
        """Give each measured implementation independent input buffers."""
        args = list(args)

        # FlagGems fused_add_rms_norm and the native vLLM baselines modify
        # input/residual in-place.
        #
        # Clone outside the timed region so the baseline and FlagGems
        # implementations always start from independent buffers and the
        # clone overhead is not included in kernel latency.
        args[0] = args[0].clone()
        args[1] = args[1].clone()

        return super().get_latency(op, *args, **kwargs)

    def init_user_config(self):
        """Apply restrictions imposed by the selected baseline."""
        super().init_user_config()

        if VENDOR == "mthreads":
            self.shapes = [
                shape for shape in self.shapes if _mthreads_shape_supported(shape)
            ]

            if not self.shapes:
                pytest.skip(
                    "No benchmark shapes are supported by "
                    "vLLM-MUSA fused_add_rms_norm."
                )

        elif VENDOR == "iluvatar":
            self.shapes = [
                shape for shape in self.shapes if _iluvatar_shape_supported(shape)
            ]

            if not self.shapes:
                pytest.skip(
                    "No benchmark shapes are supported by "
                    "vLLM-Iluvatar perf fused_add_rms_norm."
                )


# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------


@pytest.mark.fused_add_rms_norm
@pytest.mark.skipif(
    flaggems_vllm.vendor_name == "tsingmicro",
    reason="Issue #4131: not working",
)
def test_fused_add_rms_norm():
    baseline_op = _get_baseline_op()

    if VENDOR in VLLM_NATIVE_VENDORS:
        assert hasattr(
            torch.ops._C,
            "fused_add_rms_norm",
        ), "vLLM native fused_add_rms_norm is not available"

    elif VENDOR == "mthreads":
        assert hasattr(
            vendor_ops,
            "musa_fused_add_rms_norm",
        ), "vLLM-MUSA musa_fused_add_rms_norm is not available"

    elif VENDOR == "ascend":
        # vLLM-Ascend uses either its own _C_ascend custom op or the
        # torch_npu AddRmsNorm implementation.
        if enable_custom_op():
            assert hasattr(
                torch.ops._C_ascend,
                "npu_add_rms_norm_bias",
            ), (
                "vLLM-Ascend custom npu_add_rms_norm_bias " "is not available"
            )
        else:
            assert hasattr(
                torch_npu,
                "npu_add_rms_norm",
            ), "torch_npu.npu_add_rms_norm is not available"

    elif VENDOR == "iluvatar":
        assert callable(
            iluvatar_fused_add_rms_norm_perf
        ), "vLLM-Iluvatar native perf fused_add_rms_norm is not available"

    bench = FusedAddRmsNormBenchmark(
        input_fn=_input_fn,
        op_name="fused_add_rms_norm",
        # GenericBenchmark calls this field torch_op, but here it
        # represents the vendor-native baseline implementation.
        torch_op=baseline_op,
        # Top-level FlagGems API selects the current vendor backend.
        gems_op=flaggems_vllm.fused_add_rms_norm,
        dtypes=_get_supported_dtypes(),
        is_inplace=True,
    )

    bench.run()
