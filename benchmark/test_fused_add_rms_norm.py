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

# NVIDIA and Hygon expose fused_add_rms_norm through vLLM custom ops.
VLLM_NATIVE_VENDORS = {
    "nvidia",
    "hygon",
}


# -----------------------------------------------------------------------------
# Vendor-specific baseline import
# -----------------------------------------------------------------------------

if VENDOR in VLLM_NATIVE_VENDORS:
    from vllm import _custom_ops as vendor_ops
elif VENDOR == "mthreads":
    from vllm_musa import _custom_ops as vendor_ops
else:
    vendor_ops = None


# -----------------------------------------------------------------------------
# Vendor capabilities
# -----------------------------------------------------------------------------


def _mthreads_shape_supported(shape):
    """Return whether vLLM-MUSA fused_add_rms_norm supports the shape."""
    if len(shape) != 2:
        return False

    hidden_size = shape[-1]
    return hidden_size > 0 and hidden_size % 8 == 0 and hidden_size <= 16384


def _get_supported_dtypes():
    """Return dtypes supported by both FlagGems and the selected baseline."""
    if VENDOR == "mthreads":
        # vLLM-MUSA fused_add_rms_norm supports FP16/BF16 only.
        return [
            dtype
            for dtype in consts.FLOAT_DTYPES
            if dtype in (torch.float16, torch.bfloat16)
        ]

    # NVIDIA/Hygon native vLLM baseline and the generic PyTorch
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
    """vLLM native fused_add_rms_norm baseline for NVIDIA and Hygon."""
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

    # vLLM-MUSA also updates input/residual in-place.
    return x


def _get_baseline_op():
    """Select the baseline implementation for the current vendor."""
    if VENDOR in VLLM_NATIVE_VENDORS:
        return _vllm_native_op

    if VENDOR == "mthreads":
        return _mthreads_vllm_op

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

    MThreads:
        vLLM-MUSA vs FlagGems-vllm

    Other vendors:
        PyTorch reference vs FlagGems-vllm
    """

    def get_latency(self, op, *args, **kwargs):
        """Give each measured implementation independent mutable buffers."""
        args = list(args)

        # fused_add_rms_norm modifies input/residual in-place.
        # Clone outside the timed kernel region so baseline and FlagGems
        # both start from the same values.
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

    bench = FusedAddRmsNormBenchmark(
        input_fn=_input_fn,
        op_name="fused_add_rms_norm",
        # GenericBenchmark names this field torch_op, but it represents
        # the selected baseline implementation here.
        torch_op=baseline_op,
        # Top-level FlagGems API selects the vendor-specific backend.
        gems_op=flaggems_vllm.fused_add_rms_norm,
        # MThreads: FP16/BF16
        # NVIDIA/Hygon/others: consts.FLOAT_DTYPES
        dtypes=_get_supported_dtypes(),
        is_inplace=True,
    )

    bench.run()
