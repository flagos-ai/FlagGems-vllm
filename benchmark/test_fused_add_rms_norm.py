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


# -----------------------------------------------------------------------------
# Vendor-specific baseline import
#
# Only import the native vLLM implementation for the current vendor.
# This prevents, for example, NVIDIA environments from requiring vllm_musa.
# -----------------------------------------------------------------------------

if VENDOR == "nvidia":
    # Importing vLLM custom ops loads/registers the _C CUDA extension.
    from vllm import _custom_ops as vendor_ops

elif VENDOR == "mthreads":
    from vllm_musa import _custom_ops as vendor_ops

else:
    vendor_ops = None


# -----------------------------------------------------------------------------
# Vendor capabilities
# -----------------------------------------------------------------------------


def _mthreads_shape_supported(shape):
    """Whether vLLM-MUSA fused_add_rms_norm supports this shape.

    vLLM-MUSA requirements:
      - input/residual must be 2-D
      - weight must be 1-D
      - input.shape == residual.shape
      - weight.shape[0] == input.shape[1]
      - hidden_size % 8 == 0
      - hidden_size <= 16384

    The input generator already guarantees:
      - input.shape == residual.shape
      - weight.shape == (shape[-1],)
    Therefore only the remaining shape restrictions need to be checked here.
    """
    if len(shape) != 2:
        return False

    hidden_size = shape[-1]

    return hidden_size > 0 and hidden_size % 8 == 0 and hidden_size <= 16384


def _get_supported_dtypes():
    """Return the dtype intersection supported by FlagGems and baseline."""
    if VENDOR == "mthreads":
        # vLLM-MUSA fused_add_rms_norm supports FP16/BF16 only.
        return [
            dtype
            for dtype in consts.FLOAT_DTYPES
            if dtype in (torch.float16, torch.bfloat16)
        ]

    # NVIDIA vLLM fused_add_rms_norm supports the benchmark FLOAT_DTYPES.
    # Other vendors use the generic PyTorch reference below.
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
    """Generic PyTorch reference used when no vendor-native baseline is set."""
    del layer_shape

    x = x + residual
    variance = x.pow(2).mean(-1, keepdim=True)
    hidden_states = x * torch.rsqrt(variance + eps)

    return weight * hidden_states


def _nvidia_vllm_op(x, residual, layer_shape, weight, eps):
    """vLLM C/CUDA fused_add_rms_norm baseline."""
    del layer_shape

    vendor_ops.fused_add_rms_norm(
        x,
        residual,
        weight,
        eps,
    )

    # vLLM updates x/residual in-place and returns None.
    # Return x only to keep a normal benchmark callable interface.
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
    if VENDOR == "nvidia":
        return _nvidia_vllm_op

    if VENDOR == "mthreads":
        return _mthreads_vllm_op

    # Preserve the original FlagGems-vllm benchmark behavior on vendors
    # without a dedicated vLLM-native baseline here.
    return _torch_reference_op


# -----------------------------------------------------------------------------
# Benchmark
# -----------------------------------------------------------------------------


class FusedAddRmsNormBenchmark(base.GenericBenchmarkExcluse1D):
    """Benchmark FlagGems-vllm fused_add_rms_norm.

    NVIDIA:
        vLLM C/CUDA vs FlagGems-vllm

    MThreads:
        vLLM-MUSA vs FlagGems-vllm

    Other vendors:
        PyTorch reference vs FlagGems-vllm
    """

    def get_latency(self, op, *args, **kwargs):
        """Give each measured implementation independent mutable buffers.

        fused_add_rms_norm modifies input and residual in-place.

        Benchmark.run() invokes get_latency() independently for the baseline
        and FlagGems implementations. Cloning input/residual here therefore
        guarantees that:

          1. baseline and FlagGems start from the same original values;
          2. one implementation cannot mutate the other's input;
          3. clone overhead is outside the timed region.
        """
        args = list(args)

        # args:
        #   0: input
        #   1: residual
        #   2: layer_shape
        #   3: weight
        #   4: eps
        args[0] = args[0].clone()
        args[1] = args[1].clone()

        return super().get_latency(op, *args, **kwargs)

    def init_user_config(self):
        """Apply baseline-specific capability restrictions."""
        super().init_user_config()

        if VENDOR == "mthreads":
            # GenericBenchmarkExcluse1D normally contains both 2-D and 3-D
            # shapes. vLLM-MUSA accepts only a subset of the 2-D shapes.
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

    # Vendor-specific availability checks.
    if VENDOR == "nvidia":
        assert hasattr(
            torch.ops._C,
            "fused_add_rms_norm",
        ), "vLLM _C::fused_add_rms_norm is not available"

    elif VENDOR == "mthreads":
        assert hasattr(
            vendor_ops,
            "musa_fused_add_rms_norm",
        ), "vLLM-MUSA musa_fused_add_rms_norm is not available"

    bench = FusedAddRmsNormBenchmark(
        input_fn=_input_fn,
        op_name="fused_add_rms_norm",
        # GenericBenchmark calls this field torch_op, but here it means
        # "baseline implementation".
        torch_op=baseline_op,
        # Use the top-level API so runtime backend specialization/override
        # can select the implementation for the current AI accelerator.
        gems_op=flaggems_vllm.fused_add_rms_norm,
        # MThreads: FP16/BF16
        # NVIDIA/others: consts.FLOAT_DTYPES
        dtypes=_get_supported_dtypes(),
        is_inplace=True,
    )

    bench.run()
