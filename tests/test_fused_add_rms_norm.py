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
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    FUSED_SHAPES = [*utils.REDUCTION_SHAPES, (2, 8192)]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    FUSED_SHAPES = utils.REDUCTION_SHAPES


@pytest.mark.fused_add_rms_norm
@pytest.mark.parametrize("shape", FUSED_SHAPES)
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


@pytest.mark.fused_add_rms_norm
def test_fused_add_rms_norm_rejects_unsafe_execution_modes():
    device = flaggems_vllm.device
    noncontiguous_x = torch.randn((4, 2), device=device).T
    residual = torch.randn((2, 4), device=device)
    weight = torch.randn((4,), device=device)

    with pytest.raises(NotImplementedError, match="contiguous"):
        flaggems_vllm.ops.fused_add_rms_norm(noncontiguous_x, residual, (4,), weight)

    grad_x = torch.randn((2, 4), device=device, requires_grad=True)
    with pytest.raises(NotImplementedError, match="inference-only"):
        flaggems_vllm.ops.fused_add_rms_norm(grad_x, residual, (4,), weight)

    with pytest.raises(TypeError, match="same dtype"):
        flaggems_vllm.ops.fused_add_rms_norm(
            torch.randn((2, 4), device=device, dtype=torch.float16),
            torch.randn((2, 4), device=device, dtype=torch.float16),
            (4,),
            weight,
        )

    with pytest.raises(TypeError, match="only supports"):
        flaggems_vllm.ops.fused_add_rms_norm(
            torch.randn((2, 4), device=device, dtype=torch.float64),
            torch.randn((2, 4), device=device, dtype=torch.float64),
            (4,),
            weight.to(torch.float64),
        )

    aliased_x = torch.randn((2, 4), device=device)
    with pytest.raises(ValueError, match="must not overlap"):
        flaggems_vllm.ops.fused_add_rms_norm(aliased_x, aliased_x, (4,), weight)

    overlapping_storage = torch.randn((3, 4), device=device)
    partially_overlapping_x = overlapping_storage[:2]
    partially_overlapping_residual = overlapping_storage[1:]
    assert partially_overlapping_x.is_contiguous()
    assert partially_overlapping_residual.is_contiguous()
    with pytest.raises(ValueError, match="must not overlap"):
        flaggems_vllm.ops.fused_add_rms_norm(
            partially_overlapping_x,
            partially_overlapping_residual,
            (4,),
            weight,
        )

    overlapping_weight_x = torch.randn((2, 4), device=device)
    with pytest.raises(ValueError, match="must not overlap"):
        flaggems_vllm.ops.fused_add_rms_norm(
            overlapping_weight_x,
            residual,
            (4,),
            overlapping_weight_x[0],
        )
