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

import inspect

import pytest
import torch

import flaggems_vllm

if flaggems_vllm.vendor_name == "ascend":
    from baseline.silu_mul_per_token_group_quant_fp8_colmajor import (
        silu_mul_per_token_group_quant_fp8_colmajor as baseline_silu_mul_quant,
    )
else:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        silu_mul_per_token_group_quant_fp8_colmajor as baseline_silu_mul_quant,
    )

BASELINE_HAS_CLAMP = (
    "clamp_limit" in inspect.signature(baseline_silu_mul_quant).parameters
)


@pytest.mark.silu_mul_per_token_group_quant_fp8_colmajor
@pytest.mark.skipif(
    flaggems_vllm.vendor_name
    not in ("nvidia", "metax", "thead", "mthreads", "hygon", "ascend"),
    reason="this backend has no validated fused FP8 implementation",
)
@pytest.mark.parametrize(
    "shape",
    [(128, 256), (128, 4096), (512, 4096), (1024, 1792), (8192, 4096)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("use_ue8m0", [False, True])
@pytest.mark.parametrize("clamp_limit", [None, 0.0, 2.0])
def test_silu_mul_per_token_group_quant_fp8_colmajor(
    shape, dtype, use_ue8m0, clamp_limit
):
    if clamp_limit is not None and not BASELINE_HAS_CLAMP:
        pytest.skip("selected baseline does not support clamp_limit")
    torch.manual_seed(1337)
    x = torch.randn(shape, dtype=dtype, device=flaggems_vllm.device)
    # Exercise the sign of zero for positive and negative gates explicitly.
    x[0, 0] = 1.0
    x[0, 1] = -1.0
    x[0, shape[1] // 2 : shape[1] // 2 + 2] = -0.0
    reference_kwargs = {"use_ue8m0": use_ue8m0}
    if BASELINE_HAS_CLAMP:
        reference_kwargs["clamp_limit"] = clamp_limit
    ref_output, ref_scales = baseline_silu_mul_quant(x, **reference_kwargs)
    output, scales = flaggems_vllm.silu_mul_per_token_group_quant_fp8_colmajor(
        x, use_ue8m0=use_ue8m0, clamp_limit=clamp_limit
    )
    torch.testing.assert_close(scales, ref_scales, rtol=1e-6, atol=0)
    assert torch.equal(output.view(torch.uint8), ref_output.view(torch.uint8))
