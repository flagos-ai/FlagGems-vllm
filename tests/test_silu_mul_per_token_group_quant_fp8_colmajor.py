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
from vllm.platforms import current_platform

import flaggems_vllm


def native_silu_mul_per_token_group_quant_fp8_colmajor(
    x, use_ue8m0, clamp_limit, eps=1e-10
):
    gate, up = x.chunk(2, dim=-1)
    if clamp_limit is not None:
        gate = gate.float().clamp(max=clamp_limit).to(x.dtype)
        up = up.float().clamp(-clamp_limit, clamp_limit).to(x.dtype)
    # Preserve vLLM's activation and multiplication rounding at input precision.
    gate_f32 = gate.float()
    activated = (gate_f32 / (1 + torch.exp(-gate_f32))).to(x.dtype)
    values = (activated * up).float().reshape(x.shape[0], x.shape[1] // 256, 128)
    fp8_dtype = current_platform.fp8_dtype()
    fp8_max = 224.0 if current_platform.is_fp8_fnuz() else torch.finfo(fp8_dtype).max
    scales = values.abs().amax(-1).clamp(min=eps) * (1.0 / fp8_max)
    if use_ue8m0:
        scales = torch.exp2(torch.ceil(torch.log2(scales)))
    output = (values / scales[..., None]).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return output.reshape(x.shape[0], x.shape[1] // 2), scales


@pytest.mark.silu_mul_per_token_group_quant_fp8_colmajor
@pytest.mark.skipif(
    flaggems_vllm.vendor_name not in ("nvidia", "metax", "thead", "mthreads", "hygon"),
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
    torch.manual_seed(1337)
    x = torch.randn(shape, dtype=dtype, device=flaggems_vllm.device)
    ref_output, ref_scales = native_silu_mul_per_token_group_quant_fp8_colmajor(
        x.cpu(), use_ue8m0, clamp_limit
    )
    output, scales = flaggems_vllm.silu_mul_per_token_group_quant_fp8_colmajor(
        x, use_ue8m0=use_ue8m0, clamp_limit=clamp_limit
    )
    torch.testing.assert_close(scales.cpu(), ref_scales, rtol=1e-5, atol=1e-8)
    # CPU/GPU exp can straddle FP8 rounding boundaries; allow one FP8 step.
    torch.testing.assert_close(
        output.cpu().float(), ref_output.float(), rtol=0.125, atol=2**-9
    )
