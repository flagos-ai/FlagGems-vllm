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
from flaggems_vllm.ops.fp8_fp4_mqa_logits import fp8_fp4_mqa_logits

from .accuracy_utils import calc_diff, to_reference
from .fp8_fp4_quant import (
    dequantize_mxfp4,
    per_custom_dims_cast_to_fp8,
    quantize_to_mxfp4,
)

_vendor = flaggems_vllm.vendor_name
_skip_arch = False
_skip_ref = False

if _vendor == "mthreads":
    try:
        from deep_gemm import fp8_mqa_logits as _ref_fp8_mqa_logits
    except ImportError:
        _skip_ref = "requires deep_gemm with fp8_mqa_logits support"

    def _ref_dense_mqa(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits):
        return _ref_fp8_mqa_logits(
            q[0],
            kv=kv,
            weights=weights,
            cu_seq_len_k_start=cu_seqlen_ks,
            cu_seq_len_k_end=cu_seqlen_ke,
            clean_logits=clean_logits,
        )

elif _vendor == "nvidia":
    try:
        from vllm.platforms import current_platform
        from vllm.utils.deep_gemm import fp8_fp4_mqa_logits as _ref_fp8_fp4_mqa_logits

        if not (
            torch.cuda.is_available() and current_platform.has_device_capability(90)
        ):
            _skip_arch = "requires SM90+"
    except ImportError:
        _skip_ref = "requires vLLM with DeepGEMM and FP8 quantization support"

    _ref_dense_mqa = _ref_fp8_fp4_mqa_logits

else:
    _skip_arch = f"unsupported vendor: {_vendor}"
    _skip_ref = f"unsupported vendor: {_vendor}"


def reference_fp4_mqa_logits(q_packed, q_scale, k_fp8, k_scale, weights, ks, ke):
    """Float32 reference for fp8_fp4_mqa_logits with MXFP4 Q.

    Computes: logits[m,n] = sum_h(ReLU(sum_d(q[m,h,d]*k[n,d]) * k_scale[n]) * w[m,h])
    with full dequantization of both Q (E2M1) and K (FP8) to float32.
    """
    M, H, D2 = q_packed.shape
    D = D2 * 2
    N = k_fp8.shape[0]

    q_f32 = dequantize_mxfp4(q_packed, q_scale, D)

    k_f32 = k_fp8.to(torch.float8_e4m3fn).to(torch.float32)

    logits = torch.zeros(M, N, device=q_packed.device, dtype=torch.float32)
    for m in range(M):
        for h in range(H):
            dot = torch.sum(q_f32[m, h].unsqueeze(0) * k_f32, dim=1) * k_scale
            dot = dot.clamp_min(0.0)
            logits[m] += dot * weights[m, h]

    # Fill invalid positions with -inf (vectorized, matches mate semantics)
    positions = torch.arange(N, device=q_packed.device).unsqueeze(0)
    valid = (positions >= ks.unsqueeze(1)) & (positions < ke.unsqueeze(1))
    logits.masked_fill_(~valid, float("-inf"))
    return logits


device = flaggems_vllm.device

# DeepSeek V4 production config
H = 64
D = 128

# Test shapes: (M, N) covering decode and prefill workloads
DECODE_SHAPES = [(1, 1024), (1, 2048), (1, 4096), (4, 2048), (4, 4096)]
PREFILL_SHAPES = [
    (64, 4096),
    (256, 4096),
    (1024, 4096),
    (2048, 4096),
    (1024, 8192),
]


def _build_inputs(M, N, device, use_fp4=False):
    """Build FP8 quantized inputs matching vLLM DeepGEMM conventions."""
    torch.manual_seed(42)

    q_bf16 = torch.randn(M, H, D, device=device, dtype=torch.bfloat16)
    k_bf16 = torch.randn(N, D, device=device, dtype=torch.bfloat16)
    weights = torch.randn(M, H, device=device, dtype=torch.float32).abs()

    if use_fp4:
        q_packed, q_scale = quantize_to_mxfp4(q_bf16)
    else:
        q_packed = q_bf16.to(torch.float8_e4m3fn)
        q_scale = None

    k_fp8, k_scale = per_custom_dims_cast_to_fp8(k_bf16, (0,), False)
    ks = torch.zeros(M, dtype=torch.int32, device=device)
    ke = torch.full((M,), N, dtype=torch.int32, device=device)

    return q_packed, q_scale, k_fp8, k_scale, weights, ks, ke


@pytest.mark.fp8_fp4_mqa_logits
@pytest.mark.skipif(_skip_arch, reason=_skip_arch or "")
@pytest.mark.skipif(_skip_ref, reason=_skip_ref or "")
@pytest.mark.parametrize(
    "M, N",
    DECODE_SHAPES + PREFILL_SHAPES,
    ids=[f"{m}x{n}" for m, n in DECODE_SHAPES + PREFILL_SHAPES],
)
@pytest.mark.parametrize("clean_logits", [True, False])
@pytest.mark.parametrize("use_fp4", [False, True])
def test_fp8_fp4_mqa_logits(M, N, clean_logits, use_fp4):
    q_values, q_scale, k_fp8, k_scale, weights, ks, ke = _build_inputs(
        M, N, device, use_fp4=use_fp4
    )

    if use_fp4:
        ref_out = reference_fp4_mqa_logits(
            q_values, q_scale, k_fp8, k_scale, weights, ks, ke
        )
    else:
        ref_out = _ref_dense_mqa(
            q=(q_values, None),
            kv=(k_fp8, k_scale),
            weights=weights,
            cu_seqlen_ks=ks,
            cu_seqlen_ke=ke,
            clean_logits=clean_logits,
        )
        ref_out = to_reference(ref_out)

    with flaggems_vllm.use_gems():
        res_out = fp8_fp4_mqa_logits(
            q=(q_values, q_scale),
            kv=(k_fp8, k_scale),
            weights=weights,
            cu_seqlen_ks=ks,
            cu_seqlen_ke=ke,
            clean_logits=clean_logits,
        )

    if clean_logits:
        ref_neginf_mask = ref_out == float("-inf")
        neginf_mask = res_out == float("-inf")
        assert torch.equal(
            neginf_mask, ref_neginf_mask
        ), f"-inf pattern mismatch: kernel {neginf_mask.sum()} vs ref {ref_neginf_mask.sum()}"
        ref_out = ref_out.masked_fill(ref_neginf_mask, 0)
        res_out = res_out.masked_fill(ref_neginf_mask, 0)

    diff = calc_diff(res_out.float(), ref_out.float())
    assert diff < 1e-3, f"calc_diff={diff}"
