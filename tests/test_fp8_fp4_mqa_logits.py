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

try:
    from vllm.platforms import current_platform
    from vllm.third_party.deep_gemm.utils import per_custom_dims_cast_to_fp8
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits as vllm_fp8_fp4_mqa_logits

    VLLM_AVAILABLE = True
    SM90_AVAILABLE = current_platform.has_device_capability(90)
except ImportError:
    VLLM_AVAILABLE = False
    SM90_AVAILABLE = False

from vllm.third_party.deep_gemm.utils.math import (
    cast_back_from_fp4,
    per_token_cast_to_fp4,
)

import flaggems_vllm
from flaggems_vllm.ops.fp8_fp4_mqa_logits import fp8_fp4_mqa_logits

from .accuracy_utils import gems_assert_close, to_reference


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

    # Fill invalid positions with -inf
    for m in range(M):
        logits[m, : ks[m]] = float("-inf")
        logits[m, ke[m] :] = float("-inf")
    return logits


MXFP4_BLOCK_SIZE = 32


def quantize_to_mxfp4(x):
    """Quantize a bf16/fp32 tensor to MXFP4 (E2M1) packed format.

    Uses vLLM's ``per_token_cast_to_fp4`` internally, then repacks the
    per-block scales into a single int32 per token-head (little-endian)
    so the Triton kernels can extract byte_i via ``(val >> (8*i)) & 0xFF``.

    Returns:
        packed: int8 tensor with shape [..., D//2] (two E2M1 nibbles per byte)
        scales: int32 tensor with shape [..., 1] (packed ue8m0 per-block scales)
    """
    orig_shape = x.shape
    D = orig_shape[-1]
    flat = x.float().reshape(-1, D)  # [flat_rows, D]

    packed_2d, sf_2d = per_token_cast_to_fp4(
        flat, use_ue8m0=False, gran_k=MXFP4_BLOCK_SIZE
    )
    # packed_2d: [flat_rows, D//2] int8, sf_2d: [flat_rows, D//32] float32

    # Convert float32 scales to ue8m0 bytes: val = round(log2(sf)) + 127
    n_blocks = D // MXFP4_BLOCK_SIZE
    ue8m0 = (
        sf_2d.abs().clamp(min=2**-126).log2().round().clamp(-127.0, 127.0) + 127.0
    ).to(
        torch.uint8
    )  # [flat_rows, n_blocks]
    packed_scales = torch.zeros(flat.shape[0], dtype=torch.int32, device=flat.device)
    for i in range(n_blocks):
        packed_scales |= ue8m0[:, i].to(torch.int32) << (8 * i)
    packed_scales = packed_scales.reshape(*orig_shape[:-1], 1)

    packed = packed_2d.to(torch.int8).reshape(*orig_shape[:-1], D // 2)
    return packed, packed_scales


def dequantize_mxfp4(packed, scales, head_dim):
    """Dequantize MXFP4 packed E2M1 values to float32.

    Uses vLLM's ``cast_back_from_fp4`` internally, after unpacking the
    per-token-head int32 scale back to per-block float32.

    Args:
        packed: int8 [..., D//2] (two E2M1 nibbles per byte)
        scales: int32 [..., 1] (packed ue8m0: D//32 bytes little-endian in one int32)
        head_dim: D
    Returns:
        x_f32: float32 [..., D]
    """
    orig_batch = packed.shape[:-1]
    D = head_dim
    n_blocks = D // MXFP4_BLOCK_SIZE

    # Unpack int32 scales -> ue8m0 bytes -> float32 scales
    scales_flat = scales.squeeze(-1).reshape(-1)  # [flat_rows]
    block_id = torch.arange(n_blocks, device=packed.device)  # [n_blocks]
    ue8m0_bytes = ((scales_flat[..., None] >> (8 * block_id[None, :])) & 0xFF).to(
        torch.float32
    )  # [flat_rows, n_blocks]
    # Convert ue8m0 byte codes to actual scale values: 2^(byte - 127)
    sf_2d = torch.exp2(ue8m0_bytes - 127.0)  # [flat_rows, n_blocks]

    # Reshape packed to 2D [flat_rows, D//2] int8 for cast_back_from_fp4
    packed_2d = packed.reshape(-1, D // 2).to(torch.int8)

    x_2d = cast_back_from_fp4(packed_2d, sf_2d, gran_k=MXFP4_BLOCK_SIZE)
    return x_2d.reshape(*orig_batch, D)


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
@pytest.mark.skipif(
    not (torch.cuda.is_available() and SM90_AVAILABLE),
    reason="requires CUDA with Hopper architecture (SM90+)",
)
@pytest.mark.skipif(
    not VLLM_AVAILABLE,
    reason="requires vLLM with DeepGEMM and FP8 quantization support",
)
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
        ref_out = vllm_fp8_fp4_mqa_logits(
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

    # FP4 has lower precision than FP8, so use a wider tolerance
    atol = 0.15 if use_fp4 else 5e-2
    gems_assert_close(
        res_out, ref_out, res_out.dtype, equal_nan=True, atol=atol, reduce_dim=1
    )
