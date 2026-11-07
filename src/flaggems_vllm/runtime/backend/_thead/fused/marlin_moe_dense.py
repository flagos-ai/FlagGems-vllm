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

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.moe_sum import moe_sum
from flaggems_vllm.ops.silu_and_mul import silu_and_mul_out
from flaggems_vllm.runtime.backend._thead.fused.marlin_moe_w8a16 import (
    decode_e4m3,
    decode_fp8_scaled_bf16,
    decode_int8_scaled_bf16,
)


@triton.jit
def decode_bank(
    Q,
    S,
    D,
    Safe,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    QE: tl.constexpr,
    QN: tl.constexpr,
    QK: tl.constexpr,
    SE: tl.constexpr,
    SN: tl.constexpr,
    SG: tl.constexpr,
    FP8: tl.constexpr,
    FAST: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    if FP8 and not FAST:
        if tl.load(Safe + E):
            return
    expert = tl.program_id(2).to(tl.int64)
    if FP8 and not FAST:
        if tl.load(Safe + expert):
            return
    ns = (tl.program_id(0) * BN + tl.arange(0, BN)).to(tl.int64)
    k_base = (tl.program_id(1) * BK).to(tl.int64)
    ks = (k_base + tl.arange(0, BK)).to(tl.int64)
    mask = (ns[:, None] < N) & (ks[None, :] < K)
    q = tl.load(
        Q + expert * QE + ns[:, None] * QN + ks[None, :] * QK,
        mask,
        other=0 if FP8 else 128,
    )
    scale = tl.load(S + expert * SE + ns * SN + (k_base // 128) * SG, ns < N, other=0)[
        :, None
    ]
    if FP8:
        if FAST:
            value = decode_fp8_scaled_bf16(q, scale)
        else:
            value = decode_e4m3(q.to(tl.uint32), scale, tl.bfloat16, False)
    else:
        value = decode_int8_scaled_bf16(q, scale)
    tl.store(D + expert * N * K + ns[:, None] + ks[None, :] * N, value, mask)


def run_dense_w8a16(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    scale_safe1: torch.Tensor | None,
    scale_safe2: torch.Tensor | None,
    is_fp8: bool,
    apply_router_weight_on_input: bool,
) -> torch.Tensor:
    from flaggems_vllm.runtime.backend._thead.ops.fused_moe import (
        get_default_config,
        invoke_fused_moe_triton_kernel,
    )

    m, k = hidden_states.shape
    e, n2, _ = w1.shape
    n, topk = n2 // 2, topk_weights.shape[1]
    scratch = torch.empty(
        (max(w1.numel(), w2.numel()),),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    cache13 = torch.empty(
        (m * topk * max(n2, k),), device=hidden_states.device, dtype=hidden_states.dtype
    )
    c1 = cache13[: m * topk * n2].view(m, topk, n2)
    c3 = cache13[: m * topk * k].view(m, topk, k)
    c2 = torch.empty(
        (m * topk, n), device=hidden_states.device, dtype=hidden_states.dtype
    )
    b1 = scratch[: w1.numel()].view(e, k, n2).transpose(1, 2)
    b2 = scratch[: w2.numel()].view(e, n, k).transpose(1, 2)
    for index, (quantized_weight, scales, decoded_weight, scale_safe) in enumerate(
        ((w1, w1_scale, b1, scale_safe1), (w2, w2_scale, b2, scale_safe2))
    ):
        output_size, reduction = quantized_weight.shape[1:]
        for fast in (True, False) if is_fp8 else (True,):
            decode_bank[(triton.cdiv(output_size, 64), triton.cdiv(reduction, 128), e)](
                quantized_weight.view(torch.uint8),
                scales,
                scratch,
                scale_safe,
                output_size,
                reduction,
                e,
                *quantized_weight.stride(),
                *scales.stride(),
                is_fp8,
                fast,
                64,
                128,
                num_warps=4,
                num_stages=1,
            )
        first = index == 0
        config = get_default_config(
            m,
            e,
            output_size,
            reduction,
            topk,
            "bf16",
            gemm_stage="gemm1" if first else "gemm2",
        )
        invoke_fused_moe_triton_kernel(
            hidden_states if first else c2,
            decoded_weight,
            c1 if first else c3,
            None,
            None,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input if first else not apply_router_weight_on_input,
            topk if first else 1,
            config,
            tl.bfloat16,
        )
        if first:
            flat = c1.view(m * topk, n2)
            silu_and_mul_out(flat[:, :n], flat[:, n:], c2)
    moe_sum(c3, output)
    return output
