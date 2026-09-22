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

"""MUSA SwiGLU MoE with plain output-major INT4 or E4M3FN weights.

The weight layout matches the generic operator, not Marlin's int32 repack.
INT4 packs consecutive reduction elements low nibble first with bias eight.
Both formats share routing, group-scale decoding and A16 matrix multiplication.
Weights and scales are read on every call, including graph replay.
"""

from typing import Any, Callable, Optional

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.moe_align_block_size import (
    moe_align_block_size_singleton,
    moe_align_block_size_stage1,
    moe_align_block_size_stage2,
    moe_align_block_size_stage2_vec,
    moe_align_block_size_stage3,
    moe_align_block_size_stage4,
)
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._mthreads.fused.moe_sum import moe_sum as _moe_sum
from flaggems_vllm.utils import libentry, libtuner


@triton.jit
def _zero_workspace(X, N: tl.constexpr, B: tl.constexpr):
    offsets = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(X + offsets, 0, offsets < N)


def _align_routes(ids, bm, experts):
    """Reuse the four-stage aligner with Triton-initialized scratch storage."""
    routes = ids.numel()
    if routes <= 16:
        return moe_align_block_size_singleton(ids, bm)
    capacity = min(routes + experts * (bm - 1), routes * bm)
    sorted_ids = torch.empty((capacity,), device=ids.device, dtype=torch.int32)
    block_experts = torch.empty(
        (triton.cdiv(capacity, bm),), device=ids.device, dtype=torch.int32
    )
    padded = torch.empty((1,), device=ids.device, dtype=torch.int32)
    workspace = torch.empty(
        ((experts + 1) * (experts + 1),), device=ids.device, dtype=torch.int32
    )
    cumsum = workspace[: experts + 1]
    counts = workspace[experts + 1 :]
    _zero_workspace[(triton.cdiv(workspace.numel(), 256),)](
        workspace, workspace.numel(), 256
    )
    tokens_per_thread = triton.next_power_of_2(triton.cdiv(routes, experts))
    moe_align_block_size_stage1[(experts,)](
        ids,
        counts,
        experts,
        routes,
        tokens_per_thread,
        sorted_ids,
        block_experts,
        capacity,
        block_experts.numel(),
        triton.next_power_of_2(triton.cdiv(capacity, experts)),
        triton.next_power_of_2(triton.cdiv(block_experts.numel(), experts)),
    )
    if experts == triton.next_power_of_2(experts):
        moe_align_block_size_stage2_vec[(experts,)](counts, experts)
    else:
        moe_align_block_size_stage2[(experts,)](counts, experts)
    moe_align_block_size_stage3[(1,)](
        padded, counts, cumsum, experts, triton.next_power_of_2(experts), bm
    )
    moe_align_block_size_stage4[(experts,)](
        ids,
        sorted_ids,
        block_experts,
        counts,
        cumsum,
        experts,
        bm,
        routes,
        tokens_per_thread,
    )
    return sorted_ids, block_experts, padded


@triton.jit
def _decode_weight(
    W,
    S,
    expert,
    n,
    k,
    N: tl.constexpr,
    K: tl.constexpr,
    WE: tl.constexpr,
    WN: tl.constexpr,
    WK: tl.constexpr,
    SE: tl.constexpr,
    SN: tl.constexpr,
    SG: tl.constexpr,
    Q: tl.constexpr,
    GROUP: tl.constexpr,
    DTYPE: tl.constexpr,
):
    expert = expert.to(tl.int64)
    n = n.to(tl.int64)
    k = k.to(tl.int64)
    packed_k = k // 2 if Q == 0 else k
    code = tl.load(W + expert * WE + n * WN + packed_k * WK, (n < N) & (k < K), other=0)
    if Q == 0:
        value = (((code.to(tl.int32) >> (4 * (k.to(tl.int32) % 2))) & 15) - 8).to(
            tl.float32
        )
    else:
        value = code.to(tl.uint8).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    group = tl.full(k.shape, 0, tl.int64) if GROUP == -1 else k // GROUP
    scale = tl.load(
        S + expert * SE + n * SN + group * SG, (n < N) & (k < K), other=0
    ).to(tl.float32)
    return (value * scale).to(DTYPE)


def _prune_gemm_configs(configs, named_args, **kwargs):
    bm = {**named_args, **kwargs}["BM"]
    bn = 32 if bm == 64 else 64
    return [config for config in configs if config.kwargs["BN"] == bn]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mthreads_marlin_moe_gemm"),
    key=["N", "K", "R", "Q", "FIRST", "BM", "GROUP"],
    prune_configs_by={"early_config_prune": _prune_gemm_configs},
)
@triton.jit
def _gemm(
    A,
    W,
    S,
    Weights,
    Routes,
    Experts,
    Padded,
    C,
    N: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    TOPK: tl.constexpr,
    WE: tl.constexpr,
    WN: tl.constexpr,
    WK: tl.constexpr,
    SE: tl.constexpr,
    SN: tl.constexpr,
    SG: tl.constexpr,
    Q: tl.constexpr,
    GROUP: tl.constexpr,
    FIRST: tl.constexpr,
    ROUTER_INPUT: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    block = tl.program_id(0)
    if block * BM >= tl.load(Padded):
        return
    route = tl.load(Routes + block * BM + tl.arange(0, BM)).to(tl.int64)
    expert = tl.load(Experts + block).to(tl.int64)
    n = (tl.program_id(1) * BN + tl.arange(0, BN)).to(tl.int64)
    rk = tl.arange(0, BK)
    dtype: tl.constexpr = A.dtype.element_ty
    acc = tl.zeros((BM, BN), tl.float32)
    if FIRST:
        up = tl.zeros((BM, BN), tl.float32)
    for base in range(tl.cdiv(K, BK)):
        k = base * BK + rk
        row = route // TOPK if FIRST else route
        a = tl.load(
            A + row[:, None] * K + k[None, :],
            (route[:, None] < R) & (k[None, :] < K),
            other=0,
        )
        w = _decode_weight(
            W,
            S,
            expert,
            n[:, None],
            k[None, :],
            2 * N if FIRST else N,
            K,
            WE,
            WN,
            WK,
            SE,
            SN,
            SG,
            Q,
            GROUP,
            dtype,
        )
        w = tl.where(n[:, None] < N, w, 0)
        acc = tl.dot(a, tl.trans(w), acc)
        if FIRST:
            wu = _decode_weight(
                W,
                S,
                expert,
                n[:, None] + N,
                k[None, :],
                2 * N,
                K,
                WE,
                WN,
                WK,
                SE,
                SN,
                SG,
                Q,
                GROUP,
                dtype,
            )
            wu = tl.where(n[:, None] < N, wu, 0)
            up = tl.dot(a, tl.trans(wu), up)
    if (FIRST and ROUTER_INPUT) or (not FIRST and not ROUTER_INPUT):
        rw = tl.load(Weights + route, route < R, other=0).to(tl.float32)
        acc *= rw[:, None]
        if FIRST:
            up *= rw[:, None]
    if FIRST:
        gate = acc.to(dtype).to(tl.float32)
        up = up.to(dtype).to(tl.float32)
        acc = gate * tl.sigmoid(gate) * up
    tl.store(
        C + route[:, None] * N + n[None, :],
        acc,
        (route[:, None] < R) & (n[None, :] < N),
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mthreads_marlin_moe_gemv"),
    key=["N", "K", "R", "Q", "FIRST", "GROUP"],
)
@triton.jit
def _gemv(
    A,
    W,
    S,
    Weights,
    Ids,
    C,
    N: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    TOPK: tl.constexpr,
    WE: tl.constexpr,
    WN: tl.constexpr,
    WK: tl.constexpr,
    SE: tl.constexpr,
    SN: tl.constexpr,
    SG: tl.constexpr,
    Q: tl.constexpr,
    GROUP: tl.constexpr,
    FIRST: tl.constexpr,
    ROUTER_INPUT: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    route = tl.program_id(0).to(tl.int64)
    expert = tl.load(Ids + route).to(tl.int64)
    n = (tl.program_id(1) * BN + tl.arange(0, BN)).to(tl.int64)
    rk = tl.arange(0, BK)
    dtype: tl.constexpr = A.dtype.element_ty
    accum = tl.zeros((BN, BK), tl.float32)
    if FIRST:
        up_accum = tl.zeros((BN, BK), tl.float32)
    row = route // TOPK if FIRST else route
    for base in range(tl.cdiv(K, BK)):
        k = base * BK + rk
        a = tl.load(A + row * K + k, k < K, other=0).to(tl.float32)
        w = _decode_weight(
            W,
            S,
            expert,
            n[:, None],
            k[None, :],
            2 * N if FIRST else N,
            K,
            WE,
            WN,
            WK,
            SE,
            SN,
            SG,
            Q,
            GROUP,
            dtype,
        ).to(tl.float32)
        accum += w * a[None, :]
        if FIRST:
            wu = _decode_weight(
                W,
                S,
                expert,
                n[:, None] + N,
                k[None, :],
                2 * N,
                K,
                WE,
                WN,
                WK,
                SE,
                SN,
                SG,
                Q,
                GROUP,
                dtype,
            ).to(tl.float32)
            up_accum += wu * a[None, :]
    acc = tl.sum(accum, 1)
    if FIRST:
        up = tl.sum(up_accum, 1)
    if (FIRST and ROUTER_INPUT) or (not FIRST and not ROUTER_INPUT):
        weight = tl.load(Weights + route).to(tl.float32)
        acc *= weight
        if FIRST:
            up *= weight
    if FIRST:
        gate = acc.to(dtype).to(tl.float32)
        up = up.to(dtype).to(tl.float32)
        acc = gate * tl.sigmoid(gate) * up
    tl.store(C + route * N + n, acc, n < N)


def _validate(a, w1, w2, s1, s2, tw, ids, output, inplace, group, quant):
    if a.ndim != 2 or w1.ndim != 3 or w2.ndim != 3 or ids.ndim != 2:
        raise ValueError("Expected activations/routing rank 2 and weights rank 3")
    m, k = a.shape
    e, n2, _ = w1.shape
    n = n2 // 2
    if e <= 0 or n2 % 2 or min(k, n) <= 0 or k % 32 or n % 32:
        raise ValueError("Positive K/N multiples of 32 and paired gate/up required")
    if not isinstance(group, int) or (group <= 0 and not (quant == 2 and group == -1)):
        raise NotImplementedError("Unsupported quantization group size")
    if group != -1 and (k % group or n % group):
        raise ValueError("Input dimensions must be divisible by group_size")
    pack = 2 if quant == 0 else 1
    if w1.shape != (e, 2 * n, k // pack) or w2.shape != (e, k, n // pack):
        raise ValueError("Weight shapes do not match activations")
    g1, g2 = (1, 1) if group == -1 else (k // group, n // group)
    if s1.shape != (e, 2 * n, g1) or s2.shape != (e, k, g2):
        raise ValueError("Scale shapes do not match quantization groups")
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("MUSA Marlin MoE requires FP16 or BF16 activations")
    weight_dtypes = (torch.uint8,) if quant == 0 else (torch.uint8, torch.float8_e4m3fn)
    if w1.dtype not in weight_dtypes or w2.dtype not in weight_dtypes:
        raise NotImplementedError("Expected packed UINT4B8 or E4M3FN weights")
    scale_dtypes = (
        (a.dtype,) if quant == 0 else (torch.float16, torch.bfloat16, torch.float32)
    )
    if s1.dtype not in scale_dtypes or s2.dtype not in scale_dtypes:
        raise NotImplementedError("Unsupported scale dtype")
    if tw.shape != ids.shape or ids.shape[0] != m or not 1 <= ids.shape[1] <= e:
        raise ValueError("Routing must have shape [M, topk], 1 <= topk <= E")
    if ids.dtype not in (torch.int32, torch.int64) or tw.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise NotImplementedError("Unsupported routing dtype")
    tensors = (a, w1, w2, s1, s2, tw, ids)
    if a.device.type != "musa" or any(t.device != a.device for t in tensors):
        raise ValueError("All tensors must reside on the same MUSA device")
    if any(t.requires_grad for t in tensors):
        raise NotImplementedError("MUSA Marlin MoE is inference-only")
    if any(not t.is_contiguous() for t in (a, tw, ids)):
        raise NotImplementedError("Activations and routing must be contiguous")
    if inplace and output is not None:
        raise ValueError("Cannot pass both inplace=True and output")
    if output is not None:
        if (
            output.shape != a.shape
            or output.dtype != a.dtype
            or output.device != a.device
        ):
            raise ValueError("Output shape/dtype/device must match hidden_states")
        if not output.is_contiguous() or output.requires_grad:
            raise ValueError("Output must be contiguous and inference-only")
    target = a if inplace else output
    if target is not None and target.numel():
        for tensor in (w1, w2, s1, s2, tw, ids):
            if (
                tensor.numel()
                and target.untyped_storage().data_ptr()
                == tensor.untyped_storage().data_ptr()
            ):
                raise ValueError("Output must not alias weights, scales or routing")
    return m, k, n, e, ids.shape[1]


def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: Optional[torch.Tensor],
    bias2: Optional[torch.Tensor],
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type_id: int,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    activation: Optional[Any] = None,
    activation_func: Optional[Callable] = None,
    moe_sum: Optional[Callable] = None,
    expert_map: Optional[torch.Tensor] = None,
    input_global_scale1: Optional[torch.Tensor] = None,
    input_global_scale2: Optional[torch.Tensor] = None,
    global_scale1: Optional[torch.Tensor] = None,
    global_scale2: Optional[torch.Tensor] = None,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    intermediate_cache13: Optional[torch.Tensor] = None,
    intermediate_cache2: Optional[torch.Tensor] = None,
    is_k_full: bool = True,
    output: Optional[torch.Tensor] = None,
    input_dtype: Optional[torch.dtype] = None,
    inplace: bool = False,
    clamp_limit: Optional[float] = None,
    group_size: int = 128,
) -> torch.Tensor:
    """Evaluate native INT4/FP8 SwiGLU MoE without quantizing activations."""
    if quant_type_id not in (0, 2):
        raise NotImplementedError("MUSA Marlin MoE supports INT4 and FP8")
    activation_str = getattr(
        activation, "value", getattr(activation, "name", activation)
    )
    if activation_str is not None and str(activation_str).lower() != "silu":
        raise NotImplementedError("MUSA Marlin MoE supports only SiLU")
    if any(x is not None for x in (g_idx1, g_idx2, sort_indices1, sort_indices2)):
        raise NotImplementedError("MUSA Marlin MoE does not support act_order")
    if input_dtype is not None:
        raise NotImplementedError("MUSA Marlin MoE does not support FP8 input")
    unsupported = (
        bias1,
        bias2,
        activation_func,
        moe_sum,
        expert_map,
        input_global_scale1,
        input_global_scale2,
        global_scale1,
        global_scale2,
        w1_zeros,
        w2_zeros,
        workspace,
        intermediate_cache13,
        intermediate_cache2,
        clamp_limit,
    )
    if any(x is not None for x in unsupported) or not is_k_full:
        raise NotImplementedError("Unsupported MUSA Marlin MoE option")
    if w1.ndim != 3 or global_num_experts not in (-1, w1.shape[0]):
        raise NotImplementedError("MUSA Marlin MoE requires local expert weights")
    m, k, n, e, topk = _validate(
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        output,
        inplace,
        group_size,
        quant_type_id,
    )
    out = hidden_states if inplace else output
    if out is None:
        out = torch.empty_like(hidden_states)
    if m == 0:
        return out
    r = m * topk
    # Reuse decoded weights across more rows when routing is dense.
    bm = 16 if r < e * 16 else 64
    with torch_device_fn.device(hidden_states.device):
        # Direct vector reductions avoid padded matrix work for sparse routing.
        # FP16 INT4 vector conversion is not supported by this MUSA compiler.
        direct = hidden_states.dtype == torch.bfloat16 and (
            m <= 4 or (quant_type_id == 2 and r <= e)
        )
        if not direct:
            routes, experts, padded = _align_routes(topk_ids, bm, e)
        intermediate = torch.empty((r, n), device=out.device, dtype=out.dtype)
        result = torch.empty((m, topk, k), device=out.device, dtype=out.dtype)
        for a, w, s, c, first, nk, kk in (
            (hidden_states, w1, w1_scale, intermediate, True, n, k),
            (intermediate, w2, w2_scale, result, False, k, n),
        ):
            if quant_type_id == 2:
                w = w.view(torch.uint8)
            if direct:
                _gemv[lambda meta: (r, triton.cdiv(nk, meta["BN"]))](
                    a,
                    w,
                    s,
                    topk_weights,
                    topk_ids,
                    c,
                    nk,
                    kk,
                    r,
                    topk,
                    *w.stride(),
                    *s.stride(),
                    quant_type_id,
                    group_size,
                    first,
                    apply_router_weight_on_input,
                )
            else:
                _gemm[
                    lambda meta: (
                        triton.cdiv(routes.numel(), bm),
                        triton.cdiv(nk, meta["BN"]),
                    )
                ](
                    a,
                    w,
                    s,
                    topk_weights,
                    routes,
                    experts,
                    padded,
                    c,
                    nk,
                    kk,
                    r,
                    topk,
                    *w.stride(),
                    *s.stride(),
                    quant_type_id,
                    group_size,
                    first,
                    apply_router_weight_on_input,
                    bm,
                )
        _moe_sum(result, out)
    return out


def fused_marlin_moe_w4a16_int4(
    hidden_states,
    w1,
    w2,
    w1_scale,
    w2_scale,
    topk_weights,
    topk_ids,
    *,
    activation="silu",
    group_size=128,
    apply_router_weight_on_input=False,
    inplace=False,
    swap_ab=True,
):
    return fused_marlin_moe(
        hidden_states,
        w1,
        w2,
        None,
        None,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        0,
        activation=activation,
        group_size=group_size,
        apply_router_weight_on_input=apply_router_weight_on_input,
        inplace=inplace,
    )


def fused_marlin_moe_w8a16_fp8(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    *,
    w1_scale,
    w2_scale,
    group_size=128,
    inplace=False,
    output=None,
):
    return fused_marlin_moe(
        hidden_states,
        w1,
        w2,
        None,
        None,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        2,
        group_size=group_size,
        inplace=inplace,
        output=output,
    )


__all__ = [
    "fused_marlin_moe",
    "fused_marlin_moe_w4a16_int4",
    "fused_marlin_moe_w8a16_fp8",
]
