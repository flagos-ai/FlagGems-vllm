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

import os

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn

# Flat-streaming kernel tuning knobs (gcu400 calibrated; see module docstring
# at the top of the flat kernel for the dispatch rationale). P_cap=48 matches
# one program per SP (96/192 measured slightly worse, gcu400 precedent).
_FLAT_CHUNK = 4096
_FLAT_TOPK_CHUNK = 4096
_FLAT_MAX_PROGRAMS = 48
_FLAT_NUM_WARPS = 1
_FLAT_NUM_STAGES = 3


def _gcu_real_int64_storage():
    # torch_gcu narrows device int64 tensors to 4-byte physical slots unless
    # the env switch is set to exactly "1" or "true" (calibrated in
    # compiler_issues/triton_i64_narrow_load.md); any other value keeps the
    # narrowed layout.
    return os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64") in ("1", "true")


def _int32_slot_view(tensor):
    """Zero-copy int32 view of a GCU int64 tensor's 4-byte physical slots."""
    words = torch.empty(0, dtype=torch.int32, device=tensor.device)
    return words.set_(
        tensor.untyped_storage(),
        int(tensor.storage_offset()),
        tuple(int(x) for x in tensor.shape),
        tuple(int(x) for x in tensor.stride()),
    )


@triton.jit
def _stage_deepseek_v4_mega_moe_inputs_kernel(
    hidden_states,
    x_fp8,
    x_sf,
    topk_ids,
    topk_weights,
    topk_idx_out,
    topk_weights_out,
    hidden_stride_m: tl.constexpr,
    hidden_stride_k: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_k: tl.constexpr,
    x_sf_stride_m: tl.constexpr,
    x_sf_stride_k: tl.constexpr,
    topk_ids_stride_m: tl.constexpr,
    topk_ids_stride_k: tl.constexpr,
    topk_weights_stride_m: tl.constexpr,
    topk_weights_stride_k: tl.constexpr,
    topk_idx_stride_m: tl.constexpr,
    topk_idx_stride_k: tl.constexpr,
    topk_weights_out_stride_m: tl.constexpr,
    topk_weights_out_stride_k: tl.constexpr,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    IDX_TY: tl.constexpr,
) -> None:
    token_id = tl.program_id(0)
    k_block_id = tl.program_id(1)

    k_offsets = k_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < hidden_size
    hidden = tl.load(
        hidden_states + token_id * hidden_stride_m + k_offsets * hidden_stride_k,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)

    num_groups: tl.constexpr = BLOCK_K // GROUP_K
    hidden_groups = tl.reshape(tl.abs(hidden), [num_groups, GROUP_K])
    amax = tl.max(hidden_groups, axis=1)
    amax = tl.maximum(amax, 1.0e-4)

    scale = amax / 448.0
    scale_bits = scale.to(tl.uint32, bitcast=True)
    scale_exp = ((scale_bits >> 23) & 0xFF) + ((scale_bits & 0x7FFFFF) != 0).to(
        tl.uint32
    )
    scale_exp = tl.minimum(tl.maximum(scale_exp, 1), 254)
    rounded_scale = (scale_exp << 23).to(tl.float32, bitcast=True)

    hidden_groups = tl.reshape(hidden, [num_groups, GROUP_K])
    scaled = hidden_groups * (1.0 / rounded_scale)[:, None]
    scaled = tl.reshape(scaled, [BLOCK_K])
    fp8 = scaled.to(tl.float8e4nv)
    tl.store(
        x_fp8 + token_id * x_stride_m + k_offsets * x_stride_k,
        fp8,
        mask=k_mask,
    )

    scale_offsets = tl.arange(0, num_groups)
    packed_scale = tl.sum(scale_exp << (scale_offsets * 8), axis=0).to(tl.int32)
    tl.store(
        x_sf + token_id * x_sf_stride_m + k_block_id * x_sf_stride_k,
        packed_scale,
    )

    if k_block_id == 0:
        topk_offsets = tl.arange(0, BLOCK_TOPK)
        topk_mask = topk_offsets < top_k

        ids = tl.load(
            topk_ids + token_id * topk_ids_stride_m + topk_offsets * topk_ids_stride_k,
            mask=topk_mask,
            other=0,
        ).to(IDX_TY)
        tl.store(
            topk_idx_out
            + token_id * topk_idx_stride_m
            + topk_offsets * topk_idx_stride_k,
            ids,
            mask=topk_mask,
        )

        weights = tl.load(
            topk_weights
            + token_id * topk_weights_stride_m
            + topk_offsets * topk_weights_stride_k,
            mask=topk_mask,
            other=0.0,
        )
        tl.store(
            topk_weights_out
            + token_id * topk_weights_out_stride_m
            + topk_offsets * topk_weights_out_stride_k,
            weights,
            mask=topk_mask,
        )


@triton.jit
def _stage_deepseek_v4_mega_moe_inputs_flat_kernel(
    hidden_states,
    x_fp8,
    x_sf_bytes,
    topk_ids,
    topk_weights,
    topk_idx_out,
    topk_weights_out,
    total_elems,
    num_full_chunks,
    tail_elems,
    num_topk_chunks,
    topk_total,
    num_programs,
    CHUNK: tl.constexpr,
    GROUP_K: tl.constexpr,
    TOPK_CHUNK: tl.constexpr,
    IDX_TY: tl.constexpr,
) -> None:
    # Flat streaming form: the whole (num_tokens, hidden_size) plane is one
    # contiguous range, so each program grid-strides over CHUNK-element flat
    # chunks with plain `base + arange` loads/stores (the only fast memory
    # shapes on gcu400). hidden_size % 128 == 0 (host gate) keeps every
    # 4-group scale word inside one chunk and one row.
    #
    # Scale packing: x_sf arrives as a zero-copy uint8 view of the packed
    # int32 words, and the ue8m0 exponent of flat group g lands exactly at
    # byte offset (element_offset // 32), so the "pack 4 exponents into one
    # int32" step is a plain flat byte store. (A register-level
    # reshape+shift+tl.sum pack measured ~483ns per 4096-element chunk on
    # gcu400 -- the small axis reduction dominates the whole kernel; the
    # byte-view store removes it entirely.)
    pid = tl.program_id(0)
    offs = tl.arange(0, CHUNK)
    n_groups: tl.constexpr = CHUNK // GROUP_K
    group_offs = tl.arange(0, n_groups)

    for chunk_id in range(pid, num_full_chunks, num_programs):
        base = chunk_id * CHUNK
        hidden = tl.load(hidden_states + base + offs).to(tl.float32)
        groups = tl.reshape(hidden, (n_groups, GROUP_K))
        amax = tl.maximum(tl.max(tl.abs(groups), axis=1), 1.0e-4)
        scale_bits = (amax / 448.0).to(tl.uint32, bitcast=True)
        scale_exp = ((scale_bits >> 23) & 0xFF) + ((scale_bits & 0x7FFFFF) != 0).to(
            tl.uint32
        )
        scale_exp = tl.minimum(tl.maximum(scale_exp, 1), 254)
        rounded_scale = (scale_exp << 23).to(tl.float32, bitcast=True)
        fp8 = tl.reshape(
            (groups * (1.0 / rounded_scale)[:, None]).to(tl.float8e4nv),
            (CHUNK,),
        )
        tl.store(x_fp8 + base + offs, fp8)
        tl.store(
            x_sf_bytes + base // GROUP_K + group_offs,
            scale_exp.to(tl.uint8),
        )

    if pid == 0 and tail_elems > 0:
        base = num_full_chunks * CHUNK
        mask = offs < tail_elems
        hidden = tl.load(hidden_states + base + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        groups = tl.reshape(hidden, (n_groups, GROUP_K))
        amax = tl.maximum(tl.max(tl.abs(groups), axis=1), 1.0e-4)
        scale_bits = (amax / 448.0).to(tl.uint32, bitcast=True)
        scale_exp = ((scale_bits >> 23) & 0xFF) + ((scale_bits & 0x7FFFFF) != 0).to(
            tl.uint32
        )
        scale_exp = tl.minimum(tl.maximum(scale_exp, 1), 254)
        rounded_scale = (scale_exp << 23).to(tl.float32, bitcast=True)
        fp8 = tl.reshape(
            (groups * (1.0 / rounded_scale)[:, None]).to(tl.float8e4nv),
            (CHUNK,),
        )
        tl.store(x_fp8 + base + offs, fp8, mask=mask)
        tl.store(
            x_sf_bytes + base // GROUP_K + group_offs,
            scale_exp.to(tl.uint8),
            mask=group_offs * GROUP_K < tail_elems,
        )

    if num_topk_chunks > 0:
        toffs = tl.arange(0, TOPK_CHUNK)
        for chunk_id in range(pid, num_topk_chunks, num_programs):
            base = chunk_id * TOPK_CHUNK
            tmask = toffs < topk_total - base
            ids = tl.load(topk_ids + base + toffs, mask=tmask, other=0).to(IDX_TY)
            tl.store(topk_idx_out + base + toffs, ids, mask=tmask)
            weights = tl.load(topk_weights + base + toffs, mask=tmask, other=0.0)
            tl.store(topk_weights_out + base + toffs, weights, mask=tmask)


def stage_deepseek_v4_mega_moe_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
) -> None:
    num_tokens, hidden_size = hidden_states.shape
    if num_tokens == 0:
        return
    if hidden_size % 128 != 0:
        raise ValueError(
            "DeepSeek V4 MegaMoE input staging requires hidden_size to be "
            "a multiple of 128."
        )
    top_k = topk_ids.shape[1]
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "DeepSeek V4 MegaMoE input staging requires topk_weights and "
            "topk_ids to have the same shape."
        )

    # GCU physical layout: torch_gcu narrows int64 tensors to 4-byte slots
    # while they still report int64 ("GCU not support Long use Int replace"),
    # so a Triton store at the reported int64 width strides past the real
    # slots and leaves interleaved garbage. Write through a zero-copy int32
    # slot view instead (element offset == slot offset); with the int64 env
    # switch on (real 8-byte storage) the tensor is passed through untouched.
    if topk_idx_out.dtype == torch.int64:
        if _gcu_real_int64_storage():
            kernel_idx_out = topk_idx_out
            idx_ty = tl.int64
        else:
            kernel_idx_out = _int32_slot_view(topk_idx_out)
            idx_ty = tl.int32
    elif topk_idx_out.dtype == torch.int32:
        kernel_idx_out = topk_idx_out
        idx_ty = tl.int32
    else:
        raise NotImplementedError(
            "DeepSeek V4 MegaMoE input staging requires an int32 or int64 "
            "topk_idx_out tensor on this backend."
        )

    # Fast flat-streaming path. The generic grid (num_tokens,
    # hidden_size/128) pays ~0.19us per 128-element program in dispatch
    # alone on gcu400 (micro-CTA tsunami; 2048x7168 -> 114688 programs ->
    # ~21ms). The flat kernel grid-strides persistent programs over
    # 4096-element flat chunks instead. Requires fully contiguous tensors
    # and int32-safe flat extents; anything else keeps the strided legacy
    # kernel. Reshaped tensors stay at CHUNK <= 4096 elements (compiler
    # P16 safe bound).
    total_elems = num_tokens * hidden_size
    if (
        hidden_states.is_contiguous()
        and x_fp8.is_contiguous()
        and x_sf.is_contiguous()
        and topk_ids.is_contiguous()
        and topk_weights.is_contiguous()
        and kernel_idx_out.is_contiguous()
        and topk_weights_out.is_contiguous()
        and x_sf.dtype == torch.int32
        and x_fp8.dtype == torch.float8_e4m3fn
        and total_elems < 2**31
    ):
        # Zero-copy byte view of the packed ue8m0 scale words: the exponent
        # of flat group g lives at byte offset (flat_element // 32), which
        # turns word packing into a flat byte store (see kernel docstring).
        x_sf_bytes = x_sf.view(torch.uint8)
        num_full_chunks = total_elems // _FLAT_CHUNK
        tail_elems = total_elems - num_full_chunks * _FLAT_CHUNK
        topk_total = num_tokens * top_k
        topk_chunk = max(64, min(_FLAT_TOPK_CHUNK, triton.next_power_of_2(topk_total)))
        num_topk_chunks = (topk_total + topk_chunk - 1) // topk_chunk
        num_programs = min(
            num_full_chunks + (1 if tail_elems else 0) + num_topk_chunks,
            _FLAT_MAX_PROGRAMS,
        )
        with torch_device_fn.device(hidden_states.device):
            _stage_deepseek_v4_mega_moe_inputs_flat_kernel[(num_programs,)](
                hidden_states,
                x_fp8,
                x_sf_bytes,
                topk_ids,
                topk_weights,
                kernel_idx_out,
                topk_weights_out,
                total_elems,
                num_full_chunks,
                tail_elems,
                num_topk_chunks,
                topk_total,
                num_programs,
                CHUNK=_FLAT_CHUNK,
                GROUP_K=32,
                TOPK_CHUNK=topk_chunk,
                IDX_TY=idx_ty,
                num_warps=_FLAT_NUM_WARPS,
                num_stages=_FLAT_NUM_STAGES,
            )
        return

    block_k = 128
    grid = (num_tokens, triton.cdiv(hidden_size, block_k))
    block_topk = triton.next_power_of_2(top_k)
    with torch_device_fn.device(hidden_states.device):
        _stage_deepseek_v4_mega_moe_inputs_kernel[grid](
            hidden_states,
            x_fp8,
            x_sf,
            topk_ids,
            topk_weights,
            kernel_idx_out,
            topk_weights_out,
            hidden_states.stride(0),
            hidden_states.stride(1),
            x_fp8.stride(0),
            x_fp8.stride(1),
            x_sf.stride(0),
            x_sf.stride(1),
            topk_ids.stride(0),
            topk_ids.stride(1),
            topk_weights.stride(0),
            topk_weights.stride(1),
            kernel_idx_out.stride(0),
            kernel_idx_out.stride(1),
            topk_weights_out.stride(0),
            topk_weights_out.stride(1),
            hidden_size,
            top_k,
            BLOCK_K=block_k,
            GROUP_K=32,
            BLOCK_TOPK=block_topk,
            IDX_TY=idx_ty,
            num_warps=4,
        )


_stage_deepseek_v4_mega_moe_inputs = stage_deepseek_v4_mega_moe_inputs
