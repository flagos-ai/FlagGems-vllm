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

import logging
import os

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry

logger = logging.getLogger(__name__)

# enum Fp8KVCacheDataType
FP8_KV_CACHE_DATA_TYPE_AUTO = tl.constexpr(0)
FP8_KV_CACHE_DATA_TYPE_FP8E4M3 = tl.constexpr(1)
FP8_KV_CACHE_DATA_TYPE_FP8E5M2 = tl.constexpr(2)


def _slot_word_stride() -> int:
    # torch_gcu without TORCH_GCU_ENABLE_INT64_AND_UINT64 keeps ATen int64
    # values as compact int32 words (element j lives at word j); with the gate
    # on, values sit in the low word of each 8-byte slot (word 2*j). Either
    # way a Triton int64 load at the 8-byte dtype stride reads
    # neighbor-joined garbage, so slots are always read through a zero-copy
    # int32 word view with the matching word stride (same scheme as
    # indexer_k_quant_and_cache / cross_entropy_loss).
    return 2 if os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64") == "1" else 1


def _slot_words(slot_mapping: torch.Tensor):
    # Zero-copy int32 view of slot ids: value-exact because slot ids live in
    # [-1, num_blocks * block_size) with total slots far below 2**31, and the
    # two's-complement low word of -1 still reads as -1. Avoids the per-call
    # ATen int64->int32 cast kernel (measured 1.0-3.0us of device time per
    # call in the bench window on GCU400).
    if slot_mapping.dtype == torch.int64:
        return slot_mapping.contiguous().view(torch.int32), _slot_word_stride()
    return slot_mapping, 1


# GCU400 schedules one CTA per SP per wave (48 SPs); grids above that pay a
# per-program cost of ~60-70ns for these tiny copies (measured 2026-11-12:
# 84-token grids sit ~4us above the 21/42-token single-wave floor). Extra
# tokens are picked up by the grid-stride loop inside the kernel instead.
GCU_MLA_GRID_CAP = 48


@libentry()
@triton.jit
def concat_and_cache_mla_kernel(
    # pointers
    kv_c_ptr,  # in,  [num_tokens, kv_lora_rank]
    k_pe_ptr,  # in,  [num_tokens, pe_dim]
    kv_cache_ptr,  # out, [num_blocks, block_size, kv_lora_rank + pe_dim]
    slot_words_ptr,  # in, int32 word view of slot_mapping (see _slot_words)
    # strides / scalars
    slot_word_stride,
    block_stride,
    entry_stride,
    kv_c_stride,
    k_pe_stride,
    # dims
    kv_lora_rank,
    pe_dim,
    block_size,  # kv cache block size
    num_tokens,
    num_progs,  # grid size, also the token loop stride
    scale_ptr,
    # data type
    kv_dtype: tl.constexpr,  # one of Fp8KVCacheDataType
    BLOCK_KV: tl.constexpr,
    BLOCK_PE: tl.constexpr,
    EVEN_KV: tl.constexpr,  # kv_lora_rank % BLOCK_KV == 0: drop masks
    EVEN_PE: tl.constexpr,  # pe_dim % BLOCK_PE == 0: drop masks
):
    program_id = tl.program_id(0)
    if kv_dtype != FP8_KV_CACHE_DATA_TYPE_AUTO:
        scale_val = tl.load(scale_ptr)
    for token_idx in range(program_id, num_tokens, num_progs):
        slot_idx = tl.load(slot_words_ptr + token_idx * slot_word_stride)

        # Skip padded tokens
        if slot_idx >= 0:
            # Calculate cache position. Fold every scalar offset term into a
            # single i64 expression: element offsets >= 2**31 get truncated
            # by i32 addptr chains on this backend, and kv caches larger than
            # 2**31 elements are routine for MLA. The tile vector offset is
            # applied after the scalar addend, which stays safe.
            block_id = slot_idx // block_size
            block_offset = slot_idx % block_size
            cache_base = (
                block_id.to(tl.int64) * block_stride
                + block_offset.to(tl.int64) * entry_stride
            )
            pe_base = cache_base + kv_lora_rank

            # Process kv_c section
            for i in range(0, kv_lora_rank, BLOCK_KV):
                idx = i + tl.arange(0, BLOCK_KV)

                src_ptr = kv_c_ptr + token_idx * kv_c_stride + idx
                dst_ptr = kv_cache_ptr + cache_base + idx

                if EVEN_KV:
                    val = tl.load(src_ptr)
                else:
                    val = tl.load(src_ptr, mask=idx < kv_lora_rank, other=0)

                if kv_dtype != FP8_KV_CACHE_DATA_TYPE_AUTO:
                    if kv_dtype == FP8_KV_CACHE_DATA_TYPE_FP8E4M3:
                        val = (val / scale_val).to(tl.float8e4nv)
                    elif kv_dtype == FP8_KV_CACHE_DATA_TYPE_FP8E5M2:
                        val = (val / scale_val).to(tl.float8e5)
                    val = val.to(tl.uint8, bitcast=True)

                if EVEN_KV:
                    tl.store(dst_ptr, val)
                else:
                    tl.store(dst_ptr, val, mask=idx < kv_lora_rank)

            # Process k_pe section
            for j in range(0, pe_dim, BLOCK_PE):
                idx = j + tl.arange(0, BLOCK_PE)

                src_ptr = k_pe_ptr + token_idx * k_pe_stride + idx
                dst_ptr = kv_cache_ptr + pe_base + idx

                if EVEN_PE:
                    val = tl.load(src_ptr)
                else:
                    val = tl.load(src_ptr, mask=idx < pe_dim, other=0)

                if kv_dtype != FP8_KV_CACHE_DATA_TYPE_AUTO:
                    if kv_dtype == FP8_KV_CACHE_DATA_TYPE_FP8E4M3:
                        val = (val / scale_val).to(tl.float8e4nv)
                    elif kv_dtype == FP8_KV_CACHE_DATA_TYPE_FP8E5M2:
                        val = (val / scale_val).to(tl.float8e5)
                    val = val.to(tl.uint8, bitcast=True)

                if EVEN_PE:
                    tl.store(dst_ptr, val)
                else:
                    tl.store(dst_ptr, val, mask=idx < pe_dim)


class ConcatAndCacheMla(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        scale: torch.Tensor,
    ):
        if kv_cache.ndim == 4 and kv_cache.shape[2] == 1:
            kv_cache = kv_cache.squeeze(2)
        if kv_c.ndim != 2 or k_pe.ndim != 2 or kv_cache.ndim != 3:
            raise ValueError("MLA cache expects KV[N,D], RoPE[N,R] and paged [B,P,D+R]")
        if k_pe.shape[0] != kv_c.shape[0] or slot_mapping.shape != (kv_c.shape[0],):
            raise ValueError("MLA cache inputs need one slot per token")
        if k_pe.dtype != kv_c.dtype or slot_mapping.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("MLA cache input dtype mismatch")
        # Compiled zero-width RoPE views can have zero strides. The kernel
        # never accesses that empty section; nonempty inputs keep their guards.
        if (
            kv_cache.shape[-1] != kv_c.shape[-1] + k_pe.shape[-1]
            or any(t.stride(-1) != 1 for t in (kv_c, kv_cache, slot_mapping))
            or (k_pe.shape[-1] > 0 and k_pe.stride(-1) != 1)
        ):
            raise ValueError(
                "MLA cache requires matching contiguous feature dimensions"
            )
        if kv_cache_dtype != "auto" and kv_cache.dtype != torch.uint8:
            raise ValueError("For FP8 kv_cache must be uint8 dtype")
        if kv_cache_dtype == "auto" and kv_cache.dtype != kv_c.dtype:
            raise ValueError("For auto mode kv_cache must match input dtype")

        # Map string dtype to internal constants
        kv_dtype_map = {
            "auto": FP8_KV_CACHE_DATA_TYPE_AUTO,
            "fp8": FP8_KV_CACHE_DATA_TYPE_FP8E4M3,
            "fp8e4m3": FP8_KV_CACHE_DATA_TYPE_FP8E4M3,
            "fp8e5m2": FP8_KV_CACHE_DATA_TYPE_FP8E5M2,
        }
        kv_dtype = kv_dtype_map.get(kv_cache_dtype)
        if kv_dtype is None:
            raise ValueError(f"Unsupported kv_cache_dtype: {kv_cache_dtype}")
        kv_dtype = int(kv_dtype)  # tl.constexpr->int

        kv_lora_rank = kv_c.size(1)
        pe_dim = k_pe.size(1)
        num_tokens = slot_mapping.size(0)

        # make sure `scale` is a scalar tensor
        if scale.numel() != 1:
            raise ValueError("MLA cache scale must be scalar")

        device = kv_c.device
        if any(t.device != device for t in (k_pe, kv_cache, slot_mapping, scale)):
            raise ValueError("MLA cache tensors must share the input device")
        if num_tokens == 0:
            return None

        # torch_gcu keeps ATen int64 values as narrow int32 words (with the
        # int64 gate on, as the low word of each 8-byte slot), and a Triton
        # int64 load at the 8-byte dtype stride reads neighbor-joined garbage
        # (triton_i64_narrow_load family). Slots are therefore always read
        # through a zero-copy int32 word view with an env-aware word stride;
        # slot ids live in [-1, num_blocks * block_size) with total slots far
        # below 2**31, so the low word is value-exact.
        slot_words, slot_word_stride = _slot_words(slot_mapping)

        # configure kernel launch: cap programs at one scheduling wave on
        # GCU400 (48 SPs, one CTA per SP); the kernel's grid-stride loop
        # walks the remaining tokens.
        num_progs = min(num_tokens, GCU_MLA_GRID_CAP)
        grid = (num_progs,)
        if kv_lora_rank == 0:
            raise NotImplementedError("MLA cache requires a positive latent dimension")
        BLOCK_KV = triton.next_power_of_2(min(kv_lora_rank, 512))
        BLOCK_PE = triton.next_power_of_2(max(pe_dim, 1))
        EVEN_KV = kv_lora_rank % BLOCK_KV == 0
        EVEN_PE = pe_dim % BLOCK_PE == 0

        assert kv_cache.dim() == 3, "kv_cache must be a 3D tensor"
        assert (
            kv_cache.size(2) == kv_lora_rank + pe_dim
        ), "kv_cache's last dimension must match kv_lora_rank + pe_dim"
        with torch_device_fn.device(device):
            concat_and_cache_mla_kernel[grid](
                kv_c,
                k_pe,
                kv_cache,
                slot_words,
                slot_word_stride,
                kv_cache.stride(0),  # block_stride
                kv_cache.stride(1),  # entry_stride
                kv_c.stride(0),  # kv_c_stride
                k_pe.stride(0),  # k_pe_stride
                kv_lora_rank,
                pe_dim,
                kv_cache.size(1),  # kv cache block_size
                num_tokens,
                num_progs,
                scale,
                kv_dtype=kv_dtype,
                BLOCK_KV=BLOCK_KV,
                BLOCK_PE=BLOCK_PE,
                EVEN_KV=EVEN_KV,
                EVEN_PE=EVEN_PE,
            )
        return None


def concat_and_cache_mla(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    scale: torch.Tensor,
) -> None:
    logger.debug("GEMS CONCAT_AND_CACHE_MLA")
    return ConcatAndCacheMla.apply(
        kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale
    )
