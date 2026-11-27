# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

from __future__ import annotations

import logging
import math
from functools import reduce
from operator import or_

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.fp8_storage import encode_e4m3fn
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@triton.jit
def _compress_c128_kernel(
    State,
    Request,
    Position,
    Slot,
    Table,
    Work,
    N: tl.constexpr,
    N_REQ: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    N_PAGES: tl.constexpr,
    STATE_PAGE: tl.constexpr,
    STATE_ROW: tl.constexpr,
    STATE_COL: tl.constexpr,
    REQ_STRIDE: tl.constexpr,
    POS_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    TABLE_COL: tl.constexpr,
    WORK_ROW: tl.constexpr,
    WORK_COL: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PAGE_GATHER: tl.constexpr = False,
):
    # Each CTA owns a feature tile. The 128 positions of the window span
    # exactly 16 logical pages, whose physical numbers may be nonconsecutive.
    pid = tl.program_id(0)
    token = (pid // (512 // BLOCK_D)).to(tl.int64)
    col = (pid % (512 // BLOCK_D)) * BLOCK_D + tl.arange(0, BLOCK_D)
    slot = tl.load(Slot + token * SLOT_STRIDE)
    position = tl.load(Position + token * POS_STRIDE)
    request = tl.load(Request + token * REQ_STRIDE).to(tl.int64)
    first_page = (position - 127) // 8
    active = (slot >= 0) & (position >= 127) & ((position & 127) == 127)
    active &= (request >= 0) & (request < N_REQ) & (first_page + 16 <= TABLE_WIDTH)
    if active:
        table = Table + request * TABLE_ROW + first_page * TABLE_COL
        if PAGE_GATHER:
            # Load each page number once and expand its eight physical rows.
            pages = tl.load(table + tl.arange(0, 16) * TABLE_COL)
            state_row = tl.reshape(
                pages[:, None].to(tl.int64) * STATE_PAGE
                + tl.arange(0, 8)[None, :].to(tl.int64) * STATE_ROW,
                (128,),
            )
        else:
            row = tl.arange(0, 128)
            pages = tl.load(table + (row // 8) * TABLE_COL)
            state_row = (
                pages.to(tl.int64) * STATE_PAGE + (row % 8).to(tl.int64) * STATE_ROW
            )
        if (tl.min(pages, 0) >= 0) & (tl.max(pages, 0) < N_PAGES):
            ptrs = State + state_row[:, None] + col[None, :].to(tl.int64) * STATE_COL
            score = tl.load(ptrs + 512 * STATE_COL)
            mass = tl.exp(score - tl.max(score, 0)[None, :])
            weighted = tl.sum(mass * tl.load(ptrs), 0)
            result = weighted / tl.sum(mass, 0)
            tl.store(Work + token * WORK_ROW + col.to(tl.int64) * WORK_COL, result)


@triton.jit
def _store_c128_kernel(
    Work,
    Position,
    Slot,
    Weight,
    Rotary,
    Cache,
    KvSlot,
    Scale,
    eps,
    N: tl.constexpr,
    WORK_ROW: tl.constexpr,
    WORK_COL: tl.constexpr,
    POS_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    W_STRIDE: tl.constexpr,
    ROTARY_ROW: tl.constexpr,
    ROTARY_COL: tl.constexpr,
    ROTARY_LEN: tl.constexpr,
    KV_STRIDE: tl.constexpr,
    N_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    FULL: tl.constexpr,
    FP8: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    slot = tl.load(Slot + token * SLOT_STRIDE)
    position = tl.load(Position + token * POS_STRIDE)
    dst = tl.load(KvSlot + token * KV_STRIDE)
    rotary_pos = position - 127
    active = (slot >= 0) & (position >= 127) & ((position & 127) == 127)
    active &= (dst >= 0) & (dst < N_PAGES * PAGE_SIZE) & (rotary_pos < ROTARY_LEN)
    if active:
        col = tl.arange(0, 512)
        x = tl.load(Work + token * WORK_ROW + col.to(tl.int64) * WORK_COL)
        rrms = tl.rsqrt(tl.sum(x * x, 0) * (1.0 / 512.0) + eps)
        weight = tl.load(Weight + col.to(tl.int64) * W_STRIDE).to(tl.float32)
        x = x * rrms * weight

        even, odd = tl.split(tl.reshape(x, (256, 2)))
        pair = tl.arange(0, 256) - 224
        cs = Rotary + rotary_pos * ROTARY_ROW + pair.to(tl.int64) * ROTARY_COL
        cosine = tl.load(cs, pair >= 0, other=1.0)
        sine = tl.load(cs + 32 * ROTARY_COL, pair >= 0, other=0.0)
        real = tl.where(pair >= 0, even * cosine - odd * sine, even)
        imag = tl.where(pair >= 0, even * sine + odd * cosine, odd)
        rounded = tl.interleave(real, imag).to(tl.bfloat16)
        page = Cache + (dst // PAGE_SIZE) * PAGE_BYTES
        row = dst % PAGE_SIZE
        values = page + row * ROW_BYTES

        if FULL and FP8:
            scale_inv = 1.0
            if HAS_SCALE:
                scale_inv = 1.0 / tl.load(Scale)
            scaled = rounded.to(tl.float32) * scale_inv
            clipped = tl.minimum(tl.maximum(scaled, -FP8_MAX), FP8_MAX)
            tl.store(values + col, encode_e4m3fn(clipped))
        else:
            # A byte view also handles unaligned/padded raw BF16 rows.
            bf = rounded.to(tl.uint16, bitcast=True)
            bf_bytes = tl.interleave(bf.to(tl.uint8), (bf >> 8).to(tl.uint8))
            byte = tl.arange(0, 1024)
            if FULL:
                tl.store(values + byte, bf_bytes)
            else:
                tl.store(values + byte - 448, bf_bytes, byte >= 896)
                groups = tl.reshape(rounded.to(tl.float32), (8, 64))
                absmax = tl.max(tl.abs(groups), 1)
                scale = tl.maximum(absmax / FP8_MAX, 1.0e-4)
                # The biased FP32 exponent of the next power of two is the
                # UE8M0 byte. Build its reciprocal directly from the exponent.
                scale_bits = scale.to(tl.uint32, bitcast=True)
                code = ((scale_bits + 0x7FFFFF) >> 23) & 255
                inv = ((254 - code) << 23).to(tl.float32, bitcast=True)
                scaled = tl.reshape(groups * inv[:, None], (512,))
                clipped = tl.minimum(tl.maximum(scaled, -FP8_MAX), FP8_MAX)
                tl.store(values + col, encode_e4m3fn(clipped), col < 448)
                group = tl.arange(0, 8)
                scale_ptr = page + PAGE_SIZE * ROW_BYTES + row * SCALE_DIM
                tl.store(scale_ptr + group, tl.where(group < 7, code, 0).to(tl.uint8))


# Autotuning repeats idempotent writes to the same rows in the same stream.
# Warmup/rep select the non-graph timer across supported Triton versions.
_OP = "split_kv_compress_norm_rope_insert_sparse_attn_cutedsl"
_NVIDIA = runtime.device.vendor_name == "nvidia"
_ASCEND = runtime.device.vendor_name == "ascend"
if _NVIDIA:
    _compress = libentry()(
        libtuner(
            configs=runtime.get_tuned_config(_OP),
            key=["N"],
            warmup=5,
            rep=15,
            use_cuda_graph=False,
        )(_compress_c128_kernel)
    )
    _store = libentry()(
        libtuner(
            configs=runtime.get_tuned_config(_OP + "_store"),
            key=["N", "FULL", "FP8"],
            warmup=5,
            rep=15,
            use_cuda_graph=False,
        )(_store_c128_kernel)
    )
elif _ASCEND:
    from flaggems_vllm.runtime.backend._ascend.ops.split_kv_compress_norm_rope_insert_sparse_attn_cutedsl import (
        _MAX_GRID,
        _compress_c128_strided,
    )
    from flaggems_vllm.runtime.backend._ascend.ops.split_kv_compress_norm_rope_insert_sparse_attn_cutedsl import (
        _store_c128_kernel as _store,
    )

    _compress = _compress_c128_kernel
else:
    _compress = _compress_c128_kernel
    _store = _store_c128_kernel


# Keep the most recent validated geometry. Triton's compiled launchers own the
# kernels and resolve the calling thread's current stream on every call.
_guard_factory = None
_last_plan = None
if _NVIDIA:
    _guard_factory = getattr(
        getattr(getattr(torch._C, "_dynamo", None), "guards", None),
        "TensorGuards",
        None,
    )
    _hooks = getattr(getattr(triton, "knobs", None), "runtime", None)


def _empty_launch_hooks():
    # With live or unrecognized hooks, use the regular tensor-valued dispatch.
    return not (
        getattr(getattr(_hooks, "launch_enter_hook", None), "calls", True)
        or getattr(getattr(_hooks, "launch_exit_hook", None), "calls", True)
    )


def _has_full_signature(compiled, jit):
    # Triton launchers vary in whether they receive constexpr arguments.
    # Reuse only a compiled signature which explicitly lists the full ABI.
    signature = getattr(getattr(compiled, "src", None), "signature", {})
    return (
        tuple(signature) == tuple(jit.arg_names)
        and signature[jit.arg_names[-1]] == "constexpr"
    )


class _CudaPlan:
    def __init__(self, tensors, options, compress, store):
        self.check = _guard_factory(
            *tensors,
            dynamic_dims_sizes=[list(t.shape) for t in tensors],
            dynamic_dims_strides=[list(t.stride()) for t in tensors],
        ).check
        self.device = tensors[0].device.index
        self.options = options
        self.compress, self.c_args = compress
        self.store, self.s_args = store

    def try_launch(self, tensors, options):
        if (
            options != self.options
            or torch.cuda.current_device() != self.device
            or not _empty_launch_hooks()
            or not self.check(*tensors)
        ):
            return False
        pointers = tuple(map(torch.Tensor.data_ptr, tensors))
        if reduce(or_, pointers) % _compress.divisibility:
            return False
        state, req, pos, slot, table, work, weight, rotary, cache, dst = pointers[:10]
        scale = pointers[10] if len(pointers) == 11 else work
        self.compress(state, req, pos, slot, table, work, *self.c_args)
        self.store(work, pos, slot, weight, rotary, cache, dst, scale, *self.s_args)
        return True


def split_kv_compress_norm_rope_insert_sparse_attn_cutedsl(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compressed_kv: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    kv_block_stride: int,
    head_size: int = 512,
    state_width: int = 512,
    rope_head_dim: int = 64,
    fp8_max: float = 448.0,
    quant_block: int = 64,
    token_stride: int = 576,
    scale_dim: int = 8,
    compress_ratio: int = 128,
    overlap: bool = False,
    store_full_kv: bool = False,
    store_full_fp8: bool = False,
    fp8_scale: torch.Tensor | None = None,
) -> None:
    """Write C128 boundary rows into the FP32 workspace and the paged K cache.

    Each state row contains 512 values followed by their 512 scores. Packed
    pages contain all token payloads followed by all scale rows. Full caches
    use the physical token and page strides of the supplied tensor.
    """
    logger.debug("GEMS SPLIT_KV_COMPRESS_NORM_ROPE_INSERT_SPARSE_ATTN_CUTEDSL")
    global _last_plan
    if _guard_factory is not None:
        inputs = (
            state_cache,
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table,
            compressed_kv,
            rms_norm_weight,
            cos_sin_cache,
            k_cache,
            kv_slot_mapping,
        )
        if store_full_fp8 and fp8_scale is not None:
            inputs += (fp8_scale,)
        options = (
            block_size,
            rms_norm_eps,
            kv_cache_block_size,
            kv_block_stride,
            head_size,
            state_width,
            rope_head_dim,
            fp8_max,
            quant_block,
            token_stride,
            scale_dim,
            compress_ratio,
            overlap,
            store_full_kv,
            store_full_fp8,
            store_full_fp8 and fp8_scale is not None,
        )
        plan = _last_plan
        if plan is not None and plan.try_launch(inputs, options):
            return
    if (head_size, state_width, block_size, compress_ratio, overlap) != (
        512,
        512,
        8,
        128,
        False,
    ):
        raise ValueError(
            "C128 requires head_size=state_width=512, block_size=8 and overlap=False"
        )
    if (
        k_cache.ndim != 3
        or kv_cache_block_size != k_cache.shape[1]
        or kv_cache_block_size <= 0
    ):
        raise ValueError(
            "k_cache must be 3D with a positive matching kv_cache_block_size"
        )
    if positions.numel() == 0:
        return
    if (rope_head_dim, quant_block) != (64, 64):
        raise ValueError("rope_head_dim and quant_block must both be 64")
    if store_full_fp8 and not store_full_kv:
        raise ValueError("store_full_fp8 requires store_full_kv")
    if not math.isfinite(fp8_max) or fp8_max <= 0:
        raise ValueError("fp8_max must be finite and positive")
    if state_cache.ndim != 3 or state_cache.shape[1:] != (8, 1024):
        raise ValueError("state_cache must have shape [pages, 8, 1024]")
    if block_table.ndim != 2 or cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != 64:
        raise ValueError("expected a 2D block_table and a [positions, 64] rotary cache")
    if rms_norm_weight.shape != (512,) or rms_norm_weight.dtype not in (
        torch.float32,
        torch.bfloat16,
    ):
        raise ValueError("rms_norm_weight must be a BF16 or FP32 vector of length 512")
    if compressed_kv.ndim != 2 or compressed_kv.shape[1] != 512:
        raise ValueError("compressed_kv must be a [tokens, 512] FP32 workspace")
    if any(
        t.dtype != torch.float32 for t in (state_cache, cos_sin_cache, compressed_kv)
    ):
        raise ValueError("state_cache, cos_sin_cache and compressed_kv must be FP32")
    if block_table.dtype != torch.int32 or token_to_req_indices.dtype != torch.int32:
        raise ValueError("block_table and token_to_req_indices must be int32")
    mappings = (positions, slot_mapping, kv_slot_mapping)
    if any(t.dtype != torch.int64 for t in mappings):
        raise ValueError("positions, slot_mapping and kv_slot_mapping must be int64")
    if any(t.ndim != 1 for t in (*mappings, token_to_req_indices)):
        raise ValueError("positions and token mappings must be vectors")
    if compressed_kv.shape[0] < min(positions.numel(), slot_mapping.numel()):
        raise ValueError("compressed_kv must cover the position and state-slot prefix")
    if (
        compressed_kv.stride(1) == 0
        or compressed_kv.stride(0) < 511 * compressed_kv.stride(1) + 1
    ):
        raise ValueError("compressed_kv must have nonoverlapping rows")

    if k_cache.stride(2) != 1:
        raise ValueError("k_cache needs a unit inner stride")
    if not store_full_kv and kv_block_stride != k_cache.stride(0):
        raise ValueError("packed kv_block_stride must match k_cache.stride(0) in bytes")
    if store_full_kv:
        logical_dtype = (
            getattr(torch, "float8_e4m3fn", None) if store_full_fp8 else torch.bfloat16
        )
        if k_cache.dtype not in (logical_dtype, torch.uint8):
            raise ValueError(
                "full cache requires the selected BF16/E4M3FN dtype or its uint8 byte view"
            )
    elif k_cache.dtype != torch.uint8:
        raise ValueError("packed k_cache must have uint8 dtype")

    item_size = k_cache.element_size()
    page_bytes = k_cache.stride(0) * item_size
    row_bytes = k_cache.stride(1) * item_size
    width_bytes = k_cache.shape[2] * item_size
    span = (kv_cache_block_size - 1) * row_bytes + width_bytes
    if store_full_kv:
        payload = 512 if store_full_fp8 else 1024
        if width_bytes < payload or (kv_cache_block_size > 1 and row_bytes < payload):
            raise ValueError("full-cache rows must hold the entire 512-element payload")
        required = (kv_cache_block_size - 1) * row_bytes + payload
    else:
        if token_stride < 576 or scale_dim < 8:
            raise ValueError(
                "packed C128 requires token_stride >= 576 and scale_dim >= 8"
            )
        row_bytes = token_stride
        required = (
            kv_cache_block_size * token_stride
            + (kv_cache_block_size - 1) * scale_dim
            + 8
        )
    if required > span or (k_cache.shape[0] > 1 and required > page_bytes):
        raise ValueError(
            "k_cache page is too small for the selected payload and scale layout"
        )

    device = state_cache.device
    tensors = (
        *mappings,
        token_to_req_indices,
        block_table,
        compressed_kv,
        rms_norm_weight,
        cos_sin_cache,
        k_cache,
    )
    if any(t.device != device for t in tensors):
        raise ValueError("all C128 inputs must be on the same device")
    has_scale = store_full_fp8 and fp8_scale is not None
    if has_scale and (
        fp8_scale.device != device
        or fp8_scale.dtype != torch.float32
        or fp8_scale.numel() != 1
    ):
        raise ValueError("fp8_scale must be one FP32 value on the cache device")

    n = min(positions.numel(), slot_mapping.numel())
    n_compress = min(n, token_to_req_indices.numel())
    n_store = min(n, kv_slot_mapping.numel())
    compression = None
    storage = None
    with runtime.torch_device_fn.device(device):
        if n_compress and state_cache.shape[0] and block_table.numel():

            def grid(meta):
                return (n_compress * (512 // meta["BLOCK_D"]),)

            # 128 x 16 bounds the cross-backend reduction tile.
            opts = {} if _NVIDIA else {"BLOCK_D": 16, "num_warps": 4, "num_stages": 1}
            c_args = (
                n_compress,
                *block_table.shape,
                state_cache.shape[0],
                *state_cache.stride(),
                token_to_req_indices.stride(0),
                positions.stride(0),
                slot_mapping.stride(0),
                *block_table.stride(),
                *compressed_kv.stride(),
            )
            compress_kernel = _compress
            compress_grid = grid
            if _ASCEND:
                tiles = 512 // opts["BLOCK_D"]
                if n_compress * tiles > _MAX_GRID:
                    compress_kernel = _compress_c128_strided
                    opts["COMPRESS_KERNEL"] = _compress_c128_kernel
                    compress_grid = ((_MAX_GRID // tiles) * tiles,)
            compression = compress_kernel[compress_grid](
                state_cache,
                token_to_req_indices,
                positions,
                slot_mapping,
                block_table,
                compressed_kv,
                *c_args,
                PAGE_GATHER=_NVIDIA,
                **opts,
            )
        if n_store and k_cache.shape[0] and cos_sin_cache.shape[0]:
            opts = {} if _NVIDIA else {"num_warps": 4, "num_stages": 1}
            s_args = (
                rms_norm_eps,
                n_store,
                *compressed_kv.stride(),
                positions.stride(0),
                slot_mapping.stride(0),
                rms_norm_weight.stride(0),
                *cos_sin_cache.stride(),
                cos_sin_cache.shape[0],
                kv_slot_mapping.stride(0),
                k_cache.shape[0],
                kv_cache_block_size,
                page_bytes,
                row_bytes,
                scale_dim,
                store_full_kv,
                store_full_fp8,
                has_scale,
                fp8_max,
            )
            store_grid = min(n_store * 8, _MAX_GRID) if _ASCEND else n_store
            storage = _store[(store_grid,)](
                compressed_kv,
                positions,
                slot_mapping,
                rms_norm_weight,
                cos_sin_cache,
                k_cache.view(torch.uint8),
                kv_slot_mapping,
                fp8_scale if has_scale else compressed_kv,
                *s_args,
                **opts,
            )
        if (
            _guard_factory is not None
            and compression
            and storage
            and _empty_launch_hooks()
            and _has_full_signature(compression[0], _compress_c128_kernel)
            and _has_full_signature(storage[0], _store_c128_kernel)
            and all(type(t) is torch.Tensor for t in inputs)
            and reduce(or_, map(torch.Tensor.data_ptr, inputs)) % _compress.divisibility
            == 0
        ):
            block = compression[1]["BLOCK_D"]
            compress_grid = (n_compress * (512 // block), 1, 1)
            try:
                _last_plan = _CudaPlan(
                    inputs,
                    options,
                    (compression[0][compress_grid], (*c_args, block, True)),
                    (storage[0][(n_store, 1, 1)], s_args),
                )
            except (AttributeError, TypeError, RuntimeError):
                # An unavailable guard API leaves these checked launches active.
                pass


__all__ = ["split_kv_compress_norm_rope_insert_sparse_attn_cutedsl"]
