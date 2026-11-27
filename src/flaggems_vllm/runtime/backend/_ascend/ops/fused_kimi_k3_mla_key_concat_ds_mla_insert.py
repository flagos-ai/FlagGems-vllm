# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

import logging
from functools import reduce
from operator import or_
from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.fused_kimi_k3_mla_key_concat_ds_mla_insert import (
    _ascend_fma,
    _ascend_pack_tail_bytes,
    _fixed_kernel,
    _prefill,
    _write_cache_prefix,
)
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _two_sum(a, b):
    # Knuth's error-free transform. Each operation must round separately.
    s = a + b
    t = s - a
    return s, (a - (s - t)) + (b - t)


@triton.jit
def _outside(x):
    bits = x.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    return ((bits != 0) & (bits < 0x2F800000)) | (bits >= 0x4F800000)


@triton.jit
def _bounded_fma(x, a, c):
    # With nonzero magnitudes in [2**-32, 2**32), the least possible
    # product bit is 2**-97, so every nonzero intermediate is normal.
    outside = _outside(x) | _outside(a) | _outside(c)
    if tl.max(outside.to(tl.int32), 0) != 0:
        result = _ascend_fma(x, a, c)
    else:
        # Split a into 13 and 11 significant bits so both products are exact.
        high = (a.to(tl.int32, bitcast=True) & -2048).to(tl.float32, bitcast=True)
        p = x * a
        pe = x * (a - high) - (p - x * high)
        sh, sl = _two_sum(p, c)
        eh, el = _two_sum(sl, pe)
        # Round the exact residual to odd before the final RN32 addition.
        # A nonzero error on an even word moves one ULP toward the exact value.
        eb = eh.to(tl.int32, bitcast=True)
        delta = tl.where((eh < 0) == (el < 0), 1, -1)
        eo = tl.where((el != 0) & ((eb & 1) == 0), eb + delta, eb)
        result = sh + eo.to(tl.float32, bitcast=True)
        # With a zero product and a zero addend, only -0 + -0 gives -0.
        product_sign = x.to(tl.int32, bitcast=True) ^ a.to(tl.int32, bitcast=True)
        zero = (product_sign & c.to(tl.int32, bitcast=True) & -2147483648).to(
            tl.float32, bitcast=True
        )
        result = tl.where((p == 0) & (c == 0), zero, result)
    return result


@triton.jit
def _rotated_pairs(
    src, pos, table, stride, HEAD_TILE: tl.constexpr, CACHE: tl.constexpr
):
    pair = tl.arange(0, HEAD_TILE * 32)
    source = src + (pair // 32) * stride + (pair % 32) * 2
    x = tl.load(source).to(tl.float32)
    y = tl.load(source + 1).to(tl.float32)
    phase = table + tl.load(pos) * 64 + pair % 32
    cos = tl.load(phase)
    sin = tl.load(phase + 32)
    neg = ((y * sin).to(tl.int32, bitcast=True) ^ -2147483648).to(
        tl.float32, bitcast=True
    )
    even = _bounded_fma(x, cos, neg)
    # The native cache writer contracts the odd source with cosine.
    if CACHE:
        odd = _bounded_fma(y, cos, x * sin)
    else:
        odd = _bounded_fma(x, sin, y * cos)
    return even, odd


@triton.jit
def _head_pe(src, pos, table, stride, HEAD_TILE: tl.constexpr):
    even, odd = _rotated_pairs(src, pos, table, stride, HEAD_TILE, False)
    dtype: tl.constexpr = src.dtype.element_ty
    return tl.reshape(tl.interleave(even.to(dtype), odd.to(dtype)), (HEAD_TILE, 64))


# The first input's stride, shape or pointer alignment must not specialize a
# reusable binary. RoPE and the vector tile still select static instructions.
@triton.jit(
    do_not_specialize=[
        "Q",
        "K_NOPE",
        "K_PE",
        "LATENT",
        "OUT",
        "CACHE",
        "SLOTS",
        "POS",
        "TABLE",
        "Q_S0",
        "Q_S1",
        "KN_S0",
        "KN_S1",
        "PE_S0",
        "CACHE_S0",
        "CACHE_S1",
        "SLOT_S0",
        "POS_S0",
        "HEADS",
        "PAGE",
        "HAS_CACHE",
        "work_base",
    ]
)
def _ascend_prefill_kernel(
    Q,
    K_NOPE,
    K_PE,
    LATENT,
    OUT,
    CACHE,
    SLOTS,
    POS,
    TABLE,
    Q_S0,
    Q_S1,
    KN_S0,
    KN_S1,
    PE_S0,
    CACHE_S0,
    CACHE_S1,
    SLOT_S0,
    POS_S0,
    HEADS,
    PAGE,
    ROPE: tl.constexpr,
    HAS_CACHE,
    HEAD_TILE: tl.constexpr,
    work_base,
):
    # A separate program owns the four quantization groups for each token.
    # This also covers a token with zero heads, leaving only its cache item.
    groups = HEADS // HEAD_TILE
    # Work is nonnegative. Use 32-bit division for ordinary grids and retain
    # a wide work index if a very large logical grid crosses 2**32.
    if work_base.dtype == tl.int32 and HEADS.dtype == tl.int32:
        item = tl.program_id(0).to(tl.uint32) + work_base.to(tl.uint32)
        span = groups.to(tl.uint32) + 1
    else:
        item = tl.program_id(0).to(tl.uint64) + work_base.to(tl.uint64)
        span = groups.to(tl.uint64) + 1
    token = (item // span).to(tl.int64)
    group = (item % span).to(tl.int64)
    pos = POS + token * POS_S0
    if group == groups:
        if HAS_CACHE:
            slot = tl.load(SLOTS + token * SLOT_S0)
            if slot >= 0:
                row = CACHE + slot // PAGE * CACHE_S0 + slot % PAGE * CACHE_S1
                _write_cache_prefix(LATENT + token * 512, row)
                src = K_PE + token * PE_S0
                if ROPE:
                    even, odd = _rotated_pairs(src, pos, TABLE, 0, 1, True)
                    tail = _ascend_pack_tail_bytes(src, even, odd)
                else:
                    bits = tl.load(src + tl.arange(0, 64)).to(tl.uint16, bitcast=True)
                    tail = tl.interleave(bits.to(tl.uint8), (bits >> 8).to(tl.uint8))
                tl.store(row + 528 + tl.arange(0, 128), tail)
    else:
        head = group * HEAD_TILE + tl.arange(0, HEAD_TILE)
        col = tl.arange(0, 128)[None, :]
        src = K_NOPE + token * KN_S0 + head[:, None] * KN_S1
        dst = OUT + (token * HEADS + head[:, None]) * 192
        tl.store(dst + col, tl.load(src + col))

        pe_src = K_PE + token * PE_S0
        if ROPE:
            key_tail = _head_pe(pe_src, pos, TABLE, 0, 1)
        else:
            key_tail = tl.load(pe_src + tl.arange(0, 64))[None, :]
        tail_col = 128 + tl.arange(0, 64)[None, :]
        tl.store(dst + tail_col, key_tail)
        if ROPE:
            query_base = Q + token * Q_S0 + group * HEAD_TILE * Q_S1 + 128
            rotated = _head_pe(query_base, pos, TABLE, Q_S1, HEAD_TILE)
            query_tail = Q + token * Q_S0 + head[:, None] * Q_S1 + 128
            tl.store(query_tail + tl.arange(0, 64)[None, :], rotated)


prefill_kernel = libentry()(_ascend_prefill_kernel)


_plans = {}
_guards = None
_knobs = None
if triton.__version__ == "3.5.1" and torch.__version__.split("+")[0] == "2.10.0":
    try:
        from triton import knobs
    except ImportError:
        knobs = None
    _knobs = getattr(knobs, "runtime", None)
    _guards = getattr(
        getattr(getattr(torch._C, "_dynamo", None), "guards", None),
        "TensorGuards",
        None,
    )


class _Launch(NamedTuple):
    guard: object
    device: int
    run: object
    scalars: tuple


def _hooks_active():
    return getattr(
        getattr(_knobs, "launch_enter_hook", None), "calls", True
    ) or getattr(getattr(_knobs, "launch_exit_hook", None), "calls", True)


def _record(compiled, programs, tensors, scalars, page, rope):
    try:
        if (
            not _guards
            or _hooks_active()
            or any(type(t) is not torch.Tensor for t in tensors)
            or reduce(or_, map(torch.Tensor.data_ptr, tensors))
            % _fixed_kernel.divisibility
        ):
            return
        target = compiled.metadata.target
        if target.backend != "npu" or target.arch != "Ascend910B1":
            return
        check = _guards(
            *tensors, dynamic_dims_sizes=None, dynamic_dims_strides=None
        ).check
        if not check(*tensors):
            return
        plan = _Launch(
            check, tensors[0].device.index, compiled[programs, 1, 1], scalars
        )
    except (AttributeError, RuntimeError, TypeError):
        # The validated call has already completed its single in-place launch.
        return
    if len(_plans) >= 64:
        _plans.clear()
    _plans[tensors[0].numel(), page, rope] = plan


def _launch(tensors, page, rope):
    plan = _plans.get((tensors[0].numel(), page, rope))
    if plan is None:
        return False
    try:
        if (
            any(type(t) is not torch.Tensor for t in tensors)
            or not plan.guard(*tensors)
            or torch_device_fn.current_device() != plan.device
            or _hooks_active()
            or reduce(or_, map(torch.Tensor.data_ptr, tensors))
            % _fixed_kernel.divisibility
        ):
            return False
    except (AttributeError, RuntimeError, TypeError):
        return False
    if not rope:
        tensors = (*tensors, tensors[6], tensors[2])
    # The public compiled launcher reads the current NPU stream for every call.
    plan.run(*tensors, *plan.scalars)
    return True


def _rope_config(q, head_tile):
    tokens, heads = q.shape[:2]
    # Keep enough programs resident before combining more heads in one program.
    if head_tile == 8 and heads % 32 == 0 and tokens * (heads // 32 + 1) >= 128:
        head_tile = 32
    return head_tile, {"enable_fp_fusion": False}


def fused_kimi_k3_mla_key_concat_ds_mla_insert(
    q: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_out: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    cache_block_size: int,
    position_ids: Optional[torch.Tensor] = None,
    cos_sin_cache: Optional[torch.Tensor] = None,
) -> None:
    """Write full keys, optionally rotate Q in place, and insert 656-byte cache rows."""
    logger.debug("GEMS_ASCEND FUSED_KIMI_K3_MLA_KEY_CONCAT_DS_MLA_INSERT")
    rope = position_ids is not None
    if rope != (cos_sin_cache is not None):
        raise ValueError("position_ids and cos_sin_cache must be provided together")
    tensors = (q, k_nope, k_pe, kv_c_normed, k_out, k_cache, slot_mapping)
    if rope:
        tensors += (position_ids, cos_sin_cache)
    if (
        _plans
        and type(q) is torch.Tensor
        and type(cache_block_size) is int
        and _launch(tensors, cache_block_size, rope)
    ):
        return
    use_ascend_kernel = False
    if (
        triton.__version__ == "3.5.1"
        and isinstance(q, torch.Tensor)
        and q.device.type == "npu"
        # Large copy-only outputs benefit from compile-time strides in the
        # short copy kernel, which takes only a few seconds to compile.
        and (rope or q.numel() < 1024 * 1024)
    ):
        try:
            use_ascend_kernel = (
                torch_device_fn.get_device_name(q.device) == "Ascend910B1"
            )
        except (AttributeError, RuntimeError, TypeError):
            pass
    _prefill(
        tensors,
        cache_block_size,
        rope,
        _record,
        _rope_config if use_ascend_kernel and rope else None,
        prefill_kernel if use_ascend_kernel else _fixed_kernel,
    )
