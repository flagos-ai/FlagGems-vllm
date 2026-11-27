# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

import logging
from copy import copy
from functools import partial, reduce
from operator import or_
from typing import NamedTuple, Optional

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.fp8_storage import encode_e4m3fn
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry

logger = logging.getLogger(__name__)


@triton.jit
def _ascend_encode_e4m3fn(x):
    # BiSheng on 910B cannot lower the unsigned comparisons in encode_e4m3fn.
    # Clamp the positive FP32 bit pattern to 448, preserving the sign separately.
    bits = x.to(tl.int32, bitcast=True)
    magnitude = bits & 0x7FFFFFFF
    finite = tl.minimum(magnitude, 0x43E00000)
    normal = ((finite + 0x7FFFF + ((finite >> 20) & 1)) >> 20) - 960
    # At 2**23 the ULP is one. Adding it rounds |x| * 512 to the nearest
    # subnormal index with ties to even, as in Ascend per_token_group_quant_fp8.
    rounded = finite.to(tl.float32, bitcast=True) * 512.0 + 8388608.0
    subnormal = rounded.to(tl.int32, bitcast=True) - 0x4B000000
    payload = tl.where(finite < 0x3C800000, subnormal, normal)
    payload = tl.where(magnitude > 0x7F800000, 0x7F, payload)
    return (payload | ((bits >> 24) & 0x80)).to(tl.uint8)


_encode_cache_fp8 = (
    _ascend_encode_e4m3fn if runtime.device.vendor_name == "ascend" else encode_e4m3fn
)
_IS_HYGON = tl.constexpr(runtime.device.vendor_name == "hygon")
_IS_ASCEND = tl.constexpr(runtime.device.vendor_name == "ascend")


@triton.jit
def _fma_small_log2(v):
    # A positive integer with at most 24 bits is exactly representable in FP32.
    # Setting the low bit also defines the answer for zero as zero.
    bits = (v | 1).to(tl.float32).to(tl.int32, bitcast=True)
    return (bits >> 23) - 127


@triton.jit
def _fma_unpack(raw):
    """Finite magnitude = mantissa / 2**23 * 2**exp, including subnormals."""
    exponent = (raw >> 23) & 255
    frac = raw & 0x7FFFFF
    shift = 23 - _fma_small_log2(frac)
    mant = tl.where(exponent == 0, frac << shift, frac | 0x800000)
    exp = tl.where(exponent == 0, -126 - shift, exponent - 127)
    return mant, exp


@triton.jit
def _fma_shr(hi, lo, distance):
    """Shift the unsigned 54-bit significand right and report discarded bits."""
    a = tl.minimum(distance, 27)
    b = tl.minimum(tl.maximum(distance - 27, 0), 27)
    lomask = (1 << a) - 1
    himask = (1 << b) - 1
    lost = ((lo & lomask) != 0) | ((hi & himask) != 0)
    out_lo = tl.where(distance < 27, ((hi & lomask) << (27 - a)) | (lo >> a), hi >> b)
    out_hi = tl.where(distance < 27, hi >> a, 0)
    return out_hi, out_lo, lost


@triton.jit
def _ascend_fma(x, a, c):
    """Return RN32(x*a+c), with x widened from FP16 or BF16 and a,c FP32."""
    xb = x.to(tl.int32, bitcast=True)
    ab = a.to(tl.int32, bitcast=True)
    cb = c.to(tl.int32, bitcast=True)
    xm, xe = _fma_unpack(xb)
    am, ae = _fma_unpack(ab)
    cm, ce = _fma_unpack(cb)
    mp = xm >> 13
    # The product of one 11-bit and one 24-bit significand is at most 35 bits.
    # Split the 24-bit coefficient at bit 15, then carry into base 2**27.
    v = mp * (am >> 15)
    low = mp * (am & 32767) + ((v & 4095) << 15)
    ph = (v >> 12) + (low >> 27)
    pl = low & 0x7FFFFFF
    # Normalized product has its leading bit at 33 or 34. Move it to bit 52,
    # leaving one bit of headroom for a same-sign sum.
    carry = (ph >= 128).to(tl.int32)
    n = 19 - carry
    ph = (ph << n) | (pl >> (27 - n))
    pl = (pl & ((1 << (27 - n)) - 1)) << n
    pe = xe + ae + carry
    ch, cl = cm << 2, cm * 0
    pzero = (xm == 0) | (am == 0)
    e = tl.where(cm == 0, pe, tl.where(pzero, ce, tl.maximum(pe, ce)))
    ph, pl, psticky = _fma_shr(ph, pl, tl.maximum(e - pe, 0))
    ch, cl, csticky = _fma_shr(ch, cl, tl.maximum(e - ce, 0))
    pl = pl | psticky.to(tl.int32)
    cl = cl | csticky.to(tl.int32)

    psign = (xb < 0) ^ (ab < 0)
    csign = cb < 0
    same = psign == csign
    p_larger = (ph > ch) | ((ph == ch) & (pl >= cl))
    diff_lo = tl.where(p_larger, pl - cl, cl - pl)
    diff_hi = tl.where(p_larger, ph - ch, ch - ph) - (diff_lo < 0).to(tl.int32)
    sum_lo = pl + cl
    hi = tl.where(same, ph + ch + (sum_lo >> 27), diff_hi)
    lo = tl.where(same, sum_lo, diff_lo) & 0x7FFFFFF
    sign = tl.where(same | p_larger, psign, csign)
    # Opposite signed zeros or exact nonzero cancellation round to +0.
    sign = sign & (same | ((hi | lo) != 0))

    limb = tl.where(hi != 0, hi, lo)
    upper = limb >> 16
    small = tl.where(upper != 0, upper, limb)
    lead = (
        _fma_small_log2(small) + tl.where(upper != 0, 16, 0) + tl.where(hi != 0, 27, 0)
    )
    # Normal significands retain 24 bits; subnormals have an LSB of 2**-149.
    cut = tl.maximum(lead - 23, -97 - e)
    _, rounded, sticky = _fma_shr(hi, lo, tl.maximum(cut - 1, 0))
    increment = ((rounded & 1) != 0) & (sticky | ((rounded & 2) != 0))
    sig = tl.where(
        cut > 0,
        (rounded >> 1) + increment.to(tl.int32),
        lo << tl.minimum(tl.maximum(-cut, 0), 23),
    )
    rexp = e + lead - 52
    exp_field = tl.minimum(tl.maximum(rexp, -126), 127) + 126
    magnitude = tl.where(rexp > 127, 0x7F800000, (exp_field << 23) + sig)
    magnitude = tl.where((hi | lo) == 0, 0, magnitude)

    ax = xb & 0x7FFFFFFF
    aa = ab & 0x7FFFFFFF
    ac = cb & 0x7FFFFFFF
    pinf = (ax == 0x7F800000) | (aa == 0x7F800000)
    cinf = ac == 0x7F800000
    magnitude = tl.where(pinf | cinf, 0x7F800000, magnitude)
    sign = tl.where(pinf, psign, tl.where(cinf, csign, sign))
    result = magnitude | (sign.to(tl.int32) << 31)
    nan = (ax > 0x7F800000) | (aa > 0x7F800000) | (ac > 0x7F800000)
    invalid = (pinf & pzero) | (pinf & cinf & ~same)
    result = tl.where(nan | invalid, 0x7FC00000, result)
    return result.to(tl.float32, bitcast=True)


_rope_fma = _ascend_fma if runtime.device.vendor_name == "ascend" else tl.fma


@triton.jit
def _rotate_pair(x, y, cos, sin, CACHE: tl.constexpr):
    # Negate by sign bit so +0 becomes -0 on every backend.
    ys_bits = (y * sin).to(tl.int32, bitcast=True)
    neg_ys = (ys_bits ^ (-2147483648)).to(tl.float32, bitcast=True)
    first = _rope_fma(x, cos, neg_ys)
    # The CUDA query and cache writers contract different products in lane 1.
    if CACHE:
        second = _rope_fma(y, cos, x * sin)
    else:
        second = _rope_fma(x, sin, y * cos)
    return first, second


@triton.jit
def _ascend_pack_tail_bytes(src, first, second):
    first_bits = (
        first.to(src.dtype.element_ty).to(tl.uint16, bitcast=True).to(tl.uint32)
    )
    second_bits = (
        second.to(src.dtype.element_ty).to(tl.uint16, bitcast=True).to(tl.uint32)
    )
    # Mask widened words explicitly. On 910B, high sign bits from the first
    # word can otherwise overwrite the second word during byte packing.
    packed = (first_bits & 0xFFFF) | ((second_bits & 0xFFFF) << 16)
    byte = tl.arange(0, 128)
    # This gather accepts FP32 storage. Both casts reinterpret the same bits.
    gathered = tl.gather(packed.to(tl.float32, bitcast=True), byte // 4, 0).to(
        tl.int32, bitcast=True
    )
    return ((gathered >> ((byte % 4) * 8)) & 255).to(tl.uint8)


@triton.jit
def _ascend_rotated_tail_bytes(src, pos, table):
    pair = tl.arange(0, 32)
    even = tl.load(src + pair * 2).to(tl.float32)
    odd = tl.load(src + pair * 2 + 1).to(tl.float32)
    cs = table + tl.load(pos) * 64
    cos = tl.load(cs + pair)
    sin = tl.load(cs + 32 + pair)
    first, second = _rotate_pair(even, odd, cos, sin, True)
    return _ascend_pack_tail_bytes(src, first, second)


@triton.jit
def _read_pe(
    src, pos, table, ROPE: tl.constexpr, HEAD_TILE: tl.constexpr, CACHE: tl.constexpr
):
    lanes = tl.arange(0, 64)
    values = tl.load(src + lanes[None, :])
    if ROPE:
        pair = tl.arange(0, 32)
        row = table + tl.load(pos) * 64
        cos = tl.load(row + pair)[None, :]
        sin = tl.load(row + 32 + pair)[None, :]
        even, odd = tl.split(tl.reshape(values.to(tl.float32), (HEAD_TILE, 32, 2)))
        first, second = _rotate_pair(even, odd, cos, sin, CACHE)
        rotated = tl.interleave(first, second)
        if _IS_HYGON and values.dtype == tl.float16:
            # gfx936 can fold the FMA and FP16 cast into v_fma_mix*_f16.
            # Materialize RN32 before the separate RN16 conversion.
            rotated = tl.inline_asm_elementwise(
                "v_mov_b32 $0, $1",
                constraints="=v,v",
                args=[rotated],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
        values = rotated.to(values.dtype)
    return values


@triton.jit
def _write_cache_prefix(latent, row):
    col = tl.arange(0, 512)
    groups = tl.reshape(tl.load(latent + col).to(tl.float32), (4, 128))
    amax = tl.max(tl.abs(groups), 1)
    raw_scale = tl.maximum(amax / 448.0, 1.0e-4)
    # Round each positive FP32 scale up to a power of two, including ties.
    bits = raw_scale.to(tl.uint32, bitcast=True)
    scale_bits = (bits + 0x7FFFFF) & 0x7F800000
    scale = scale_bits.to(tl.float32, bitcast=True)
    fp8 = _encode_cache_fp8(groups / scale[:, None])
    tl.store(row + col, tl.reshape(fp8, (512,)))

    # Page, row and storage offsets can all be odd. Emit little-endian
    # bytes for the four FP32 scales.
    shifts = tl.arange(0, 4) * 8
    scale_bytes = (scale_bits[:, None] >> shifts[None, :]).to(tl.uint8)
    tl.store(row + 512 + tl.arange(0, 16), tl.reshape(scale_bytes, (16,)))


@triton.jit
def _write_cache(
    KV,
    KPE,
    CACHE,
    SLOTS,
    POS,
    TABLE,
    token,
    KS0,
    CS0,
    CS1,
    SS0,
    PAGE: tl.constexpr,
    ROPE: tl.constexpr,
):
    slot = tl.load(SLOTS + token * SS0)
    if slot >= 0:
        row = CACHE + slot // PAGE * CS0 + slot % PAGE * CS1
        _write_cache_prefix(KV + token * 512, row)
        if _IS_ASCEND and ROPE:
            tail = _ascend_rotated_tail_bytes(KPE + token * KS0, POS, TABLE)
        else:
            src = KPE + token * KS0 + tl.arange(0, 1)[:, None]
            pe = tl.reshape(_read_pe(src, POS, TABLE, ROPE, 1, True), (64,))
            words = pe.to(tl.uint16, bitcast=True).to(tl.uint32)
            tail = tl.interleave(words.to(tl.uint8), (words >> 8).to(tl.uint8))
        tl.store(row + 528 + tl.arange(0, 128), tail)


@triton.jit
def _prefill_kernel(
    Q,
    K_NOPE,
    K_PE,
    LATENT,
    OUT,
    CACHE,
    SLOTS,
    POS,
    TABLE,
    Q_S0: tl.constexpr,
    Q_S1: tl.constexpr,
    KN_S0: tl.constexpr,
    KN_S1: tl.constexpr,
    PE_S0: tl.constexpr,
    CACHE_S0: tl.constexpr,
    CACHE_S1: tl.constexpr,
    SLOT_S0: tl.constexpr,
    POS_S0: tl.constexpr,
    HEADS: tl.constexpr,
    PAGE: tl.constexpr,
    ROPE: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    HEAD_TILE: tl.constexpr,
    work_base,
):
    # A separate program owns the four quantization groups for each token.
    # This also covers a token with zero heads, leaving only its cache item.
    groups: tl.constexpr = HEADS // HEAD_TILE
    item = tl.program_id(0).to(tl.int64) + work_base
    token = item // (groups + 1)
    group = item % (groups + 1)
    pos = POS + token * POS_S0
    if group == groups:
        if HAS_CACHE:
            _write_cache(
                LATENT,
                K_PE,
                CACHE,
                SLOTS,
                pos,
                TABLE,
                token,
                PE_S0,
                CACHE_S0,
                CACHE_S1,
                SLOT_S0,
                PAGE,
                ROPE,
            )
    else:
        head = group * HEAD_TILE + tl.arange(0, HEAD_TILE)
        col = tl.arange(0, 128)[None, :]
        src = K_NOPE + token * KN_S0 + head[:, None] * KN_S1
        dst = OUT + (token * HEADS + head[:, None]) * 192
        tl.store(dst + col, tl.load(src + col))

        pe_src = K_PE + token * PE_S0 + tl.arange(0, 1)[:, None]
        key_tail = _read_pe(pe_src, pos, TABLE, ROPE, 1, False)
        tail_col = 128 + tl.arange(0, 64)[None, :]
        tl.store(dst + tail_col, key_tail)
        if ROPE:
            query_tail = Q + token * Q_S0 + head[:, None] * Q_S1 + 128
            rotated = _read_pe(query_tail, pos, TABLE, True, HEAD_TILE, False)
            tl.store(query_tail + tl.arange(0, 64)[None, :], rotated)


# A tuning trial would write mapped slots repeatedly, and rotate Q on every
# replay. Select the measured launch configuration once, including cold calls.
_fixed_kernel = libentry()(_prefill_kernel)
_launch_options = {}
if runtime.device.vendor_name == "nvidia":
    (_config,) = runtime.get_tuned_config("fused_kimi_k3_mla_key_concat_ds_mla_insert")
    _launch_options = {"num_warps": _config.num_warps, "num_stages": _config.num_stages}


# Reuse the loaded module on H20 between calls with matching metadata. Only
# the Triton 3.6 NVIDIA launcher and Torch 2.13 tensor-guard ABI were tested.
_h20_plans = {}
_guard_factory = None
if (
    runtime.device.vendor_name == "nvidia"
    and triton.__version__.split(".")[:2] == ["3", "6"]
    and torch.__version__.split(".")[:2] == ["2", "13"]
):
    _guard_factory = getattr(
        getattr(getattr(torch._C, "_dynamo", None), "guards", None),
        "TensorGuards",
        None,
    )
    _hooks = getattr(getattr(triton, "knobs", None), "runtime", None)
    _device = getattr(torch._C, "_cuda_getDevice", None)
    _stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)


class _H20Launch(NamedTuple):
    check: object
    device: int
    alignment: int
    tensor_types: tuple
    module: object  # Owns the function for as long as the plan is cached.
    launch: object
    prefix: tuple
    scalars: tuple


def _hooks_active():
    return getattr(
        getattr(_hooks, "launch_enter_hook", None), "calls", True
    ) or getattr(getattr(_hooks, "launch_exit_hook", None), "calls", True)


def _record_h20(compiled, programs, tensors, scalars, page, rope):
    if (
        not (_device and _stream)
        or _hooks_active()
        or any(
            type(t) is not torch.Tensor or t.data_ptr() % _fixed_kernel.divisibility
            for t in tensors
        )
        or "H20" not in torch.cuda.get_device_name(tensors[0].device)
    ):
        return
    run = compiled.run
    if not (
        type(run).__name__ == "CudaLauncher"
        and type(run).__module__ == "triton.backends.nvidia.driver"
        and getattr(run, "global_scratch_size", -1) == 0
        and getattr(run, "profile_scratch_size", -1) == 0
        and not getattr(run, "launch_cooperative_grid", True)
        and not getattr(run, "launch_pdl", True)
        and getattr(getattr(run, "launch", None), "__module__", "")
        == "__triton_launcher"
    ):
        return
    try:
        check = _guard_factory(
            *tensors, dynamic_dims_sizes=None, dynamic_dims_strides=None
        ).check
        # Ask this driver build for its launch flags and zero-scratch arguments.
        # Only a private copy sees the recorder; no kernel is enqueued here.
        probe = copy(run)
        if probe is run or not check(*tensors):
            return
        seen = []
        probe.launch = lambda *args: seen.append(args)
        probe(
            programs,
            1,
            1,
            None,
            compiled.function,
            compiled.packed_metadata,
            None,
            None,
            None,
        )
    except (AttributeError, TypeError, RuntimeError):
        return
    if len(seen) != 1 or seen[0][:4] != (programs, 1, 1, None):
        return
    if len(_h20_plans) >= 64:
        _h20_plans.clear()
    _h20_plans[tensors[0].numel(), page, rope] = _H20Launch(
        check,
        tensors[0].device.index,
        _fixed_kernel.divisibility,
        tuple(map(type, tensors)),
        compiled,
        partial(run.launch, programs, 1, 1),
        seen[0][4:],
        scalars,
    )


def _launch_h20(tensors, page, rope):
    plan = _h20_plans.get((tensors[0].numel(), page, rope))
    if (
        plan is None
        or tuple(map(type, tensors)) != plan.tensor_types
        or not plan.check(*tensors)
        or _device() != plan.device
        or _hooks_active()
    ):
        return False
    # Storage offsets and replacement buffers can change pointer alignment
    # even if dtype, shape, stride and device still match the compiled kernel.
    pointers = tuple(map(torch.Tensor.data_ptr, tensors))
    if reduce(or_, pointers) % plan.alignment:
        return False
    if not rope:
        pointers += (pointers[6], pointers[2])
    plan.launch(_stream(plan.device), *plan.prefix, *pointers, *plan.scalars)
    return True


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
    """Write full keys, optionally rotate Q in place, and insert 656-byte cache rows.

    Nonnegative slots must be unique and in bounds. Positions must index the
    FP32 table. Writable tensor elements must be disjoint from other arguments.
    The model uses BF16. The low-level FP16 path stores its cache tail in FP16.
    """
    logger.debug("GEMS FUSED_KIMI_K3_MLA_KEY_CONCAT_DS_MLA_INSERT")
    rope = position_ids is not None
    if rope != (cos_sin_cache is not None):
        raise ValueError("position_ids and cos_sin_cache must be provided together")
    payload = (q, k_nope, k_pe, kv_c_normed, k_out)
    tensors = (*payload, k_cache, slot_mapping)
    if rope:
        tensors += (position_ids, cos_sin_cache)
    if (
        _h20_plans
        and type(q) is torch.Tensor
        and type(cache_block_size) is int
        and _launch_h20(tensors, cache_block_size, rope)
    ):
        return
    _prefill(tensors, cache_block_size, rope, _record_h20 if _guard_factory else None)


def _prefill(
    tensors,
    cache_block_size,
    rope,
    record_launch=None,
    select_config=None,
    kernel=_fixed_kernel,
):
    payload = tensors[:5]
    q, k_nope, k_pe, kv_c_normed, k_out = payload
    k_cache, slot_mapping = tensors[5:7]
    position_ids, cos_sin_cache = tensors[7:] if rope else (None, None)
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("expected tensor arguments")
    device = q.device
    if any(t.layout != torch.strided or t.device != device for t in tensors):
        raise ValueError("all tensors must be strided and on the same device")
    if q.dtype not in (torch.bfloat16, torch.float16) or any(
        t.dtype != q.dtype for t in payload[1:]
    ):
        raise TypeError(
            "q, k_nope, k_pe, kv_c_normed and k_out must share BF16 or FP16 dtype"
        )
    if k_cache.dtype != torch.uint8 or slot_mapping.dtype != torch.int64:
        raise TypeError("k_cache must be uint8 and slot_mapping must be int64")
    if type(cache_block_size) is not int or cache_block_size <= 0:
        raise ValueError("cache_block_size must be a positive integer")
    if q.ndim != 3 or q.size(2) != 192:
        raise ValueError("q must have shape [T, H, 192]")
    tokens, heads, _ = q.shape
    if k_nope.shape != (tokens, heads, 128) or k_pe.shape != (tokens, 64):
        raise ValueError("expected k_nope [T, H, 128] and k_pe [T, 64]")
    if kv_c_normed.shape != (tokens, 512) or k_out.shape != q.shape:
        raise ValueError("expected kv_c_normed [T, 512] and k_out [T, H, 192]")
    if k_cache.ndim != 3 or k_cache.shape[1:] != (cache_block_size, 656):
        raise ValueError("expected k_cache [num_blocks, cache_block_size, 656]")
    if any(t.stride(-1) != 1 for t in (q, k_nope, k_pe, k_cache)):
        raise ValueError(
            "q, k_nope, k_pe and k_cache must have unit last-dimension stride"
        )
    if not kv_c_normed.is_contiguous() or not k_out.is_contiguous():
        raise ValueError("kv_c_normed and k_out must be contiguous")
    if slot_mapping.shape != (tokens,):
        raise ValueError("slot_mapping must have shape [T]")
    if rope:
        if position_ids.dtype != torch.int64 or cos_sin_cache.dtype != torch.float32:
            raise TypeError("RoPE requires int64 position_ids and a float32 table")
        if position_ids.shape != (tokens,):
            raise ValueError("position_ids must have shape [T]")
        if (
            cos_sin_cache.ndim != 2
            or cos_sin_cache.size(1) != 64
            or not cos_sin_cache.is_contiguous()
            or (tokens and cos_sin_cache.size(0) == 0)
        ):
            raise ValueError("expected contiguous cos_sin_cache [max_position, 64]")
    if device.type == "cpu" or device.type != runtime.device.name:
        raise NotImplementedError("an active accelerator device is required")
    if tokens == 0:
        return

    vendor = runtime.device.vendor_name
    head_tile = 1
    if heads >= 4 and heads % 4 == 0:
        head_tile = 8 if heads >= 96 and heads % 8 == 0 else 4
    options = _launch_options
    if select_config is not None:
        head_tile, options = select_config(q, head_tile)
    total = tokens * (heads // head_tile + 1)
    grid_limit = 65535 if vendor == "ascend" else 2**31 - 1
    has_cache = k_cache.size(0) != 0
    pointers = (
        q if heads else k_pe,
        k_nope if heads else k_pe,
        k_pe,
        kv_c_normed,
        k_out if heads else k_pe,
        k_cache if has_cache else k_pe,
        slot_mapping,
        position_ids if rope else slot_mapping,
        cos_sin_cache if rope else k_pe,
    )
    constants = (
        q.stride(0),
        q.stride(1),
        k_nope.stride(0),
        k_nope.stride(1),
        k_pe.stride(0),
        k_cache.stride(0),
        k_cache.stride(1),
        slot_mapping.stride(0),
        position_ids.stride(0) if rope else 1,
        heads,
        cache_block_size,
        rope,
        has_cache,
        head_tile,
    )
    with torch_device_fn.device(device):
        for base in range(0, total, grid_limit):
            programs = min(total - base, grid_limit)
            compiled, _ = kernel[(programs,)](*pointers, *constants, base, **options)
            if (
                record_launch is not None
                and heads
                and has_cache
                and total <= grid_limit
            ):
                record_launch(
                    compiled,
                    programs,
                    tensors,
                    (*constants, base),
                    cache_block_size,
                    rope,
                )


__all__ = ["fused_kimi_k3_mla_key_concat_ds_mla_insert"]
