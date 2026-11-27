# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors

import logging
from copy import copy
from functools import partial, reduce
from operator import or_
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.fp8_storage import encode_e4m3fn
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)

# Only enable cached launchers for the backend ABIs with verified stream,
# pointer-specialization and launch-hook behavior.
_vendor = runtime.device.vendor_name
_triton_version = triton.__version__.split(".")[:2]
_npu_launch = _vendor == "ascend" and _triton_version == ["3", "5"]
_hygon_launch = _vendor == "hygon"
_hcu_within_2gb = None
_launch_plans = {}
_tensor_guards = None
if _npu_launch or (_vendor in ("nvidia", "hygon") and _triton_version == ["3", "6"]):
    from triton import knobs

    if _hygon_launch and hasattr(triton.backends, "hcu"):
        from triton.backends.hcu.compiler import HIPBackend

        _hcu_within_2gb = getattr(HIPBackend, "is_within_2gb", None)

    _guards = getattr(getattr(torch._C, "_dynamo", None), "guards", None)
    _tensor_guards = getattr(_guards, "TensorGuards", None)
    _current_device = (
        torch_device_fn.current_device
        if _npu_launch
        else getattr(torch._C, "_cuda_getDevice", None)
    )
    _current_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)


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
def _ascend_rotated_tail_bytes(src, pos, table):
    pair = tl.arange(0, 32)
    even = tl.load(src + pair * 2).to(tl.float32)
    odd = tl.load(src + pair * 2 + 1).to(tl.float32)
    cs = table + tl.load(pos) * 64
    cos = tl.load(cs + pair)
    sin = tl.load(cs + 32 + pair)
    first, second = _rotate_pair(even, odd, cos, sin, True)
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
def _write_query(
    QL,
    QPE,
    OUT,
    POS,
    TABLE,
    token,
    group,
    QS0,
    QS1,
    PS0,
    PS1,
    HEADS: tl.constexpr,
    ROPE: tl.constexpr,
    HEAD_TILE: tl.constexpr,
):
    heads = group.to(tl.int64) * HEAD_TILE + tl.arange(0, HEAD_TILE)
    out = OUT + (token * HEADS + heads[:, None]) * 576
    latent = QL + token * QS0 + heads[:, None] * QS1
    cols = tl.arange(0, 512)[None, :]
    tl.store(out + cols, tl.load(latent + cols))
    src = QPE + token * PS0 + heads[:, None] * PS1
    pe = _read_pe(src, POS, TABLE, ROPE, HEAD_TILE, False)
    tl.store(out + 512 + tl.arange(0, 64)[None, :], pe)


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
        col = tl.arange(0, 512)
        groups = tl.reshape(tl.load(KV + token * 512 + col).to(tl.float32), (4, 128))
        amax = tl.max(tl.abs(groups), 1)
        raw_scale = tl.maximum(amax / 448.0, 1.0e-4)
        # Round each positive FP32 scale up to a power of two, including ties.
        bits = raw_scale.to(tl.uint32, bitcast=True)
        scale_bits = (bits + 0x7FFFFF) & 0x7F800000
        scale = scale_bits.to(tl.float32, bitcast=True)
        fp8 = _encode_cache_fp8(groups / scale[:, None])
        tl.store(row + col, tl.reshape(fp8, (512,)))

        # Page, row and storage offsets can all be odd. Emit little-endian
        # bytes for the four FP32 scales and the 64-element BF16/FP16 tail.
        shifts = tl.arange(0, 4) * 8
        scale_bytes = (scale_bits[:, None] >> shifts[None, :]).to(tl.uint8)
        tl.store(row + 512 + tl.arange(0, 16), tl.reshape(scale_bytes, (16,)))
        if _IS_ASCEND and ROPE:
            tail = _ascend_rotated_tail_bytes(KPE + token * KS0, POS, TABLE)
        else:
            src = KPE + token * KS0 + tl.arange(0, 1)[:, None]
            pe = tl.reshape(_read_pe(src, POS, TABLE, ROPE, 1, True), (64,))
            words = pe.to(tl.uint16, bitcast=True).to(tl.uint32)
            tail = tl.interleave(words.to(tl.uint8), (words >> 8).to(tl.uint8))
        tl.store(row + 528 + tl.arange(0, 128), tail)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("fused_kimi_k3_mla_decode_q_concat_ds_mla_insert"),
    key=["HEADS", "ROPE", "HEAD_TILE"],
)
@triton.jit
def _decode_kernel(
    QL,
    QPE,
    KV,
    KPE,
    OUT,
    CACHE,
    SLOTS,
    POS,
    TABLE,
    QS0,
    QS1,
    PS0,
    PS1,
    KS0,
    CS0,
    CS1,
    SS0,
    IDS0,
    HEADS: tl.constexpr,
    PAGE: tl.constexpr,
    ROPE: tl.constexpr,
    HEAD_TILE: tl.constexpr,
    work_base,
):
    # Every token owns a cache work item, including when it has no query heads.
    groups: tl.constexpr = HEADS // HEAD_TILE
    item = tl.program_id(0).to(tl.int64) + work_base
    token = item // (groups + 1)
    group = item % (groups + 1)
    pos = POS + token * IDS0
    if group == groups:
        _write_cache(
            KV, KPE, CACHE, SLOTS, pos, TABLE, token, KS0, CS0, CS1, SS0, PAGE, ROPE
        )
    else:
        _write_query(
            QL,
            QPE,
            OUT,
            pos,
            TABLE,
            token,
            group,
            QS0,
            QS1,
            PS0,
            PS1,
            HEADS,
            ROPE,
            HEAD_TILE,
        )


# Grouped launches specialize query strides. The general kernel also handles
# small workloads and layouts with dynamic strides.
_decode_tile = _decode_kernel.jit_function


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("fused_kimi_k3_mla_decode_q_concat_ds_mla_insert"),
    key=["HEADS", "ROPE", "HEAD_TILE"],
)
@triton.jit
def _grouped_decode_kernel(
    QL,
    QPE,
    KV,
    KPE,
    OUT,
    CACHE,
    SLOTS,
    POS,
    TABLE,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    PS0: tl.constexpr,
    PS1: tl.constexpr,
    KS0,
    CS0,
    CS1,
    SS0,
    IDS0,
    HEADS: tl.constexpr,
    PAGE: tl.constexpr,
    ROPE: tl.constexpr,
    HEAD_TILE: tl.constexpr,
    work_base,
):
    _decode_tile(
        QL,
        QPE,
        KV,
        KPE,
        OUT,
        CACHE,
        SLOTS,
        POS,
        TABLE,
        QS0,
        QS1,
        PS0,
        PS1,
        KS0,
        CS0,
        CS1,
        SS0,
        IDS0,
        HEADS,
        PAGE,
        ROPE,
        HEAD_TILE,
        work_base,
    )


class _CachedLaunch(NamedTuple):
    guard: object
    device: int
    kernel: object  # Owns the compiled module until this entry is evicted.
    run: object
    metadata: tuple
    scalars: tuple


def _hcu_pointer_ranges(tensors):
    if knobs.hcu.use_buffer_ops:
        return tuple(map(_hcu_within_2gb, tensors))
    return ()


def _cuda_launcher(kernel, programs):
    run = kernel.run
    launch_metadata = (kernel.function, kernel.packed_metadata, None, None, None)
    if (
        type(run).__module__ == "triton.backends.nvidia.driver"
        and type(run).__name__ == "CudaLauncher"
        and getattr(run, "global_scratch_size", -1) == 0
        and getattr(run, "profile_scratch_size", -1) == 0
        and not getattr(run, "launch_cooperative_grid", True)
        and not getattr(run, "launch_pdl", True)
        and getattr(getattr(run, "launch", None), "__module__", None)
        == "__triton_launcher"
    ):
        # For a zero-scratch CUDA kernel this wrapper only forwards launch
        # flags, scratch placeholders and metadata to the C entry. Capture the
        # driver's actual prefix with a copy so concurrent launchers and hooks
        # remain untouched. Never retain tensors, addresses or the stream.
        recorder = []
        try:
            probe = copy(run)
            if probe is not run:
                probe.launch = lambda *args: recorder.append(args)
                placeholder = object()
                probe(programs, 1, 1, placeholder, *launch_metadata)
        except (AttributeError, TypeError):
            pass
        else:
            if len(recorder) == 1 and recorder[0][:4] == (programs, 1, 1, placeholder):
                launch_metadata = recorder[0][4:]
                run = run.launch
    return partial(run, programs, 1, 1), launch_metadata


def _remember_launch(kernel, programs, tensors, scalars, page, rope):
    if not (_current_device and (_npu_launch or _current_stream)) or any(
        type(t) is not torch.Tensor or t.data_ptr() % _decode_kernel.divisibility
        for t in tensors
    ):
        return
    try:
        guard = _tensor_guards(
            *tensors, dynamic_dims_sizes=None, dynamic_dims_strides=None
        ).check
    except TypeError:
        return
    if not guard(*tensors):
        return
    if len(_launch_plans) >= 128:
        _launch_plans.clear()
    if _npu_launch:
        run, metadata = kernel[(programs, 1, 1)], ()
    elif _hygon_launch:
        if _hcu_within_2gb is None:
            return
        run = kernel[(programs, 1, 1)]
        metadata = _hcu_pointer_ranges(tensors)
    else:
        run, metadata = _cuda_launcher(kernel, programs)
    first = tensors[0]
    _launch_plans[first.numel(), page, rope] = _CachedLaunch(
        guard,
        first.device.index,
        kernel,
        run,
        metadata,
        scalars,
    )


def _launch_if_cached(tensors, page, rope):
    plan = _launch_plans.get((tensors[0].numel(), page, rope))
    if plan is None or not plan.guard(*tensors) or _current_device() != plan.device:
        return False
    # Active hooks need Triton's regular launcher to construct launch metadata.
    if getattr(knobs.runtime.launch_enter_hook, "calls", True) or getattr(
        knobs.runtime.launch_exit_hook, "calls", True
    ):
        return False
    # TensorGuards matches sizes, strides, dtype, type and device. Storage
    # offsets and replacement buffers require checking fresh pointer alignment.
    addresses = tuple(map(torch.Tensor.data_ptr, tensors))
    if reduce(or_, addresses) % _decode_kernel.divisibility:
        return False
    if _npu_launch:
        # The compiled NPU launcher resolves the calling thread's current stream.
        if not rope:
            tensors = (*tensors, tensors[6], tensors[3])
        plan.run(*tensors, *plan.scalars)
        return True
    if _hygon_launch:
        # HCU specializes on allocation size as well as pointer alignment.
        if _hcu_pointer_ranges(tensors) != plan.metadata:
            return False
        if not rope:
            tensors = (*tensors, tensors[6], tensors[3])
        plan.run(*tensors, *plan.scalars, stream=_current_stream(plan.device))
        return True
    if not rope:
        addresses = (*addresses, addresses[6], addresses[3])
    plan.run(_current_stream(plan.device), *plan.metadata, *addresses, *plan.scalars)
    return True


def fused_kimi_k3_mla_decode_q_concat_ds_mla_insert(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    mqa_q: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    cache_block_size: int,
    position_ids: torch.Tensor | None = None,
    cos_sin_cache: torch.Tensor | None = None,
) -> None:
    """Write the decode query and pack one 656-byte cache row per valid slot.

    The model uses BF16. The low-level FP16 variant stores its PE tail in FP16.
    Nonnegative slots must be unique and in range, positions must index the
    RoPE table, and outputs must have disjoint storage from all other tensors.
    """
    logger.debug("GEMS FUSED_KIMI_K3_MLA_DECODE_Q_CONCAT_DS_MLA_INSERT")
    if (position_ids is None) != (cos_sin_cache is None):
        raise ValueError("position_ids and cos_sin_cache must be provided together")
    rope = position_ids is not None
    tensors = (ql_nope, q_pe, kv_c_normed, k_pe, mqa_q, k_cache, slot_mapping)
    if rope:
        tensors = (*tensors, position_ids, cos_sin_cache)
    if (
        _launch_plans
        and type(ql_nope) is torch.Tensor
        and type(cache_block_size) is int
    ):
        if _launch_if_cached(tensors, cache_block_size, rope):
            return
    payload = tensors[:5]
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("expected tensor inputs and outputs")
    device, dtype = ql_nope.device, ql_nope.dtype
    if any(t.layout != torch.strided or t.device != device for t in tensors):
        raise ValueError("all tensors must be strided and on the same device")
    if dtype not in (torch.bfloat16, torch.float16) or any(
        t.dtype != dtype for t in payload[1:]
    ):
        raise TypeError(
            "query and latent payload tensors must share dtype BF16 or FP16"
        )
    if k_cache.dtype != torch.uint8 or slot_mapping.dtype != torch.int64:
        raise TypeError("k_cache must be uint8 and slot_mapping must be int64")
    if type(cache_block_size) is not int or cache_block_size <= 0:
        raise ValueError("cache_block_size must be a positive integer")
    if ql_nope.ndim != 3 or ql_nope.shape[2] != 512:
        raise ValueError("ql_nope must have shape [B,H,512]")
    batch, heads, _ = ql_nope.shape
    if (
        q_pe.shape != (batch, heads, 64)
        or kv_c_normed.shape != (batch, 512)
        or k_pe.shape != (batch, 64)
        or mqa_q.shape != (batch, heads, 576)
        or slot_mapping.shape != (batch,)
    ):
        raise ValueError("inconsistent decode tensor shapes")
    if k_cache.ndim != 3 or k_cache.shape[1:] != (cache_block_size, 656):
        raise ValueError("expected k_cache [num_blocks,cache_block_size,656]")
    if any(t.stride(-1) != 1 for t in (*payload, k_cache)):
        raise ValueError("payload and cache last-dimension strides must be one")
    if not kv_c_normed.is_contiguous() or not mqa_q.is_contiguous():
        raise ValueError("kv_c_normed and mqa_q must be contiguous")
    if rope:
        if position_ids.dtype != torch.int64 or cos_sin_cache.dtype != torch.float32:
            raise TypeError("RoPE requires int64 position_ids and a float32 table")
        if position_ids.shape != (batch,):
            raise ValueError("position_ids must have shape [B]")
        if (
            cos_sin_cache.ndim != 2
            or cos_sin_cache.shape[1] != 64
            or not cos_sin_cache.is_contiguous()
            or (batch > 0 and cos_sin_cache.shape[0] == 0)
        ):
            raise ValueError("expected contiguous cos_sin_cache [max_position,64]")
    if device.type == "cpu" or device.type != runtime.device.name:
        raise NotImplementedError("an active accelerator device is required")
    if batch == 0:
        return
    vendor = runtime.device.vendor_name
    query_strides = (*ql_nope.stride()[:2], *q_pe.stride()[:2])
    head_tile = 1
    tile, minimum_heads = (8, 512) if vendor == "hygon" else (4, 4096)
    if (
        vendor in ("nvidia", "hygon")
        and heads % tile == 0
        and batch * heads >= minimum_heads
        and all(stride % 8 == 0 for stride in query_strides)
    ):
        head_tile = tile
    # Share the phase row and vector FMA across heads. Preserve enough independent
    # programs on small batches to occupy the 910B vector cores.
    if vendor == "ascend" and rope and heads % 4 == 0 and batch * heads >= 256:
        head_tile = 8 if heads % 8 == 0 and batch * heads > 2048 else 4
    total = batch * (heads // head_tile + 1)
    # Ascend limits the number of programs in a launch to 65535.
    limit = 65535 if vendor == "ascend" else 2**31 - 1
    pointers = (
        ql_nope,
        q_pe,
        kv_c_normed,
        k_pe,
        mqa_q,
        k_cache,
        slot_mapping,
        position_ids if rope else slot_mapping,
        cos_sin_cache if rope else k_pe,
    )
    metadata = (
        *query_strides,
        k_pe.stride(0),
        k_cache.stride(0),
        k_cache.stride(1),
        slot_mapping.stride(0),
        position_ids.stride(0) if rope else 1,
        heads,
        cache_block_size,
        rope,
        head_tile,
    )
    launch = _grouped_decode_kernel if head_tile > 1 else _decode_kernel
    with torch_device_fn.device(device):
        for base in range(0, total, limit):
            programs = min(total - base, limit)
            scalars = (*metadata, base)
            kernel, _ = launch[(programs,)](*pointers, *scalars)
            if _tensor_guards and total <= limit:
                _remember_launch(
                    kernel, programs, tensors, scalars, cache_block_size, rope
                )


__all__ = ["fused_kimi_k3_mla_decode_q_concat_ds_mla_insert"]
