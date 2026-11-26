# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

import logging
from threading import Lock

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.triton_reshape_and_cache_flash import _encode_e4m3
from flaggems_vllm.utils import libentry, libtuner
from flaggems_vllm.utils.triton_version_utils import _triton_version_at_least

logger = logging.getLogger(__name__)

_INPUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_QUANT_RANGES = {
    torch.int8: (-128.0, 127.0),
    torch.float8_e4m3fn: (-448.0, 448.0),
    # The vLLM FNUZ quantization protocol leaves the outer ±240 codes unused.
    torch.float8_e4m3fnuz: (-224.0, 224.0),
}
_IS_NVIDIA = runtime.device.vendor_name == "nvidia"
_IS_ASCEND = runtime.device.vendor_name == "ascend"
_IS_MTHREADS = runtime.device.vendor_name == "mthreads"
_IS_METAX = runtime.device.vendor_name == "metax"
_REFINE_DIV = _IS_MTHREADS
# Bound the number of specializations retained by the launcher cache.
_LAUNCH_CACHE_LIMIT = 128
_launch_cache = {}
_launch_cache_lock = Lock()


@triton.jit
def _divide(numerator, denominator, REFINE: tl.constexpr, ASCEND: tl.constexpr):
    if ASCEND:
        # The Ascend vector compiler scalarizes one-element reductions and
        # cannot lower the resulting scalar precise_divf. Float division is
        # supported for the scale and reciprocal.
        return numerator / denominator
    quotient = tl.div_rn(numerator, denominator)
    if REFINE:
        # MUSA may lower div_rn to an approximate reciprocal. Recover the
        # residual before either the scale or a half-way code is rounded.
        residual = tl.fma(-quotient, denominator, numerator)
        quotient += tl.div_rn(residual, denominator)
    return quotient


@triton.jit
def _quantize_head(
    source,
    cache,
    scale_cache,
    dim,
    WIDTH: tl.constexpr,
    QMIN: tl.constexpr,
    QMAX: tl.constexpr,
    INT8: tl.constexpr,
    SOFTWARE_FP8: tl.constexpr,
    REFINE_DIV: tl.constexpr,
    ASCEND: tl.constexpr,
):
    mask = dim < WIDTH
    row = tl.load(source + dim, mask=mask, other=0).to(tl.float32)
    peak = tl.max(tl.abs(row), axis=0, keep_dims=ASCEND)
    scale = tl.maximum(_divide(peak, QMAX, REFINE_DIV, ASCEND), 1.0e-6)
    if ASCEND:
        tl.store(scale_cache + tl.arange(0, 1), scale)
    else:
        tl.store(scale_cache, scale)

    code = row * _divide(1.0, scale, REFINE_DIV, ASCEND)
    if INT8:
        code = tl.where(code >= 0, code + 0.5, code - 0.5)
    code = tl.clamp(code, QMIN, QMAX)
    if SOFTWARE_FP8:
        code = _encode_e4m3(code, QMAX == 224.0)
    tl.store(cache + dim, code, mask=mask)


@triton.jit
def _cache_per_token_head(
    key,
    value,
    key_cache,
    value_cache,
    k_scale_cache,
    v_scale_cache,
    slots,
    pages,
    K_STRIDES: tl.constexpr,
    V_STRIDES: tl.constexpr,
    KC_STRIDES: tl.constexpr,
    VC_STRIDES: tl.constexpr,
    KS_STRIDES: tl.constexpr,
    VS_STRIDES: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    K_TILE: tl.constexpr,
    V_TILE: tl.constexpr,
    COMMON_TILE: tl.constexpr,
    QMIN: tl.constexpr,
    QMAX: tl.constexpr,
    INT8: tl.constexpr,
    SOFTWARE_FP8: tl.constexpr,
    REFINE_DIV: tl.constexpr,
    ASCEND: tl.constexpr,
    TOKEN_BUCKET: tl.constexpr,
    HEADS: tl.constexpr,
    CONTIGUOUS_CACHE: tl.constexpr,
    INDEPENDENT_TILES: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    slot = tl.load(slots + token * SLOT_STRIDE).to(tl.int64)
    if (slot >= 0) & (slot < pages.to(tl.int64) * PAGE_SIZE):
        page, offset = slot // PAGE_SIZE, slot % PAGE_SIZE
        if INDEPENDENT_TILES:
            k_dims = tl.arange(0, K_TILE)
            v_dims = tl.arange(0, V_TILE)
        else:
            k_dims = tl.arange(0, COMMON_TILE)
            v_dims = k_dims

        _quantize_head(
            key + token * K_STRIDES[0] + head * K_STRIDES[1],
            key_cache
            + page * KC_STRIDES[0]
            + offset * KC_STRIDES[1]
            + head * KC_STRIDES[2],
            k_scale_cache
            + page * KS_STRIDES[0]
            + offset * KS_STRIDES[1]
            + head * KS_STRIDES[2],
            k_dims,
            DK,
            QMIN,
            QMAX,
            INT8,
            SOFTWARE_FP8,
            REFINE_DIV,
            ASCEND,
        )
        _quantize_head(
            value + token * V_STRIDES[0] + head * V_STRIDES[1],
            value_cache
            + page * VC_STRIDES[0]
            + offset * VC_STRIDES[1]
            + head * VC_STRIDES[2],
            v_scale_cache
            + page * VS_STRIDES[0]
            + offset * VS_STRIDES[1]
            + head * VS_STRIDES[2],
            v_dims,
            DV,
            QMIN,
            QMAX,
            INT8,
            SOFTWARE_FP8,
            REFINE_DIV,
            ASCEND,
        )


if _IS_NVIDIA:
    _tuned_cache_per_token_head = libentry()(
        libtuner(
            configs=runtime.get_tuned_config(
                "triton_reshape_and_cache_flash_per_token_head_quant"
            ),
            key=["DK", "DV", "TOKEN_BUCKET", "HEADS", "CONTIGUOUS_CACHE"],
            restore_value=[
                "key_cache",
                "value_cache",
                "k_scale_cache",
                "v_scale_cache",
            ],
        )(_cache_per_token_head)
    )

_CACHE_LAUNCH = runtime.device.vendor_name in {
    "nvidia",
    "ascend",
    "thead",
    "mthreads",
    "metax",
} and _triton_version_at_least(3, 3)


def triton_reshape_and_cache_flash_per_token_head_quant(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Quantize each K/V token-head independently and update all four caches.

    Data tensors have unit stride in the final dimension. Cache views must
    have non-overlapping destinations and valid slots must be unique.
    """
    logger.debug("GEMS TRITON_RESHAPE_AND_CACHE_FLASH_PER_TOKEN_HEAD_QUANT")
    device = key.device
    caches = (key_cache, value_cache, k_scale_cache, v_scale_cache)
    tensors = (key, value, *caches, slot_mapping)
    if device.type == "cpu" or any(t.device != device for t in tensors[1:]):
        raise ValueError("all tensors must be on the same accelerator device")

    cache_key = None
    if _CACHE_LAUNCH and device.type == runtime.device.name:
        # The cached binary carries the validated sizes and strides, including
        # Triton's pointer alignment specialization. Callers still supply all
        # seven tensors and each launch uses the caller's current stream.
        cache_key = (device,) + tuple(
            (t.shape, t.stride(), t.dtype, t.data_ptr() % 16 == 0) for t in tensors
        )
        cached = _launch_cache.get(cache_key)
        if cached is not None:
            runner, constants, byte_cache = cached
            if byte_cache:
                tensors = (
                    key,
                    value,
                    key_cache.view(torch.uint8),
                    value_cache.view(torch.uint8),
                    k_scale_cache,
                    v_scale_cache,
                    slot_mapping,
                )
            if runtime.torch_device_fn.current_device() == device.index:
                runner(*tensors, *constants)
            else:
                with runtime.torch_device_fn.device(device):
                    runner(*tensors, *constants)
            return

    if key.ndim != 3 or value.ndim != 3 or key.shape[:2] != value.shape[:2]:
        raise ValueError("key/value must be [T,H,D] with matching T and H")
    tokens, heads, dk = key.shape
    dv = value.shape[2]
    if heads <= 0 or min(dk, dv) <= 0:
        raise ValueError("H, Dk and Dv must be positive")
    if key.dtype not in _INPUT_DTYPES or value.dtype not in _INPUT_DTYPES:
        raise ValueError("key/value dtype must be FP16, BF16 or FP32")
    if slot_mapping.shape != (tokens,) or slot_mapping.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("slot_mapping must be int32/int64 with shape [T]")
    if key_cache.ndim != 4 or key_cache.shape[1] == 0:
        raise ValueError("key_cache must be [P,S,H,Dk] with positive S")
    pages, page_size = key_cache.shape[:2]
    cache_prefix = (pages, page_size, heads)
    if key_cache.shape != (*cache_prefix, dk) or value_cache.shape != (
        *cache_prefix,
        dv,
    ):
        raise ValueError(
            "data cache shapes must match P, S, H and their respective widths"
        )
    if any(t.stride(-1) != 1 for t in (key, value, key_cache, value_cache)):
        raise ValueError(
            "input and cache data must have unit stride in the final dimension"
        )
    cache_dtype = key_cache.dtype
    if cache_dtype not in _QUANT_RANGES or value_cache.dtype != cache_dtype:
        raise ValueError("data caches must have the same INT8 or E4M3 FP8 dtype")
    if any(t.shape != cache_prefix or t.dtype != torch.float32 for t in caches[2:]):
        raise ValueError("scale caches must be FP32 with shape [P,S,H]")
    if tokens == 0 or pages == 0:
        return

    qmin, qmax = _QUANT_RANGES[cache_dtype]
    int8 = cache_dtype == torch.int8
    # Use the existing integer E4M3 encoder only when the backend has no native
    # FP8 Triton conversion for this format.  MThreads and MetaX hardware
    # supports native fp8e4m3fn (confirmed by fp8-cast-probe).
    if int8:
        software_fp8 = False
    elif cache_dtype == torch.float8_e4m3fnuz:
        software_fp8 = True
    elif _IS_NVIDIA:
        software_fp8 = torch.cuda.get_device_capability(device) < (8, 9)
    else:
        software_fp8 = not (_IS_MTHREADS or _IS_METAX)
    if software_fp8:
        key_cache = key_cache.view(torch.uint8)
        value_cache = value_cache.view(torch.uint8)
    grid = (tokens, heads)
    constants = (
        pages,
        key.stride()[:2],
        value.stride()[:2],
        key_cache.stride()[:3],
        value_cache.stride()[:3],
        k_scale_cache.stride(),
        v_scale_cache.stride(),
        slot_mapping.stride(0),
        page_size,
        dk,
        dv,
        triton.next_power_of_2(dk),
        triton.next_power_of_2(dv),
        triton.next_power_of_2(max(dk, dv)),
        qmin,
        qmax,
        int8,
        software_fp8,
        _REFINE_DIV,
        _IS_ASCEND,
        triton.next_power_of_2(tokens),
        heads,
        key_cache.is_contiguous() and value_cache.is_contiguous(),
    )
    launch_args = (
        key,
        value,
        key_cache,
        value_cache,
        k_scale_cache,
        v_scale_cache,
        slot_mapping,
    )
    with runtime.torch_device_fn.device(device):
        if _IS_NVIDIA:
            compiled, meta = _tuned_cache_per_token_head[grid](
                *launch_args, *constants, enable_fp_fusion=False
            )
            independent_tiles = meta["INDEPENDENT_TILES"]
        else:
            independent_tiles = False
            compiled = _cache_per_token_head[grid](
                *launch_args,
                *constants,
                INDEPENDENT_TILES=independent_tiles,
                enable_fp_fusion=False,
            )
        if cache_key is not None and compiled is not None:
            entry = (
                compiled[(*grid, 1)],
                (*constants, independent_tiles),
                software_fp8,
            )
            with _launch_cache_lock:
                if len(_launch_cache) >= _LAUNCH_CACHE_LIMIT:
                    _launch_cache.pop(next(iter(_launch_cache)))
                _launch_cache[cache_key] = entry
