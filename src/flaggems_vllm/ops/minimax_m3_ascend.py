# SPDX-License-Identifier: Apache-2.0
"""Ascend-only MiniMax M3 operator contracts; no framework/model imports.

Positive slots/positions must be in bounds and write slots must be unique.
Those data-dependent preconditions belong to the caller and are not read back
to the host, preserving graph capture. Negative cache slots never write.
"""

import inspect
import math

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn


def _check_npu(*tensors):
    for tensor in tensors:
        if tensor is not None and tensor.device.type != "npu":
            raise NotImplementedError("MiniMax M3 Ascend operators require NPU tensors")
    devices = {tensor.device for tensor in tensors if tensor is not None}
    if len(devices) > 1:
        raise ValueError("all tensors must be on the same NPU device")


def _check_slots(slots, count):
    if slots is None or slots.dtype not in (torch.int32, torch.int64):
        raise ValueError("cache slots must be int32/int64")
    if slots.ndim != 1 or slots.numel() < count or not slots.is_contiguous():
        raise ValueError("cache slots must be contiguous and cover all tokens")


@triton.jit
def _norm_rope_kernel(
    QKV,
    QW,
    KW,
    IQW,
    IKW,
    CS,
    POS,
    OUT,
    WIDTH: tl.constexpr,
    NH: tl.constexpr,
    NK: tl.constexpr,
    NI: tl.constexpr,
    R: tl.constexpr,
    EPS: tl.constexpr,
    CS_STRIDE: tl.constexpr,
    D: tl.constexpr = 128,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, D)
    ptr = token * WIDTH + head * D + d
    x = tl.load(QKV + ptr).to(tl.float32)
    is_value = (head >= NH + NK) & (head < NH + 2 * NK)
    if not is_value:
        if head < NH:
            w = tl.load(QW + d)
        elif head < NH + NK:
            w = tl.load(KW + d)
        elif head < NH + 2 * NK + NI:
            w = tl.load(IQW + d)
        else:
            w = tl.load(IKW + d)
        # Original Ascend composition rounds gamma to input dtype first.
        gamma = (1.0 + w.to(tl.float32)).to(w.dtype).to(tl.float32)
        variance = tl.sum(x * x, 0) / D
        y = (x * tl.rsqrt(variance + EPS) * gamma).to(w.dtype).to(tl.float32)
        half = R // 2
        other_d = tl.where(d < half, d + half, d - half)
        other_x = tl.load(
            QKV + token * WIDTH + head * D + other_d, mask=d < R, other=0
        ).to(tl.float32)
        other_w = tl.load(QW + other_d, mask=d < R, other=0)
        if head >= NH:
            if head < NH + NK:
                other_w = tl.load(KW + other_d, mask=d < R, other=0)
            elif head < NH + 2 * NK + NI:
                other_w = tl.load(IQW + other_d, mask=d < R, other=0)
            else:
                other_w = tl.load(IKW + other_d, mask=d < R, other=0)
        other_gamma = (1.0 + other_w.to(tl.float32)).to(other_w.dtype).to(tl.float32)
        other_y = (
            (other_x * tl.rsqrt(variance + EPS) * other_gamma)
            .to(other_w.dtype)
            .to(tl.float32)
        )
        position = tl.load(POS + token)
        lane = d % half
        cos = tl.load(CS + position * CS_STRIDE + lane, mask=d < R, other=1).to(
            tl.float32
        )
        sin = tl.load(CS + position * CS_STRIDE + half + lane, mask=d < R, other=0).to(
            tl.float32
        )
        rotated = y * cos + tl.where(d < half, -other_y, other_y) * sin
        x = tl.where(d < R, rotated, y)
    tl.store(OUT + ptr, x)


@triton.jit
def _rotary_kernel(
    X,
    COS,
    SIN,
    OUT,
    NT: tl.constexpr,
    NH: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    HALF: tl.constexpr,
    B: tl.constexpr,
    NEOX: tl.constexpr,
    CS_BATCH: tl.constexpr,
    SEQ: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // NH
    d = tl.arange(0, B)
    half = R // 2
    cs_token = token if CS_BATCH else token % SEQ
    lane = d % half if NEOX else d // 2
    paired = tl.where(d < half, d + half, d - half) if NEOX else (d ^ 1)
    x = tl.load(X + row * D + d, mask=d < D, other=0)
    other = tl.load(X + row * D + paired, mask=d < R, other=0)
    cos = tl.load(COS + cs_token * half + lane, mask=d < R, other=1)
    sin = tl.load(SIN + cs_token * half + lane, mask=d < R, other=0)
    first = d < half if NEOX else (d % 2 == 0)
    if NEOX:
        # Existing NeoX path explicitly promotes operands to FP32.
        a = x.to(tl.float32) * cos.to(tl.float32)
        b = other.to(tl.float32) * sin.to(tl.float32)
        result = tl.where(first, a - b, a + b)
    else:
        # Existing GPT-J Torch formula rounds each product in input dtype.
        a = (x * cos.to(x.dtype)).to(x.dtype)
        b = (other * sin.to(x.dtype)).to(x.dtype)
        result = tl.where(first, a - b, a + b)
    tl.store(OUT + row * D + d, tl.where(d < R, result, x), mask=d < D)


def apply_minimax_m3_rotary_ascend(x, cos, sin, is_neox_style=True):
    """Out-of-place partial rotary for preprocessed [B,]S,H,D tensors."""
    _check_npu(x, cos, sin)
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise NotImplementedError("rotary supports BF16, FP16 and FP32")
    if x.ndim not in (3, 4) or cos.shape != sin.shape or cos.ndim not in (2, 3):
        raise ValueError("expected [B,]S,H,D and matching [B,]S,R/2 caches")
    r = cos.shape[-1] * 2
    if r > x.shape[-1]:
        raise ValueError("rotary prefix exceeds head dimension")
    if cos.shape[:-1] not in (x.shape[:-2], (x.shape[-3],)):
        raise ValueError("cos/sin sequence shape cannot broadcast to input")
    if r == 0:
        return x.clone()
    x = x.contiguous()
    cos, sin = cos.contiguous(), sin.contiguous()
    output = torch.empty_like(x)
    nt = x.numel() // (x.shape[-2] * x.shape[-1]) if x.shape[-2] * x.shape[-1] else 0
    if nt and x.shape[-2]:
        with torch_device_fn.device(x.device):
            _rotary_kernel[(nt * x.shape[-2],)](
                x,
                cos,
                sin,
                output,
                nt,
                x.shape[-2],
                x.shape[-1],
                r,
                r // 2,
                triton.next_power_of_2(x.shape[-1]),
                is_neox_style,
                cos.ndim == x.ndim - 1,
                x.shape[-3],
                enable_fp_fusion=False,
            )
    return output


@triton.jit
def _cache_kernel(
    CACHE,
    SLOTS,
    KEY,
    VALUE,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr,
    NB: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    C4: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    V0: tl.constexpr,
    V1: tl.constexpr,
    PLANE: tl.constexpr,
    INDEX: tl.constexpr,
    B: tl.constexpr,
):
    token = tl.program_id(0)
    offsets = tl.arange(0, B)
    slot = tl.load(SLOTS + token)
    block, pos = slot // BS, slot % BS
    head, dim = offsets // D, offsets % D
    key = tl.load(KEY + token * K0 + head * K1 + dim, mask=offsets < H * D, other=0)
    if INDEX:
        address = block * C0 + pos * C1 + dim * C2
        tl.store(CACHE + address, key, mask=(slot >= 0) & (offsets < D))
    else:
        if PLANE:
            kbase = block * C1 + pos * C2
            vbase = C0 + kbase
        else:
            kbase = block * C0 + pos * C2
            vbase = C1 + kbase
        tl.store(
            CACHE + kbase + head * C3 + dim * C4,
            key,
            mask=(slot >= 0) & (offsets < H * D),
        )
        value = tl.load(
            VALUE + token * V0 + head * V1 + dim, mask=offsets < H * D, other=0
        )
        tl.store(
            CACHE + vbase + head * C3 + dim * C4,
            value,
            mask=(slot >= 0) & (offsets < H * D),
        )


def _validate_cache(cache, slots, key, value, block_size, plane_major, index):
    _check_npu(cache, slots, key, value)
    if block_size <= 0 or key.dtype != torch.bfloat16 or cache.dtype != key.dtype:
        raise NotImplementedError(
            "cache insertion supports BF16 and positive block size"
        )
    _check_slots(slots, key.shape[0])
    if index:
        if (
            cache.ndim != 3
            or cache.shape[1] != block_size
            or key.ndim != 2
            or cache.shape[-1] != key.shape[-1]
        ):
            raise ValueError("index cache must be [blocks, block_size, D], keys [N,D]")
    elif (
        cache.ndim != 5
        or cache.shape[0 if plane_major else 1] != 2
        or cache.shape[2] != block_size
        or key.ndim != 3
        or value.shape != key.shape
        or cache.shape[3:] != key.shape[1:]
        or value.dtype != key.dtype
    ):
        raise ValueError("main cache layout/key/value contract mismatch")
    if key.stride(-1) != 1 or (value is not None and value.stride(-1) != 1):
        raise NotImplementedError("key/value head dimensions must be contiguous")


def insert_minimax_m3_kv_cache_ascend(
    cache, slots, key, value, block_size, *, plane_major
):
    """Mutate paged BF16 main cache; layout is mandatory and explicit."""
    _validate_cache(cache, slots, key, value, block_size, plane_major, False)
    if key.shape[0] == 0:
        return
    with torch_device_fn.device(key.device):
        _cache_kernel[(key.shape[0],)](
            cache,
            slots,
            key,
            value,
            key.shape[0],
            key.shape[1],
            key.shape[2],
            block_size,
            cache.shape[1 if plane_major else 0],
            *cache.stride(),
            key.stride(0),
            key.stride(1),
            value.stride(0),
            value.stride(1),
            plane_major,
            False,
            triton.next_power_of_2(key.shape[1] * key.shape[2]),
        )


def insert_minimax_m3_index_cache_ascend(cache, slots, key, block_size):
    """Mutate paged BF16 single-head index cache, skipping negative slots."""
    if key.ndim not in (2, 3) or (key.ndim == 3 and key.shape[1] != 1):
        raise ValueError("index keys must be [N,D] or [N,1,D]")
    key = key.view(key.shape[0], key.shape[-1])
    _validate_cache(cache, slots, key, None, block_size, False, True)
    if key.shape[0] == 0:
        return
    with torch_device_fn.device(key.device):
        _cache_kernel[(key.shape[0],)](
            cache,
            slots,
            key,
            key,
            key.shape[0],
            1,
            key.shape[1],
            block_size,
            cache.shape[0],
            *cache.stride(),
            0,
            0,
            key.stride(0),
            0,
            0,
            0,
            False,
            True,
            triton.next_power_of_2(key.shape[1]),
        )


def fused_minimax_m3_qknorm_rope_kv_insert_ascend(
    qkv,
    q_norm_weight,
    k_norm_weight,
    cos_sin_cache,
    positions,
    num_heads,
    num_kv_heads,
    rotary_dim,
    eps,
    index_q_norm_weight=None,
    index_k_norm_weight=None,
    num_index_heads=0,
    slot_mapping=None,
    index_slot_mapping=None,
    kv_cache=None,
    index_cache=None,
    block_size=0,
    q_out=None,
    index_q_out=None,
    kv_cache_dtype="auto",
    *,
    plane_major=False,
):
    """BF16 D128 forward-only packed QKV contract, matching vLLM mutations.

    Cache layout defaults to upstream block-major. Runtime-owned sparse cache
    callers must explicitly pass their plane-major declaration.
    """
    _check_npu(
        qkv,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        positions,
        index_q_norm_weight,
        index_k_norm_weight,
        slot_mapping,
        index_slot_mapping,
        kv_cache,
        index_cache,
        q_out,
        index_q_out,
    )
    if qkv.dtype != torch.bfloat16 or kv_cache_dtype not in ("auto", "bfloat16"):
        raise NotImplementedError("Ascend M3 fused path supports BF16 only")
    heads = (
        num_heads + 2 * num_kv_heads + (num_index_heads + 1 if num_index_heads else 0)
    )
    if num_heads <= 0 or num_kv_heads <= 0 or num_index_heads < 0 or eps < 0:
        raise ValueError("invalid head counts/epsilon")
    if qkv.ndim != 2 or qkv.shape[1] != heads * 128 or not qkv.is_contiguous():
        raise ValueError("QKV must be contiguous packed [N, heads*128]")
    if rotary_dim <= 0 or rotary_dim > 128 or rotary_dim % 2:
        raise ValueError("rotary_dim must be positive even and <=128")
    if (
        cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[1] != rotary_dim
        or not cos_sin_cache.is_contiguous()
    ):
        raise ValueError("cos/sin cache must be contiguous [positions,rotary_dim]")
    if (
        positions.dtype not in (torch.int32, torch.int64)
        or positions.ndim != 1
        or positions.numel() < qkv.shape[0]
        or not positions.is_contiguous()
    ):
        raise ValueError("positions must be contiguous integer vector covering tokens")
    weights = [q_norm_weight, k_norm_weight]
    if num_index_heads:
        weights += [index_q_norm_weight, index_k_norm_weight]
    if any(
        w is None or w.shape != (128,) or w.dtype != qkv.dtype or not w.is_contiguous()
        for w in weights
    ):
        raise ValueError(
            "each active normalization weight must be contiguous BF16 [128]"
        )
    if cos_sin_cache.dtype != qkv.dtype:
        raise ValueError("cos/sin cache must match QKV dtype")
    for output, count in ((q_out, num_heads), (index_q_out, num_index_heads)):
        if output is not None and (
            count <= 0
            or output.shape != (qkv.shape[0], count * 128)
            or output.dtype != qkv.dtype
            or not output.is_contiguous()
        ):
            raise ValueError("query output shape/dtype/layout mismatch")
    x = qkv.view(qkv.shape[0], heads, 128)
    key = x[:, num_heads : num_heads + num_kv_heads]
    value = x[:, num_heads + num_kv_heads : num_heads + 2 * num_kv_heads]
    if kv_cache is not None:
        _validate_cache(
            kv_cache, slot_mapping, key, value, block_size, plane_major, False
        )
    index_slots = index_slot_mapping if index_slot_mapping is not None else slot_mapping
    if index_cache is not None:
        if not num_index_heads:
            raise ValueError("index cache requires sparse index heads")
        _validate_cache(
            index_cache, index_slots, x[:, -1], None, block_size, False, True
        )
    if qkv.shape[0] == 0:
        return
    result = torch.empty_like(qkv)
    with torch_device_fn.device(qkv.device):
        _norm_rope_kernel[(qkv.shape[0], heads)](
            qkv,
            q_norm_weight,
            k_norm_weight,
            index_q_norm_weight if num_index_heads else q_norm_weight,
            index_k_norm_weight if num_index_heads else k_norm_weight,
            cos_sin_cache,
            positions,
            result,
            qkv.shape[1],
            num_heads,
            num_kv_heads,
            num_index_heads,
            rotary_dim,
            float(eps),
            cos_sin_cache.stride(0),
            enable_fp_fusion=False,
        )
    normalized = result.view_as(x)
    q = normalized[:, :num_heads]
    (qkv[:, : num_heads * 128] if q_out is None else q_out).copy_(q.flatten(1))
    key.copy_(normalized[:, num_heads : num_heads + num_kv_heads])
    if num_index_heads:
        start = num_heads + 2 * num_kv_heads
        iq = normalized[:, start : start + num_index_heads].flatten(1)
        (
            qkv[:, start * 128 : (start + num_index_heads) * 128]
            if index_q_out is None
            else index_q_out
        ).copy_(iq)
        x[:, -1].copy_(normalized[:, -1])
    if kv_cache is not None:
        insert_minimax_m3_kv_cache_ascend(
            kv_cache, slot_mapping, key, value, block_size, plane_major=plane_major
        )
    if index_cache is not None:
        insert_minimax_m3_index_cache_ascend(
            index_cache, index_slots, x[:, -1], block_size
        )


def _invoke_index(function, args, kwargs):
    bound = inspect.signature(function).bind(*args, **kwargs)
    bound.apply_defaults()
    values = bound.arguments
    tensors = []
    for value in values.values():
        if isinstance(value, torch.Tensor):
            tensors.append(value)
        elif isinstance(value, (tuple, list)):
            tensors.extend(x for x in value if isinstance(x, torch.Tensor))
    _check_npu(*tensors)
    query = values.get("idx_q")
    score = values.get("score")
    if query is not None:
        if (
            query.ndim != 3
            or query.dtype != torch.bfloat16
            or query.shape[-1] not in (1, 128)
        ):
            raise NotImplementedError("M3 index query supports BF16 [N,H,1|128]")
        heads, tokens = query.shape[1], query.shape[0]
        cache = values["index_kv_cache"]
        if isinstance(cache, (tuple, list)):
            cache = cache[0]
        if cache.ndim == 5 and cache.shape[0] == 2:
            cache = cache[0]
        if cache.ndim == 4 and cache.shape[2] == 1:
            cache = cache.squeeze(2)
        if (
            cache.ndim != 3
            or cache.shape[1:] != (128, query.shape[-1])
            or cache.dtype != query.dtype
        ):
            raise ValueError("index cache must expose BF16 [blocks,128,D]")
        if cache.stride(-1) != 1:
            raise ValueError("index cache last dimension must be contiguous")
        table = values["block_table"]
        if (
            table.ndim != 2
            or table.dtype not in (torch.int32, torch.int64)
            or table.stride(-1) != 1
        ):
            raise ValueError("block table must be an integer matrix")
        if values["num_kv_heads"] != heads:
            raise ValueError("index head count must equal KV head count")
    else:
        if score.ndim != 3 or score.dtype != torch.float32:
            raise ValueError("M3 index score must be FP32 [H,N,blocks]")
        heads, tokens = score.shape[:2]
    if heads <= 0:
        raise ValueError("index head count must be positive")
    for name in ("cu_seqlens_q", "seq_lens", "prefix_lens"):
        tensor = values.get(name)
        if tensor is not None and (
            tensor.ndim != 1
            or tensor.dtype not in (torch.int32, torch.int64)
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be an integer vector")
    if query is not None:
        batch = values["seq_lens"].numel()
        if table.shape[0] != batch or table.shape[1] < triton.cdiv(
            values["max_seq_len"], 128
        ):
            raise ValueError("block table does not cover the declared batch/context")
        if "cu_seqlens_q" in values and (
            values["cu_seqlens_q"].numel() != batch + 1
            or values["prefix_lens"].numel() != batch
        ):
            raise ValueError("prefill sequence metadata does not match batch")
        if (
            "decode_query_len" in values
            and tokens != batch * values["decode_query_len"]
        ):
            raise ValueError("decode query tokens do not match batch and query length")
    elif values["cu_seqlens_q"].numel() != values["prefix_lens"].numel() + 1:
        raise ValueError("top-k sequence metadata does not match batch")
    for name in (
        "max_query_len",
        "max_seq_len",
        "decode_query_len",
        "max_decode_query_len",
    ):
        value = values.get(name)
        if value is not None and value < 0:
            raise ValueError(f"{name} must be nonnegative")
    if "decode_query_len" in values and values["decode_query_len"] <= 0:
        raise ValueError("decode_query_len must be positive")
    if values.get("sm_scale") is not None and not math.isfinite(values["sm_scale"]):
        raise ValueError("index scale must be finite")
    out = values.get("out")
    if "topk" in values:
        topk = values["topk"]
        if topk <= 0 or min(values["init_blocks"], values["local_blocks"]) < 0:
            raise ValueError(
                "topk must be positive and forced block counts nonnegative"
            )
        if out is not None:
            if (
                out.dtype != torch.int32
                or out.ndim != 3
                or out.shape[0] != heads
                or out.shape[1] < tokens
                or out.shape[2] < topk
            ):
                raise ValueError("out must be int32 [heads,>=tokens,>=topk]")
            if any(
                t is not out
                and t.untyped_storage().data_ptr() == out.untyped_storage().data_ptr()
                for t in tensors
            ):
                raise ValueError("index output must not alias input storage")
        if tokens == 0:
            return (
                torch.empty(
                    (heads, 0, topk),
                    dtype=torch.int32,
                    device=(query if query is not None else score).device,
                )
                if out is None
                else out[:, :0, :topk]
            )
    elif tokens == 0:
        blocks = triton.cdiv(values["max_seq_len"], 128)
        return torch.empty(
            (heads, 0, triton.cdiv(blocks, 16) * 16),
            dtype=torch.float32,
            device=query.device,
        )
    with torch_device_fn.device((query if query is not None else score).device):
        return function(*args, **kwargs)


def minimax_m3_index_score_ascend(
    idx_q,
    index_kv_cache,
    block_table,
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    max_query_len,
    max_seq_len,
    num_kv_heads,
    sm_scale=None,
):
    from flaggems_vllm.runtime.backend._ascend.ops.minimax_m3_index import (
        minimax_m3_index_score,
    )

    return _invoke_index(
        minimax_m3_index_score,
        (
            idx_q,
            index_kv_cache,
            block_table,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            max_query_len,
            max_seq_len,
            num_kv_heads,
            sm_scale,
        ),
        {},
    )


def minimax_m3_index_topk_ascend(
    score,
    cu_seqlens_q,
    prefix_lens,
    max_query_len,
    topk,
    init_blocks,
    local_blocks,
    out=None,
):
    from flaggems_vllm.runtime.backend._ascend.ops.minimax_m3_index import (
        minimax_m3_index_topk,
    )

    return _invoke_index(
        minimax_m3_index_topk,
        (
            score,
            cu_seqlens_q,
            prefix_lens,
            max_query_len,
            topk,
            init_blocks,
            local_blocks,
            out,
        ),
        {},
    )


def minimax_m3_index_decode_ascend(
    idx_q,
    index_kv_cache,
    block_table,
    seq_lens,
    max_seq_len,
    topk,
    init_blocks,
    local_blocks,
    num_kv_heads,
    decode_query_len,
    max_decode_query_len=None,
    out=None,
    sm_scale=None,
):
    from flaggems_vllm.runtime.backend._ascend.ops.minimax_m3_index_decode import (
        minimax_m3_index_decode,
    )

    return _invoke_index(
        minimax_m3_index_decode,
        (
            idx_q,
            index_kv_cache,
            block_table,
            seq_lens,
            max_seq_len,
            topk,
            init_blocks,
            local_blocks,
            num_kv_heads,
            decode_query_len,
            max_decode_query_len,
            out,
            sm_scale,
        ),
        {},
    )


@triton.jit
def _gemma_weight_kernel(W, OUT, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    w = tl.load(W + i, mask=i < N, other=0)
    tl.store(OUT + i, 1.0 + w.to(tl.float32), mask=i < N)


def prepare_minimax_m3_gemma_weight_ascend(weight):
    """Create detached 1+w; the framework owns its cross-call lifetime."""
    _check_npu(weight)
    if (
        weight.ndim != 1
        or weight.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or not weight.is_contiguous()
    ):
        raise NotImplementedError("Gemma weight must be contiguous floating vector")
    output = torch.empty_like(weight)
    if weight.numel():
        with torch_device_fn.device(weight.device):
            _gemma_weight_kernel[(triton.cdiv(weight.numel(), 128),)](
                weight, output, weight.numel(), 128
            )
    return output


@triton.jit
def _count_valid_kernel(TOPK, OUT, K: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, B)
    value = tl.load(TOPK + row * K + i, mask=i < K, other=-1)
    tl.store(OUT + row, tl.sum((value >= 0).to(tl.int32), 0))


def attend_minimax_m3_sparse_ascend(
    query,
    cache,
    topk,
    block_table,
    seq_lens,
    query_lens,
    num_kv_heads,
    scale,
    block_size,
    output,
    *,
    plane_major,
):
    """Count valid blocks and call the operator-owned native CANN M3 API."""
    _check_npu(query, cache, topk, block_table, seq_lens, query_lens, output)
    if (
        cache.ndim != 5
        or cache.shape[0 if plane_major else 1] != 2
        or cache.shape[2] != block_size
    ):
        raise ValueError("main cache layout does not match its declaration")
    if (
        query.dtype != torch.bfloat16
        or cache.dtype != query.dtype
        or output.dtype != query.dtype
        or output.shape != query.shape
    ):
        raise NotImplementedError(
            "sparse attention requires matching BF16 query/cache/output"
        )
    if (
        topk.ndim != 3
        or topk.dtype != torch.int32
        or topk.shape[1] != query.shape[0]
        or topk.shape[-1] <= 0
    ):
        raise ValueError("topk must be int32 [index_heads, query_tokens, K>0]")
    if num_kv_heads <= 0 or cache.shape[3] != num_kv_heads or not math.isfinite(scale):
        raise ValueError("invalid KV head count/attention scale")
    if (
        block_size != 128
        or query.shape[-1] != 128
        or topk.shape[-1] > 16
        or topk.shape[0] != num_kv_heads
    ):
        raise NotImplementedError(
            "native sparse attention requires D128/block128/topk<=16 and KV-head index groups"
        )
    if (
        block_table.dtype != torch.int32
        or block_table.ndim != 2
        or query.ndim != 3
        or query.shape[1] % num_kv_heads
    ):
        raise ValueError(
            "native sparse attention block table/query head contract mismatch"
        )
    for lengths in (seq_lens, query_lens):
        if (
            lengths.ndim != 1
            or lengths.dtype not in (torch.int32, torch.int64)
            or not lengths.is_contiguous()
        ):
            raise ValueError("sequence metadata must be integer vectors")
    if (
        block_table.stride(-1) != 1
        or block_table.shape[0] != seq_lens.numel()
        or query_lens.numel() != seq_lens.numel()
    ):
        raise ValueError("attention sequence metadata does not match batch")
    from flaggems_vllm.ops.minimax_m3_ascend_native import load_minimax_m3_ascend_native

    native = load_minimax_m3_ascend_native()
    topk = topk.contiguous()
    counts = torch.empty(topk.shape[:-1], device=topk.device, dtype=torch.int32)
    if topk.numel() == 0:
        return
    key, value = (cache[0], cache[1]) if plane_major else (cache[:, 0], cache[:, 1])
    with torch_device_fn.device(query.device):
        _count_valid_kernel[(counts.numel(),)](
            topk, counts, topk.shape[-1], triton.next_power_of_2(topk.shape[-1])
        )
        result = native(
            query,
            key,
            value,
            topk,
            block_table,
            selectNumIdx=counts,
            actualSeqLengths=query_lens,
            actualSeqLengthsKv=seq_lens,
            numKeyValueHeads=num_kv_heads,
            scaleValue=scale,
            blockSize=block_size,
            topK=topk.shape[-1],
            innerPrecise=4,
        )
        output.copy_(result)


__all__ = [
    "apply_minimax_m3_rotary_ascend",
    "fused_minimax_m3_qknorm_rope_kv_insert_ascend",
    "insert_minimax_m3_kv_cache_ascend",
    "insert_minimax_m3_index_cache_ascend",
    "minimax_m3_index_score_ascend",
    "minimax_m3_index_topk_ascend",
    "minimax_m3_index_decode_ascend",
    "prepare_minimax_m3_gemma_weight_ascend",
    "attend_minimax_m3_sparse_ascend",
]
