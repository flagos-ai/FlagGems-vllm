# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Callable, Optional

import torch
import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner


@triton.jit
def _sparse_fp8_row(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Indices,
    Lengths,
    Partial,
    Stats,
    Output,
    LSE,
    Sink,
    row,
    H: tl.constexpr,
    SQ: tl.constexpr,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    ROPE: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    HAS_SINK: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
):
    heads = tl.program_id(1) * BH + tl.arange(0, BH)
    split = tl.program_id(2)
    dims = tl.arange(0, 512)
    columns = tl.arange(0, BN)
    query = tl.load(
        Q + (row * H + heads[:, None]) * 512 + dims[None, :], heads[:, None] < H, 0.0
    )
    qscale = tl.load(QS + row * H + heads, heads < H, 0)
    if ROPE:
        rdims = tl.arange(0, 64)
        qrope = tl.load(
            QR + (row * H + heads[:, None]) * 64 + rdims[None, :],
            heads[:, None] < H,
            0.0,
        )
    q_l1 = tl.sum(tl.abs(query.to(tl.float32)), 1) * qscale
    if ROPE:
        qr_l1 = tl.sum(tl.abs(qrope.to(tl.float32)), 1) * qscale
    length = tl.minimum(tl.maximum(tl.load(Lengths + row), 0), K) if HAS_LENGTH else K
    maximum = tl.full((BH,), float("-inf"), tl.float32)
    denominator = tl.full((BH,), 0.0, tl.float32)
    accumulator = tl.full((BH, 512), 0.0, tl.float32)
    split_end = tl.minimum((split + 1) * CHUNK, length)
    for begin in range(split * CHUNK, split_end, BN):
        offsets = begin + columns
        tokens = tl.load(Indices + row * K + offsets, offsets < split_end, -1)
        valid = (offsets < split_end) & (tokens >= 0) & (tokens < TOKENS)
        safe_tokens = tl.where(valid, tokens, 0)
        keys = tl.load(
            KV + safe_tokens[None, :].to(tl.int64) * 512 + dims[:, None],
            valid[None, :],
            0.0,
        )
        kscale = tl.load(KS + safe_tokens, valid, 0)
        scores = tl.dot(query, keys).to(tl.float32)
        if ROPE:
            krope = tl.load(
                KR + safe_tokens[None, :].to(tl.int64) * 64 + rdims[:, None],
                valid[None, :],
                0,
            )
            scores += tl.dot(qrope, krope).to(tl.float32)
        scores *= qscale[:, None] * kscale[None, :] * SCALE
        # Large dequantized inputs can amplify FP32 accumulation and scale
        # rounding beyond the LSE budget. Recompute those tiles in IEEE FP32.
        max_ks = tl.max(kscale, 0)
        error_bound = (
            q_l1
            * (448.0 * max_ks)
            * SCALE
            * ((512.0 * 2.0**-24) / (1.0 - 512.0 * 2.0**-24))
        )
        if ROPE:
            max_kr = tl.max(tl.max(tl.abs(krope.to(tl.float32)), 0) * kscale, 0)
            error_bound += (
                qr_l1 * max_kr * SCALE * ((64.0 * 2.0**-24) / (1.0 - 64.0 * 2.0**-24))
            )
        if tl.max(error_bound, 0) > 1.0e-4:
            scores = tl.full((BH, BN), 0.0, tl.float32)
            repair_dims = tl.arange(0, 64)
            for feature in range(0, 512, 64):
                feature_dims = feature + repair_dims
                exact_q = (
                    tl.load(
                        Q + (row * H + heads[:, None]) * 512 + feature_dims[None, :],
                        heads[:, None] < H,
                        0.0,
                        volatile=True,
                    ).to(tl.float32)
                    * qscale[:, None]
                )
                exact_k = (
                    tl.load(
                        KV
                        + safe_tokens[None, :].to(tl.int64) * 512
                        + feature_dims[:, None],
                        valid[None, :],
                        0.0,
                        volatile=True,
                    ).to(tl.float32)
                    * kscale[None, :]
                )
                scores += tl.dot(exact_q, exact_k, input_precision="ieee")
            scores *= SCALE
            if ROPE:
                exact_qr = qrope.to(tl.float32) * qscale[:, None]
                exact_kr = krope.to(tl.float32) * kscale[None, :]
                scores += tl.dot(exact_qr, exact_kr, input_precision="ieee") * SCALE
        scores = tl.where(valid[None, :], scores, float("-inf"))
        next_max = tl.maximum(maximum, tl.max(scores, 1))
        safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
        correction = tl.exp(maximum - safe_max)
        probability = tl.exp(scores - safe_max[:, None])
        denominator = denominator * correction + tl.sum(probability, 1)
        weights = probability * kscale[None, :]
        pscale = tl.maximum(tl.max(weights, 1) / 448.0, 1.0e-30)
        normalized = weights / pscale[:, None]
        high = normalized.to(tl.float8e4nv)
        low = ((normalized - high.to(tl.float32)) * 16.0).to(tl.float8e4nv)
        values = tl.trans(keys)
        product = tl.dot(high, values).to(tl.float32)
        product = tl.dot(low, values, product * 16.0).to(tl.float32) * (1.0 / 16.0)
        accumulator = accumulator * correction[:, None] + product * pscale[:, None]
        maximum = next_max
    if SPLITS == 1:
        logsum = tl.where(denominator > 0, maximum + tl.log(denominator), float("inf"))
        inverse = tl.div_rn(1.0, tl.where(denominator > 0, denominator, 1.0))
        if HAS_SINK:
            sink = tl.load(Sink + heads, heads < H, 0.0)
            inverse *= tl.where(
                denominator > 0, 1.0 / (1.0 + tl.exp(sink - logsum)), 0.0
            )
        tl.store(
            Output + (row * H + heads[:, None]) * 512 + dims[None, :],
            accumulator * inverse[:, None],
            heads[:, None] < H,
        )
        tl.store(LSE + (row // SQ * H + heads) * SQ + row % SQ, logsum, heads < H)
    else:
        base = (row * SPLITS + split) * H + heads
        tl.store(
            Partial + base[:, None] * 512 + dims[None, :],
            accumulator,
            heads[:, None] < H,
        )
        tl.store(Stats + base * 2, maximum, heads < H)
        tl.store(Stats + base * 2 + 1, denominator, heads < H)


# Keep the rare repair path out of the fused combine kernel's fast path.
_sparse_fp8_row_outlined = triton.jit(_sparse_fp8_row.fn, noinline=True)


_SPARSE_MLA_CONFIGS = runtime.get_tuned_config("mthreads_sparse_fp8_mla")


@libentry()
@libtuner(
    configs=_SPARSE_MLA_CONFIGS,
    key=["ROWS", "H", "K", "ROPE", "SPLITS"],
)
@triton.jit
def sparse_fp8_mla(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Indices,
    Lengths,
    Partial,
    Stats,
    Output,
    LSE,
    Sink,
    ROWS: tl.constexpr,
    H: tl.constexpr,
    SQ: tl.constexpr,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    ROPE: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    HAS_SINK: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    _sparse_fp8_row(
        Q,
        QR,
        KV,
        KR,
        QS,
        KS,
        Indices,
        Lengths,
        Partial,
        Stats,
        Output,
        LSE,
        Sink,
        row,
        H,
        SQ,
        K,
        TOKENS,
        ROPE,
        SCALE,
        HAS_LENGTH,
        HAS_SINK,
        SPLITS,
        CHUNK,
        BH,
        BN,
    )


@triton.jit
def compact_sparse_fp8(
    Q,
    KV,
    QS,
    KS,
    Indices,
    BadRows,
    BadCount,
    Output,
    LSE,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    SCALE: tl.constexpr,
    ROWS: tl.constexpr,
    SCAN_ROWS: tl.constexpr,
):
    if ROWS:
        positions = tl.program_id(0) * SCAN_ROWS + tl.arange(0, SCAN_ROWS)
        active = (positions < ROWS) & (
            tl.load(BadRows + positions, positions < ROWS, 1) == 0
        )
        first = tl.min(tl.where(active, positions, ROWS), 0)
        while first < ROWS:
            row = first.to(tl.int64)
            _sparse_fp8_row(
                Q,
                None,
                KV,
                None,
                QS,
                KS,
                Indices,
                None,
                None,
                None,
                Output,
                LSE,
                None,
                row,
                64,
                1,
                K,
                TOKENS,
                0,
                SCALE,
                False,
                False,
                1,
                tl.cdiv(K, 64) * 64,
                32,
                64,
            )
            active = active & (positions > first)
            first = tl.min(tl.where(active, positions, ROWS), 0)
    else:
        count = tl.load(BadCount)
        for request in range(tl.program_id(0), count, tl.num_programs(0)):
            row = tl.load(BadRows + request).to(tl.int64)
            _sparse_fp8_row(
                Q,
                None,
                KV,
                None,
                QS,
                KS,
                Indices,
                None,
                None,
                None,
                Output,
                LSE,
                None,
                row,
                64,
                1,
                K,
                TOKENS,
                0,
                SCALE,
                False,
                False,
                1,
                tl.cdiv(K, 64) * 64,
                32,
                64,
            )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mthreads_sparse_fp8_combine"),
    key=["ROWS", "H", "SQ", "SPLITS"],
)
@triton.jit
def combine_sparse_fp8(
    Partial,
    Stats,
    Sink,
    Output,
    LSE,
    ROWS: tl.constexpr,
    H: tl.constexpr,
    SQ: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_SINK: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    BV: tl.constexpr,
):
    row, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    splits = tl.arange(0, BLOCK_SPLITS)
    dims = tile * BV + tl.arange(0, BV)
    base = (row * SPLITS + splits) * H + head
    maxima = tl.load(Stats + base * 2, splits < SPLITS, float("-inf"))
    sums = tl.load(Stats + base * 2 + 1, splits < SPLITS, 0.0)
    maximum = tl.max(maxima, 0)
    safe_max = tl.where(maximum == float("-inf"), 0.0, maximum)
    weights = tl.exp(maxima - safe_max)
    denominator = tl.sum(weights * sums, 0)
    logsum = tl.where(denominator > 0, safe_max + tl.log(denominator), float("inf"))
    partial = tl.load(
        Partial + base[:, None] * 512 + dims[None, :], splits[:, None] < SPLITS, 0.0
    )
    values = tl.sum(partial * weights[:, None], 0) / tl.where(
        denominator > 0, denominator, 1.0
    )
    if HAS_SINK:
        sink = tl.load(Sink + head)
        values *= tl.where(denominator > 0, 1.0 / (1.0 + tl.exp(sink - logsum)), 0.0)
    tl.store(Output + (row * H + head) * 512 + dims, values)
    if tile == 0:
        tl.store(LSE + (row // SQ * H + head) * SQ + row % SQ, logsum)


@triton.jit
def flagged_sparse_fp8(
    Q,
    KV,
    QS,
    KS,
    Indices,
    Flags,
    Output,
    LSE,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    SCALE: tl.constexpr,
    FLAGS_PER_ROW: tl.constexpr,
    BLOCK_FLAGS: tl.constexpr,
):
    # Check a block of flags at once for the large single-page path.
    SCAN_ROWS: tl.constexpr = 32 if ROWS >= 128 and FLAGS_PER_ROW == 1 else 1
    columns = tl.arange(0, BLOCK_FLAGS)
    for first in range(
        tl.program_id(0) * SCAN_ROWS, ROWS, tl.num_programs(0) * SCAN_ROWS
    ):
        if SCAN_ROWS > 1:
            positions = first + tl.arange(0, SCAN_ROWS)
            active = tl.load(Flags + positions, positions < ROWS, 0) != 0
            request = tl.min(tl.where(active, positions, ROWS), 0)
        else:
            flags = tl.load(
                Flags + first * FLAGS_PER_ROW + columns, columns < FLAGS_PER_ROW, 0
            )
            request = tl.where(tl.max(flags, 0) != 0, first, ROWS)
        while request < ROWS:
            row = request.to(tl.int64)
            _sparse_fp8_row(
                Q,
                None,
                KV,
                None,
                QS,
                KS,
                Indices,
                None,
                None,
                None,
                Output,
                LSE,
                None,
                row,
                64,
                1,
                K,
                TOKENS,
                0,
                SCALE,
                False,
                False,
                1,
                tl.cdiv(K, 64) * 64,
                32,
                64,
            )
            if SCAN_ROWS > 1:
                request = tl.min(
                    tl.where(active & (positions > request), positions, ROWS), 0
                )
            else:
                request = ROWS


def prepare_flash_mla_sparse_fwd_w8a8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> tuple[
    Callable[[], tuple[torch.Tensor, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]
]:
    """Prepare contiguous page64 E4M3 FP8 MLA; scales must be positive and finite.

    RoPE is divided by content scales. Calls through a prepared handle must be
    serialized on one stream; tensor contents may change between executions.
    """
    if q_nope.ndim != 4 or q_nope.shape[1] not in (1, 2) or q_nope.shape[-1] != 512:
        raise NotImplementedError("requires Q [batch, 1 or 2, heads, 512]")
    batch, queries, heads, _ = q_nope.shape
    if heads not in (0, 16, 32) and (heads < 64 or heads % 64):
        raise NotImplementedError("requires 16, 32, or a multiple of 64 query heads")
    if (
        k_nope.ndim not in (3, 4)
        or k_nope.shape[1] != 64
        or k_nope.shape[-1] != 512
        or (k_nope.ndim == 4 and k_nope.shape[2] != 1)
    ):
        raise NotImplementedError("requires KV [pages,64,512] or [pages,64,1,512]")
    if q_nope.dtype != torch.float8_e4m3fn or k_nope.dtype != torch.float8_e4m3fn:
        raise TypeError("Q and KV content must be E4M3 FP8")
    if (
        q_rope.dtype not in (torch.bfloat16, torch.float16)
        or k_rope.dtype != q_rope.dtype
    ):
        raise TypeError("RoPE must have matching BF16 or FP16 dtype")
    if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32:
        raise TypeError("scales must be FP32")
    if (
        q_rope.ndim != 4
        or q_rope.shape[-1] not in (0, 64)
        or q_rope.shape[:-1] != q_nope.shape[:-1]
        or k_rope.shape != (*k_nope.shape[:-1], q_rope.shape[-1])
        or q_scale.shape != (*q_nope.shape[:-1], 1)
        or k_scale.shape != (*k_nope.shape[:-1], 1)
    ):
        raise ValueError("RoPE/scales must match content rows with width 0 or 64 / 1")
    if (
        indices.dtype != torch.int32
        or indices.ndim != 3
        or indices.shape[:2] != (batch, queries)
        or indices.shape[-1] > 65536
    ):
        raise ValueError("indices must be INT32 [batch, queries, topk <= 65536]")
    if topk_length is not None and topk_length.shape == (batch,) and queries == 1:
        topk_length = topk_length[:, None]
    if topk_length is not None and (
        topk_length.shape != (batch, queries) or topk_length.dtype != torch.int32
    ):
        raise ValueError("topk_length must be INT32 [batch, queries]")
    if attn_sink is not None and (
        attn_sink.shape != (heads,) or attn_sink.dtype != torch.float32
    ):
        raise ValueError("attn_sink must be FP32 [heads]")
    if out is not None and (
        out.shape != (batch, queries, heads, 512) or out.dtype != q_rope.dtype
    ):
        raise ValueError("out must match output shape and RoPE dtype")
    inputs = (q_nope, q_rope, k_nope, k_rope, q_scale, k_scale, indices)
    optional = tuple(t for t in (topk_length, attn_sink) if t is not None)
    tensors = inputs + optional + ((out,) if out is not None else ())
    if q_nope.device.type != "musa" or any(t.device != q_nope.device for t in tensors):
        raise ValueError("all inputs must share a MUSA device")
    if any(not t.is_contiguous() for t in tensors):
        raise NotImplementedError("requires contiguous inputs and output")
    if (
        out is not None
        and out.numel()
        and any(
            out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
            for t in inputs + optional
        )
    ):
        raise ValueError("out must not alias an input")
    if softmax_scale is not None and (
        isinstance(softmax_scale, bool) or not isinstance(softmax_scale, (float, int))
    ):
        raise TypeError("softmax_scale must be a Python number")
    scale = (
        (512 + q_rope.shape[-1]) ** -0.5
        if softmax_scale is None
        else float(softmax_scale)
    )
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax_scale must be positive and finite")
    rows, topk = batch * queries, indices.shape[-1]
    splits = max(
        1,
        min(
            32,
            triton.cdiv(topk, 256),
            triton.cdiv(256, max(1, rows * triton.cdiv(heads, 32))),
        ),
    )
    chunk = triton.cdiv(topk, splits * 64) * 64
    if rows * splits * heads * 2056 > 8 * 1024**3 or k_nope.shape[0] * 64 >= 2**31:
        raise NotImplementedError(
            "workspace exceeds 8 GiB or cache exceeds INT32 indices"
        )
    output = (
        torch.empty(q_nope.shape, dtype=q_rope.dtype, device=q_nope.device)
        if out is None
        else out
    )
    lse = torch.empty(
        (batch, heads, queries), dtype=torch.float32, device=q_nope.device
    )
    large_single_page = rows >= 128 and topk == 128
    if (
        (0 < rows <= 16 or large_single_page)
        and queries == 1
        and heads == 64
        and q_rope.shape[-1] == 0
        and topk > 0
        and k_nope.shape[0] > 0
        and topk_length is None
        and attn_sink is None
    ):
        try:
            from flaggems_vllm.runtime.backend._mthreads._sparse_fp8_pipeline import (
                prepare_short_pipeline,
            )
        except ImportError:
            short = None
        else:
            with runtime.torch_device_fn.device(q_nope.device):
                short = prepare_short_pipeline(
                    q_nope,
                    k_nope,
                    q_scale,
                    k_scale,
                    indices,
                    output,
                    lse,
                    scale,
                    combine_sparse_fp8.jit_function,
                    _sparse_fp8_row_outlined,
                )
        if short is not None:
            run_short, flags = short
            if flags is not None:
                fallback_args = (
                    q_nope,
                    k_nope,
                    q_scale,
                    k_scale,
                    indices,
                    flags,
                    output,
                    lse,
                    rows,
                    topk,
                    k_nope.shape[0] * 64,
                    scale,
                    flags.shape[1],
                    triton.next_power_of_2(flags.shape[1]),
                )
                with runtime.torch_device_fn.device(q_nope.device):
                    programs = rows
                    if large_single_page:
                        programs = min(
                            rows,
                            runtime.torch_device_fn.get_device_properties(
                                q_nope.device
                            ).multi_processor_count,
                        )
                    grid = (programs, 2, 1)
                    fallback = flagged_sparse_fp8.warmup(
                        *fallback_args, num_warps=4, num_stages=1, grid=grid
                    )
                    fallback_runner = fallback[grid]

            def run_fast():
                with runtime.torch_device_fn.device(q_nope.device):
                    run_short()
                    if flags is not None:
                        fallback_runner(*fallback_args)
                return output, lse

            return run_fast, run_fast()

    if (
        rows >= 128
        and queries == 1
        and heads == 64
        and q_rope.shape[-1] == 0
        and topk >= 128
        and k_nope.shape[0] > 0
        and topk_length is None
        and attn_sink is None
    ):
        try:
            from flaggems_vllm.runtime.backend._mthreads._sparse_fp8_pipeline import (
                prepare_pipeline,
            )
        except ImportError:
            pipeline = None
        else:
            with runtime.torch_device_fn.device(q_nope.device):
                pipeline = prepare_pipeline(
                    q_nope, k_nope, q_scale, k_scale, indices, output, lse, scale
                )
        if pipeline is not None:
            run_pipeline, bad_rows, bad_count = pipeline
            programs = min(
                rows,
                runtime.torch_device_fn.get_device_properties(
                    q_nope.device
                ).multi_processor_count,
            )
            scan_rows = (
                triton.next_power_of_2(triton.cdiv(rows, programs))
                if bad_count is None
                else 0
            )
            fallback_args = (
                q_nope,
                k_nope,
                q_scale,
                k_scale,
                indices,
                bad_rows,
                bad_count,
                output,
                lse,
                topk,
                k_nope.shape[0] * 64,
                scale,
                rows if scan_rows else 0,
                scan_rows,
            )
            with runtime.torch_device_fn.device(q_nope.device):
                fallback = compact_sparse_fp8.warmup(
                    *fallback_args, num_warps=4, num_stages=1, grid=(programs, 2, 1)
                )
                fallback_runner = fallback[(programs, 2, 1)]

            def run_fast():
                with runtime.torch_device_fn.device(q_nope.device):
                    run_pipeline()
                    fallback_runner(*fallback_args)
                return output, lse

            return run_fast, run_fast()

    if splits == 1:
        partial, stats = None, None
    else:
        partial = torch.empty(
            (rows, splits, heads, 512), dtype=torch.float32, device=q_nope.device
        )
        stats = torch.empty(
            (rows, splits, heads, 2), dtype=torch.float32, device=q_nope.device
        )

    launch_mla = sparse_fp8_mla[
        lambda meta: (rows, triton.cdiv(heads, meta["BH"]), splits)
    ]
    launch_options = {}
    mla_args = (
        *inputs[:6],
        indices,
        topk_length,
        partial,
        stats,
        output,
        lse,
        attn_sink,
        rows,
        heads,
        queries,
        topk,
        k_nope.shape[0] * 64,
        q_rope.shape[-1],
        scale,
        topk_length is not None,
        attn_sink is not None,
        splits,
        chunk,
    )

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal launch_mla, launch_options
        if rows == 0 or heads == 0:
            return output, lse
        with runtime.torch_device_fn.device(q_nope.device):
            try:
                launch_mla(*mla_args, **launch_options)
            except OutOfResources:
                # The cached tile may exceed shared memory on another compiler.
                config = min(
                    _SPARSE_MLA_CONFIGS,
                    key=lambda c: (c.kwargs["BH"] * c.kwargs["BN"], c.num_warps),
                )
                launch_options = config.all_kwargs()
                grid = (rows, triton.cdiv(heads, launch_options["BH"]), splits)
                launch_mla = sparse_fp8_mla.jit_function[grid]
                launch_mla(*mla_args, **launch_options)
            if splits > 1:
                combine_sparse_fp8[lambda meta: (rows, heads, 512 // meta["BV"])](
                    partial,
                    stats,
                    attn_sink,
                    output,
                    lse,
                    rows,
                    heads,
                    queries,
                    splits,
                    attn_sink is not None,
                    triton.next_power_of_2(splits),
                )
        return output, lse

    return run, run()


def flash_mla_sparse_fwd_w8a8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_cache_lora: torch.Tensor,
    k_cache_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    indices: torch.Tensor,
    softmax_scale: Optional[float] = None,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    *,
    out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, output = prepare_flash_mla_sparse_fwd_w8a8(
        q_nope,
        q_rope,
        k_cache_lora,
        k_cache_rope,
        q_scale,
        k_scale,
        indices,
        softmax_scale=softmax_scale,
        attn_sink=attn_sink,
        topk_length=topk_length,
        out=out,
    )
    return output
