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

"""MODEL1 sparse decode with an FP8 KV cache for MetaX C550.

``_schedule`` lays every request's 16-token tiles (main topk tiles, then extra
tiles) end to end and splits them over one partition per multiprocessor;
``_attention`` runs one partition per 512-thread program, its work done by the
MACA core in flash_mla_with_kvcache_model1.cu (compiled to LLVM bitcode by
mxcc and linked through ``extern_libs``); ``_combine`` or ``_combine_groups``
merges the requests that span several partitions and applies the attention
sink. The schedule is computed on the first call with a given
FlashMLASchedMeta and reused by later calls with the same object.

KV cache layout (MODEL1 FP8): per page, ``page_size`` tokens of 576 bytes
(448 e4m3 NoPE values, then 64 BF16 RoPE values), followed by 8 bytes per
token (7 ue8m0 scales for the seven 64-wide NoPE groups, 1 pad byte).
"""

import functools
import hashlib
import os
import subprocess
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.core import extern

HEADS = tl.constexpr(64)
HEAD_DIM = tl.constexpr(512)
LN2 = tl.constexpr(0.6931471805599453)
LOG2E = 1.4426950408889634
BLOCK_N = 16  # tokens per tile (fixed by the core)
# Cost of starting a request, in tiles, when balancing partitions.
SCHEDULE_OVERHEAD = 5
# _combine when a request is expected to split into at least this many pieces
# (min(its tiles, partitions / batch)), else _combine_groups.
COMBINE_MANY_PIECES = 8
COMBINE_PROGRAMS = 832
COMBINE_CHUNK = 4
COMBINE_GROUP = 8

_CORE = Path(__file__).with_suffix(".cu")
_CORE_SIGNATURE = (tl.int64, tl.int64, tl.int64, tl.int32, tl.int64, tl.int64, tl.int32,
                   tl.int64, tl.int32, tl.int64, tl.int32, tl.int64, tl.int64, tl.int32,
                   tl.int64, tl.int64, tl.int64, tl.int64, tl.int64, tl.int64, tl.float32,
                   tl.int32, tl.int32)  # fmt: skip


@functools.cache
def _core_library():
    """The core as LLVM bitcode, built once per source in the Triton cache."""
    source = _CORE.read_bytes()
    cache = Path(os.environ.get("TRITON_CACHE_DIR", Path.home() / ".triton" / "cache"))
    path = cache / f"mla-model1-{hashlib.sha256(source).hexdigest()[:16]}.bc"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(f".{os.getpid()}.tmp")
        mxcc = Path(os.environ["MACA_PATH"]) / "mxgpu_llvm" / "bin" / "mxcc"
        subprocess.run(
            [str(mxcc), "-x", "maca", "--offload-arch=xcore1000", "-maca-device-only",
             "-fgpu-rdc", "-emit-llvm", "-c", "-O3", str(_CORE), "-o", str(partial)],
            check=True,
        )  # fmt: skip
        partial.replace(path)
    return str(path)


@extern
def _partition(*args, _semantic=None):
    """mla_partition of the core: one call per thread of a 512-thread program."""
    return tl.core.extern_elementwise(
        "",
        "",
        list(args),
        {_CORE_SIGNATURE: ("mla_partition", tl.float32)},
        is_pure=False,
        _semantic=_semantic,
    )


@triton.jit
def _request_tiles(
    lengths, extra_lengths, b, HAS_EXTRA: tl.constexpr, BLOCK: tl.constexpr
):
    """Tiles of request b: its main tiles (at least one), then its extra tiles."""
    tiles = (tl.maximum(tl.load(lengths + b), 1) + BLOCK - 1) // BLOCK
    if HAS_EXTRA:
        tiles += (tl.maximum(tl.load(extra_lengths + b), 0) + BLOCK - 1) // BLOCK
    return tiles


@triton.jit
def _schedule(
    lengths,
    extra_lengths,
    metadata,
    split_requests,
    batch,
    PARTS: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
    OVERHEAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program fills the partitions greedily: each request costs its tiles
    plus OVERHEAD, and each partition takes up to an even share of the total.
    Metadata row of a partition: first request, its first tile, last request,
    its end tile (exclusive), split index of the first request's piece, the
    first request's lengths (main, extra), split index of the last request's
    piece if that request continues in the next partition. Splits are numbered
    request by request. split_requests lists the requests that span several
    partitions (request, first split, end split); its row PARTS holds their
    count."""
    total = 0
    for b in range(batch):
        total += _request_tiles(lengths, extra_lengths, b, HAS_EXTRA, BLOCK) + OVERHEAD
    payload = (total + PARTS - 1) // PARTS + OVERHEAD
    # Cursor: the request being placed, its first unplaced tile and how many
    # of its pieces are placed; splits used by the placed requests; the
    # split_requests row of the request being split.
    request = 0
    tile = 0
    pieces = 0
    splits = 0
    open_row = 0
    split_count = 0
    for part in range(PARTS):
        row = metadata + part * 8
        safe = tl.minimum(request, batch - 1)
        tl.store(row, request)
        tl.store(row + 1, tile)
        tl.store(row + 4, splits + pieces)
        tl.store(row + 5, tl.load(lengths + safe))
        tl.store(row + 6, tl.load(extra_lengths + safe))
        if pieces == 1:
            tl.store(split_requests + split_count * 4, request)
            tl.store(split_requests + split_count * 4 + 1, splits)
            open_row = split_count
            split_count += 1
        remain = payload
        going = request < batch
        while going:
            rest = (
                _request_tiles(lengths, extra_lengths, request, HAS_EXTRA, BLOCK) - tile
            )
            if remain >= rest + OVERHEAD:
                splits += pieces + 1
                if pieces > 0:
                    tl.store(split_requests + open_row * 4 + 2, splits)
                remain -= rest + OVERHEAD
                request += 1
                tile = 0
                pieces = 0
                going = request < batch
            else:
                if remain > OVERHEAD:
                    tile += remain - OVERHEAD
                    pieces += 1
                going = False
        if tile > 0:
            tl.store(row + 2, request)
            tl.store(row + 3, tile)
            tl.store(row + 7, splits + pieces - 1)
        else:
            last = tl.maximum(request - 1, 0)
            tl.store(row + 2, request - 1)
            tl.store(
                row + 3,
                _request_tiles(lengths, extra_lengths, last, HAS_EXTRA, BLOCK),
            )
            tl.store(row + 7, 0)
    tl.store(split_requests + PARTS * 4, split_count)


@triton.jit
def _attention(
    q,
    cache,
    page_stride,
    page_shift,
    extra_cache,
    extra_page_stride,
    extra_page_shift,
    indices,
    topk,
    extra_indices,
    extra_topk,
    lengths,
    extra_lengths,
    has_extra,
    metadata,
    partial,
    partial_lse,
    out,
    lse,
    sink,
    scale_log2,
    batch,
):
    """One partition per program of 512 threads. Arguments are cast because
    Triton specializes 0 and 1 into constants, while the core has one fixed
    signature."""
    _partition(
        tl.cast(q, tl.int64),
        tl.cast(cache, tl.int64),
        tl.cast(page_stride, tl.int64),
        tl.cast(page_shift, tl.int32),
        tl.cast(extra_cache, tl.int64),
        tl.cast(extra_page_stride, tl.int64),
        tl.cast(extra_page_shift, tl.int32),
        tl.cast(indices, tl.int64),
        tl.cast(topk, tl.int32),
        tl.cast(extra_indices, tl.int64),
        tl.cast(extra_topk, tl.int32),
        tl.cast(lengths, tl.int64),
        tl.cast(extra_lengths, tl.int64),
        tl.cast(has_extra, tl.int32),
        tl.cast(metadata, tl.int64),
        tl.cast(partial, tl.int64),
        tl.cast(partial_lse, tl.int64),
        tl.cast(out, tl.int64),
        tl.cast(lse, tl.int64),
        tl.cast(sink, tl.int64),
        tl.cast(scale_log2, tl.float32),
        tl.cast(batch, tl.int32),
        tl.program_id(0),
    )


@triton.jit
def _finish(maximum, denominator, sink, h, HAS_SINK: tl.constexpr):
    """Natural-log lse and output factor of merged pieces (maximum and
    denominator in the log2 domain), with the sink exp(lse) / (exp(lse) +
    exp(sink)), including sink = +-inf and lse = +inf."""
    valid = denominator > 0
    natural = tl.where(valid, (tl.log2(denominator) + maximum) * LN2, float("inf"))
    factor = tl.where(valid, 1.0 / denominator, 0.0)
    if HAS_SINK:
        sink_value = tl.load(sink + h)
        factor *= tl.where(
            sink_value == float("inf"), 0.0, 1.0 / (1.0 + tl.exp(sink_value - natural))
        )
    return natural, factor


@triton.jit
def _combine(
    partial,
    partial_lse,
    split_requests,
    parts,
    sink,
    out,
    lse,
    HAS_SINK: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """Merge the requests that split into many pieces: the programs loop over
    (request, head) items; per item, all log2 sums are read at once and the
    pieces CHUNK at a time."""
    count = tl.load(split_requests + parts * 4)
    every = tl.arange(0, MAX_SPLITS)
    c = tl.arange(0, CHUNK)
    d = tl.arange(0, HEAD_DIM)
    for item in range(tl.program_id(0), count * HEADS, tl.num_programs(0)):
        r = item // HEADS
        h = item % HEADS
        b = tl.load(split_requests + r * 4)
        first = tl.load(split_requests + r * 4 + 1)
        end = tl.load(split_requests + r * 4 + 2)
        values = tl.load(
            partial_lse + (first + every) * HEADS + h,
            mask=first + every < end,
            other=float("-inf"),
        )
        maximum = tl.max(values, 0)
        safe = tl.where(maximum == float("-inf"), 0.0, maximum)
        denominator = tl.sum(tl.exp2(values - safe), 0)
        result = tl.zeros((HEAD_DIM,), tl.float32)
        for start in range(first, end, CHUNK):
            split = start + c
            ok = split < end
            weight = tl.exp2(
                tl.load(partial_lse + split * HEADS + h, mask=ok, other=float("-inf"))
                - safe
            )
            piece = tl.load(
                partial + (split[:, None] * HEADS + h) * HEAD_DIM + d[None, :],
                mask=ok[:, None],
                other=0.0,
            )
            result += tl.sum(piece.to(tl.float32) * weight[:, None], 0)
        natural, factor = _finish(maximum, denominator, sink, h, HAS_SINK)
        tl.store(
            out + (b * HEADS + h) * HEAD_DIM + d, (result * factor).to(tl.bfloat16)
        )
        tl.store(lse + b * HEADS + h, natural)


@triton.jit
def _combine_groups(
    partial,
    partial_lse,
    split_requests,
    parts,
    sink,
    out,
    lse,
    HAS_SINK: tl.constexpr,
    GROUP: tl.constexpr,
):
    """Merge the requests that split into a few pieces: one program per
    request and GROUP heads, one piece at a time."""
    r = tl.program_id(0)
    if r < tl.load(split_requests + parts * 4):
        b = tl.load(split_requests + r * 4)
        first = tl.load(split_requests + r * 4 + 1)
        end = tl.load(split_requests + r * 4 + 2)
        h = tl.program_id(1) * GROUP + tl.arange(0, GROUP)
        rows = h[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
        maximum = tl.full((GROUP,), float("-inf"), tl.float32)
        denominator = tl.zeros((GROUP,), tl.float32)
        result = tl.zeros((GROUP, HEAD_DIM), tl.float32)
        for split in range(first, end):
            value = tl.load(partial_lse + split * HEADS + h)
            new_maximum = tl.maximum(maximum, value)
            safe = tl.where(new_maximum == float("-inf"), 0.0, new_maximum)
            alpha = tl.exp2(maximum - safe)
            weight = tl.exp2(value - safe)
            piece = tl.load(partial + split * HEADS * HEAD_DIM + rows).to(tl.float32)
            result = result * alpha[:, None] + piece * weight[:, None]
            denominator = denominator * alpha + weight
            maximum = new_maximum
        natural, factor = _finish(maximum, denominator, sink, h, HAS_SINK)
        tl.store(
            out + b * HEADS * HEAD_DIM + rows,
            (result * factor[:, None]).to(tl.bfloat16),
        )
        tl.store(lse + b * HEADS + h, natural)


@functools.cache
def _parts(device):
    """One partition per multiprocessor."""
    return torch.cuda.get_device_properties(device).multi_processor_count


@functools.cache
def _is_c550(device):
    try:
        from flaggems_vllm import runtime

        return (
            runtime.device.vendor_name == "metax"
            and torch.cuda.get_device_name(device) == "MetaX C550"
        )
    except Exception:
        return False


def _cache_supported(cache):
    page = cache.shape[1]
    return (
        cache.ndim == 4
        and cache.shape[2] == 1
        and cache.shape[3] == 584
        and page & (page - 1) == 0
        and cache.element_size() == 1
        and cache.stride(1) == 584
    )


def _indices_supported(indices):
    # The core reads four ids at a time.
    return (
        indices is not None
        and indices.is_contiguous()
        and indices.shape[-1] % 4 == 0
        and indices.data_ptr() % 16 == 0
    )


def can_use_model1_metax(
    q,
    kv,
    indices,
    out,
    lse,
    extra_kv=None,
    extra_indices=None,
    topk_length=None,
    extra_topk_length=None,
):
    """Whether this path can serve the call; callers fall back if not."""
    if not _is_c550(q.device):
        return False
    if (
        q.dtype != torch.bfloat16
        or q.ndim != 4
        or tuple(q.shape[1:]) != (1, 64, 512)
        or not q.is_contiguous()
    ):
        return False
    if not _indices_supported(indices) or topk_length is None:
        return False
    if not _cache_supported(kv):
        return False
    if out is not None and (
        out.shape != q.shape or out.dtype != q.dtype or not out.is_contiguous()
    ):
        return False
    if lse is not None and (lse.dtype != torch.float32 or not lse.is_contiguous()):
        return False
    if extra_kv is not None and (
        not _indices_supported(extra_indices)
        or extra_topk_length is None
        or not _cache_supported(extra_kv)
    ):
        return False
    return True


def sparse_decode_model1_metax(
    q,
    kv,
    indices,
    *,
    attn_sink,
    topk_length,
    extra_kv,
    extra_indices,
    extra_topk_length,
    out,
    lse,
    sm_scale,
    sched_meta=None,
):
    """Writes out [B, 1, 64, 512] and lse [B, 64, 1]. The tile schedule is kept
    in sched_meta.tile_scheduler_metadata / num_splits and reused while set."""
    batch = q.shape[0]
    parts = _parts(q.device)
    has_extra = extra_kv is not None
    extra_cache = extra_kv if has_extra else kv
    extra_indices = extra_indices if has_extra else indices
    extra_lengths = extra_topk_length if has_extra else topk_length
    if sched_meta is not None and sched_meta.tile_scheduler_metadata is not None:
        metadata = sched_meta.tile_scheduler_metadata
        split_requests = sched_meta.num_splits
    else:
        metadata = torch.empty((parts, 8), dtype=torch.int32, device=q.device)
        split_requests = torch.empty((parts + 1, 4), dtype=torch.int32, device=q.device)
        _schedule[(1,)](
            topk_length,
            extra_lengths,
            metadata,
            split_requests,
            batch,
            parts,
            has_extra,
            SCHEDULE_OVERHEAD,
            BLOCK_N,
            num_warps=1,
        )
        if sched_meta is not None:
            sched_meta.tile_scheduler_metadata = metadata
            sched_meta.num_splits = split_requests
    # Pieces of split requests (normalized, BF16) and their log2 sums; every
    # partition adds at most one piece beyond one per request.
    partial = torch.empty(
        (batch + parts, 64, 512), dtype=torch.bfloat16, device=q.device
    )
    partial_lse = torch.empty((batch + parts, 64), dtype=torch.float32, device=q.device)
    sink = attn_sink if attn_sink is not None else lse
    _attention[(parts,)](
        q.data_ptr(),
        kv.data_ptr(),
        kv.stride(0),
        kv.shape[1].bit_length() - 1,
        extra_cache.data_ptr(),
        extra_cache.stride(0),
        extra_cache.shape[1].bit_length() - 1,
        indices.data_ptr(),
        indices.shape[-1],
        extra_indices.data_ptr(),
        extra_indices.shape[-1],
        topk_length.data_ptr(),
        extra_lengths.data_ptr(),
        int(has_extra),
        metadata.data_ptr(),
        partial.data_ptr(),
        partial_lse.data_ptr(),
        out.data_ptr(),
        lse.data_ptr(),
        attn_sink.data_ptr() if attn_sink is not None else 0,
        sm_scale * LOG2E,
        batch,
        num_warps=8,
        num_stages=1,
        extern_libs={"mla_model1": _core_library()},
    )
    max_tiles = triton.cdiv(indices.shape[-1], BLOCK_N)
    if has_extra:
        max_tiles += triton.cdiv(extra_indices.shape[-1], BLOCK_N)
    if min(max_tiles, parts / batch) >= COMBINE_MANY_PIECES:
        # At most one split request per batch entry, 64 items each.
        _combine[(min(COMBINE_PROGRAMS, 64 * batch),)](
            partial,
            partial_lse,
            split_requests,
            parts,
            sink,
            out,
            lse,
            attn_sink is not None,
            triton.next_power_of_2(parts),
            COMBINE_CHUNK,
            num_warps=1,
        )
    else:
        _combine_groups[(parts, 64 // COMBINE_GROUP)](
            partial,
            partial_lse,
            split_requests,
            parts,
            sink,
            out,
            lse,
            attn_sink is not None,
            COMBINE_GROUP,
            num_warps=2,
        )
