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

import functools
import hashlib
import math
import os
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import extern
from triton.runtime.cache import get_cache_manager

from flaggems_vllm.ops.flash_mla import _get_num_sms
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry
from flaggems_vllm.utils.shape_utils import can_use_int32_index


@triton.jit
def accumulate_int8_pv(
    KVBase, Valid, PHigh, PLow, Rescale, PScale, Accumulator, OFFSET: tl.constexpr
):
    # Reload V in small tiles to bound live registers and shared-memory usage.
    dims = OFFSET + tl.arange(0, 128)
    value = tl.load(KVBase + dims[None, :], Valid[:, None], 0)
    high = tl.dot(tl.trans(value), tl.trans(PHigh), out_dtype=tl.int32).to(tl.float32)
    low = tl.dot(tl.trans(value), tl.trans(PLow), out_dtype=tl.int32).to(tl.float32)
    return (
        Accumulator * Rescale[None, :] + (high + low * (1.0 / 127.0)) * PScale[None, :]
    )


@libentry()
@triton.jit
def dense_int8_mla_kernel(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Table,
    Lengths,
    Partial,
    Stats,
    Output,
    LSE,
    Q_STRIDES: tl.constexpr,
    QR_STRIDES: tl.constexpr,
    KV_STRIDES: tl.constexpr,
    KR_STRIDES: tl.constexpr,
    QS_STRIDES: tl.constexpr,
    KS_STRIDES: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    SQ: tl.constexpr,
    CAUSAL: tl.constexpr,
    BH: tl.constexpr,
):
    head_tile, batch, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    heads = head_tile * BH + tl.arange(0, BH)
    queries, qheads = heads // H, heads % H
    merged_heads = SQ * H
    content_dims = tl.arange(0, 64)
    rope_dims = tl.arange(0, 64)
    tokens = tl.arange(0, 64)
    q_base = (
        Q
        + batch * Q_STRIDES[0]
        + queries[:, None] * Q_STRIDES[1]
        + qheads[:, None] * Q_STRIDES[2]
    )
    qr = tl.load(
        QR
        + batch * QR_STRIDES[0]
        + queries[:, None] * QR_STRIDES[1]
        + qheads[:, None] * QR_STRIDES[2]
        + rope_dims[None, :],
        heads[:, None] < merged_heads,
        0,
    )
    qs = tl.load(
        QS + batch * QS_STRIDES[0] + queries * QS_STRIDES[1] + qheads * QS_STRIDES[2],
        heads < merged_heads,
        0,
    )
    length = tl.load(Lengths + batch * LENGTH_STRIDE)
    query_length = tl.full((BH,), length, tl.int32)
    if CAUSAL:
        query_length = tl.maximum(length - SQ + queries + 1, 0)
    maximum = tl.full((BH,), -float("inf"), tl.float32)
    denominator = tl.zeros((BH,), tl.float32)
    acc0 = tl.zeros((128, BH), tl.float32)
    acc1 = tl.zeros((128, BH), tl.float32)
    acc2 = tl.zeros((128, BH), tl.float32)
    acc3 = tl.zeros((128, BH), tl.float32)
    valid_pages = tl.cdiv(length, 64)
    first_page = valid_pages * split // SPLITS
    end_page = valid_pages * (split + 1) // SPLITS
    for page_idx in range(first_page, end_page):
        page = tl.load(Table + batch * TABLE_STRIDE + page_idx).to(tl.int64)
        valid = page_idx * 64 + tokens < length
        kv_base = KV + page * KV_STRIDES[0] + tokens[:, None] * KV_STRIDES[1]
        content_raw = tl.zeros((64, BH), tl.int32)
        # A full query tile kept across PV spills on C550.
        for chunk in tl.range(8, loop_unroll_factor=1, disable_licm=True):
            dims = chunk * 64 + content_dims
            q = tl.load(
                q_base + dims[None, :], heads[:, None] < merged_heads, 0, volatile=True
            )
            key = tl.load(kv_base + dims[None, :], valid[:, None], 0)
            content_raw = tl.dot(key, tl.trans(q), content_raw, out_dtype=tl.int32)
        kr = tl.load(
            KR
            + page * KR_STRIDES[0]
            + tokens[:, None] * KR_STRIDES[1]
            + rope_dims[None, :],
            valid[:, None],
            0,
        )
        ks = tl.load(KS + page * KS_STRIDES[0] + tokens * KS_STRIDES[1], valid, 0)
        content_scores = tl.trans(content_raw).to(tl.float32)
        rope_scores = tl.trans(tl.dot(kr, tl.trans(qr), out_dtype=tl.float32))
        scores = (
            (content_scores + rope_scores)
            * qs[:, None]
            * ks[None, :]
            * (SCALE * 1.4426950408889634)
        )
        scores = tl.where(
            page_idx * 64 + tokens[None, :] < query_length[:, None],
            scores,
            -float("inf"),
        )
        next_maximum = tl.maximum(maximum, tl.max(scores, 1))
        safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        probabilities = tl.exp2(scores - safe_maximum[:, None])
        rescale = tl.exp2(maximum - safe_maximum)
        denominator = denominator * rescale + tl.sum(probabilities, 1)
        scaled = probabilities * ks[None, :]
        probability_scale = tl.maximum(tl.max(scaled, 1) / 127.0, 1.0e-30)
        scaled /= probability_scale[:, None]
        high = tl.floor(scaled + 0.5).to(tl.int8)
        low = tl.floor((scaled - high.to(tl.float32)) * 127.0 + 0.5).to(tl.int8)
        acc0 = accumulate_int8_pv(
            kv_base, valid, high, low, rescale, probability_scale, acc0, 0
        )
        acc1 = accumulate_int8_pv(
            kv_base, valid, high, low, rescale, probability_scale, acc1, 128
        )
        acc2 = accumulate_int8_pv(
            kv_base, valid, high, low, rescale, probability_scale, acc2, 256
        )
        acc3 = accumulate_int8_pv(
            kv_base, valid, high, low, rescale, probability_scale, acc3, 384
        )
        maximum = next_maximum
    out_dims = tl.arange(0, 128)
    if SPLITS == 1:
        inverse = 1.0 / tl.where(denominator > 0, denominator, 1.0)
        base = (
            Output + (batch * merged_heads + heads[None, :]) * 512 + out_dims[:, None]
        )
        tl.store(base, acc0 * inverse[None, :], heads[None, :] < merged_heads)
        tl.store(base + 128, acc1 * inverse[None, :], heads[None, :] < merged_heads)
        tl.store(base + 256, acc2 * inverse[None, :], heads[None, :] < merged_heads)
        tl.store(base + 384, acc3 * inverse[None, :], heads[None, :] < merged_heads)
        logsum = tl.where(
            denominator > 0,
            maximum * 0.6931471805599453 + tl.log(denominator),
            float("inf"),
        )
        tl.store(
            LSE + (batch * H + qheads) * SQ + queries, logsum, heads < merged_heads
        )
    else:
        base = (
            Partial
            + ((split * B + batch) * merged_heads + heads[None, :]) * 512
            + out_dims[:, None]
        )
        tl.store(base, acc0, heads[None, :] < merged_heads)
        tl.store(base + 128, acc1, heads[None, :] < merged_heads)
        tl.store(base + 256, acc2, heads[None, :] < merged_heads)
        tl.store(base + 384, acc3, heads[None, :] < merged_heads)
        tl.store(
            Stats + (split * 2 * B + batch) * merged_heads + heads,
            maximum,
            heads < merged_heads,
        )
        tl.store(
            Stats + ((split * 2 + 1) * B + batch) * merged_heads + heads,
            denominator,
            heads < merged_heads,
        )


@libentry()
@triton.jit
def dense_int8_mla_merge(
    Partial,
    Stats,
    Output,
    LSE,
    B: tl.constexpr,
    H: tl.constexpr,
    SQ: tl.constexpr,
    SPLITS: tl.constexpr,
    NORMALIZED: tl.constexpr = False,
):
    batch, head = tl.program_id(0), tl.program_id(1)
    splits = tl.arange(0, SPLITS)
    dims = tl.arange(0, 512)
    merged_heads = H * SQ
    maxima = tl.load(Stats + (splits * 2 * B + batch) * merged_heads + head)
    maximum = tl.max(maxima, 0)
    safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp2(maxima - safe_maximum)
    sums = tl.load(Stats + ((splits * 2 + 1) * B + batch) * merged_heads + head)
    denominator = tl.sum(sums * weights, 0)
    values = tl.load(
        Partial
        + ((splits[:, None] * B + batch) * merged_heads + head) * 512
        + dims[None, :]
    )
    if NORMALIZED:
        weights = sums * weights / tl.where(denominator > 0, denominator, 1.0)
        output = tl.sum(values * weights[:, None], 0)
    else:
        output = tl.sum(values * weights[:, None], 0) / tl.where(
            denominator > 0, denominator, 1.0
        )
    tl.store(Output + (batch * merged_heads + head) * 512 + dims, output)
    logsum = tl.where(
        denominator > 0,
        maximum * 0.6931471805599453 + tl.log(denominator),
        float("inf"),
    )
    tl.store(LSE + (batch * H + head % H) * SQ + head // H, logsum)


class FlashMLAInt8PreparedHandle:
    """Reuse workspace for decode; calls must be serialized on one stream."""

    def __init__(
        self, tensors: tuple[torch.Tensor, ...], splits: int, scale: float, causal: bool
    ):
        self.tensors = tensors
        query, rope, _, _, _, _, _, _ = tensors
        self.batch, self.queries, self.heads, _ = query.shape
        self.causal = causal
        self.head_tile = 16
        self.splits = splits
        self.scale = scale
        self.num_warps = 4
        self.kernel = dense_int8_mla_kernel
        self.kernel_arguments = {}
        self.batch_order = None
        self.batch_order_kernel = None
        self.grid = (
            triton.cdiv(self.queries * self.heads, self.head_tile),
            self.batch,
            self.splits,
        )
        self.normalized_partials = False
        self.output = torch.empty(
            (self.batch, self.queries, self.heads, 512),
            dtype=rope.dtype,
            device=query.device,
        )
        self.lse = torch.empty(
            (self.batch, self.heads, self.queries),
            dtype=torch.float32,
            device=query.device,
        )
        workspace_splits = splits if splits > 1 else 0
        self.partial = torch.empty(
            (workspace_splits, self.batch, self.queries * self.heads, 512),
            dtype=torch.float32,
            device=query.device,
        )
        self.stats = torch.empty(
            (workspace_splits, 2, self.batch, self.queries * self.heads),
            dtype=torch.float32,
            device=query.device,
        )

    def _prepare_batch_order(self):
        if self.batch_order is not None:
            lengths = self.tensors[7]
            self.batch_order_kernel[(1,)](
                lengths,
                self.batch_order,
                lengths.stride(0),
                self.batch,
                triton.next_power_of_2(self.batch),
                num_warps=4,
                num_stages=1,
            )

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.batch == 0 or self.heads == 0:
            return self.output, self.lse
        query, rope, kv, kv_rope, qs, ks, table, lengths = self.tensors
        with torch_device_fn.device(query.device):
            self._prepare_batch_order()
            self.kernel[self.grid](
                *self.tensors,
                self.partial,
                self.stats,
                self.output,
                self.lse,
                query.stride(),
                rope.stride(),
                kv.stride(),
                kv_rope.stride(),
                qs.stride(),
                ks.stride(),
                table.stride(0),
                lengths.stride(0),
                self.batch,
                self.heads,
                self.splits,
                self.scale,
                self.queries,
                self.causal,
                BH=self.head_tile,
                num_warps=self.num_warps,
                num_stages=2,
                scenario="mla",
                **self.kernel_arguments,
            )
            if self.splits > 1:
                dense_int8_mla_merge[(self.batch, self.queries * self.heads)](
                    self.partial,
                    self.stats,
                    self.output,
                    self.lse,
                    self.batch,
                    self.heads,
                    self.queries,
                    self.splits,
                    NORMALIZED=self.normalized_partials,
                    num_warps=4,
                )
        return self.output, self.lse


@libentry()
@triton.jit
def dense_int8_probability_merge(
    Stats,
    Metadata,
    PS,
    LSE,
    Lengths,
    SplitFlag,
    LS: tl.constexpr,
    B: tl.constexpr,
    LENGTH_BLOCK: tl.constexpr,
    H: tl.constexpr,
    SQ: tl.constexpr,
    ROWS: tl.constexpr,
    SPLITS: tl.constexpr,
    ADAPTIVE: tl.constexpr,
    QUANTIZED_CACHE: tl.constexpr,
    SORT_BATCHES: tl.constexpr = False,
):
    rows = tl.program_id(0) * 32 + tl.arange(0, 32)
    if QUANTIZED_CACHE:
        positions = tl.arange(0, SPLITS * 4)
        row_batches = rows // (H * SQ)
        row_heads = rows % (H * SQ)
        offsets = (
            (
                row_batches[:, None].to(tl.int64) * SPLITS * 3
                + (positions[None, :] // 4) * 3
            )
            * 4
            + positions[None, :] % 4
        ) * H * SQ + row_heads[:, None]
        maxima = tl.load(Stats + offsets, rows[:, None] < ROWS, -float("inf"))
        weighted_maxima = tl.load(
            Stats + offsets + 4 * H * SQ, rows[:, None] < ROWS, 0.0
        )
        local_denominators = tl.load(
            Stats + offsets + 8 * H * SQ, rows[:, None] < ROWS, 0.0
        )
        maximum = tl.max(maxima, 1)
        safe = tl.where(maximum == -float("inf"), 0.0, maximum)
        weights = tl.exp2(maxima - safe[:, None])
        denominator = tl.sum(local_denominators * weights, 1)
        scale = tl.maximum(tl.max(weighted_maxima * weights, 1), 1.0e-30)
        output_scale = scale / tl.where(denominator > 0, denominator, 1.0)
        logsum = tl.where(
            denominator > 0,
            maximum * 0.6931471805599453 + tl.log(denominator),
            float("inf"),
        )
        tl.store(
            LSE + (row_batches * H + row_heads % H) * SQ + row_heads // H,
            logsum,
            rows < ROWS,
        )
    else:
        splits = tl.arange(0, SPLITS)
        offsets = (rows[:, None] * SPLITS + splits[None, :]) * 2
        maxima = tl.load(Stats + offsets, rows[:, None] < ROWS, -float("inf"))
        weighted_maxima = tl.load(Stats + offsets + 1, rows[:, None] < ROWS, 0.0)
        maximum = tl.max(maxima, 1)
        safe = tl.where(maximum == -float("inf"), 0.0, maximum)
        weights = tl.exp2(maxima - safe[:, None])
        weighted = tl.max(weighted_maxima * weights, 1)
        scale = tl.maximum(weighted * (1.0 / 16129.0), 1.0e-30)
        output_scale = scale
    tl.store(Metadata + rows * 2, safe, rows < ROWS)
    tl.store(Metadata + rows * 2 + 1, 1.0 / scale, rows < ROWS)
    tl.store(PS + rows, output_scale, rows < ROWS)
    if tl.program_id(0) == 0:
        if ADAPTIVE or SORT_BATCHES:
            batches = tl.arange(0, LENGTH_BLOCK)
            lengths = tl.load(Lengths + batches * LS, batches < B, 0)
            maximum_length = tl.max(lengths, 0)
            minimum_length = tl.min(tl.where(batches < B, lengths, 2147483647), 0)
        if ADAPTIVE:
            tl.store(SplitFlag, maximum_length != minimum_length)
        else:
            tl.store(SplitFlag, 0)
        if SORT_BATCHES:
            if maximum_length != minimum_length:
                keys = tl.where(batches < B, lengths * LENGTH_BLOCK + batches, -1)
                ordered = tl.sort(keys, descending=True) % LENGTH_BLOCK
            else:
                ordered = batches
            tl.store(SplitFlag + 1 + batches, ordered, batches < B)


@libentry()
@gluon.jit
def dense_int8_value_merge(
    Partial,
    PartialD,
    Metadata,
    Scale,
    Output,
    LSE,
    SplitFlag,
    B: gl.constexpr,
    H: gl.constexpr,
    SQ: gl.constexpr,
    SPLITS: gl.constexpr,
    QUANTIZED_CACHE: gl.constexpr = False,
    MERGE_HEADS: gl.constexpr = 4,
    UNIFORM_SPLITS: gl.constexpr = 1,
):
    if UNIFORM_SPLITS > 1 or gl.load(SplitFlag) != 0:
        parent: gl.constexpr = gl.BlockedLayout(
            [1, 1, 4], [1, 1, 64], [1, 2, 2], [2, 1, 0]
        )
        meta: gl.constexpr = gl.SliceLayout(2, parent)
        outlayout: gl.constexpr = gl.SliceLayout(0, parent)
        batch = gl.program_id(0)
        hs = gl.program_id(1) * MERGE_HEADS
        splits = gl.arange(0, SPLITS, layout=gl.SliceLayout(1, meta))
        heads = hs + gl.arange(0, MERGE_HEADS, layout=gl.SliceLayout(0, meta))
        oh = hs + gl.arange(0, MERGE_HEADS, layout=gl.SliceLayout(1, outlayout))
        dims = gl.arange(0, 512, layout=gl.SliceLayout(0, outlayout))
        s2 = gl.expand_dims(splits, 1)
        h2 = gl.expand_dims(heads, 0)
        s3 = gl.expand_dims(s2, 2)
        h3 = gl.expand_dims(h2, 2)
        d3 = gl.expand_dims(gl.expand_dims(dims, 0), 0)
        # Native PV writes only active partitions.
        split_count = SPLITS
        if UNIFORM_SPLITS > 1 and UNIFORM_SPLITS < SPLITS:
            split_count = gl.where(gl.load(SplitFlag) != 0, SPLITS, UNIFORM_SPLITS)
        values = gl.load(
            Partial + ((s3.to(gl.int64) * B + batch) * H * SQ + h3) * 512 + d3,
            (h3 < H * SQ) & (s3 < split_count),
            0.0,
        )
        scale = gl.load(Scale + batch * H * SQ + oh, oh < H * SQ, 0.0)
        if QUANTIZED_CACHE:
            output = gl.sum(values, 0) * gl.expand_dims(scale, 1)
        else:
            denominator = gl.sum(
                gl.load(
                    PartialD + (s2 * B + batch) * H * SQ + h2,
                    (h2 < H * SQ) & (s2 < split_count),
                    0.0,
                ),
                0,
            )
            denom = gl.convert_layout(
                gl.where(denominator > 0, denominator, 1.0),
                gl.SliceLayout(1, outlayout),
            )
            output = gl.sum(values, 0) * gl.expand_dims(scale / denom, 1)
        gl.store(
            Output
            + (batch.to(gl.int64) * H * SQ + gl.expand_dims(oh, 1)) * 512
            + gl.expand_dims(dims, 0),
            output,
            gl.expand_dims(oh < H * SQ, 1),
        )
        if not QUANTIZED_CACHE:
            maximum = gl.load(Metadata + (batch * H * SQ + oh) * 2, oh < H * SQ, 0.0)
            denominator = gl.convert_layout(denominator, gl.SliceLayout(1, outlayout))
            logsum = gl.where(
                denominator > 0,
                maximum * 0.6931471805599453 + gl.log(denominator),
                float("inf"),
            )
            gl.store(LSE + (batch * H + oh % H) * SQ + oh // H, logsum, oh < H * SQ)


@libentry()
@triton.jit
def dense_int8_batch_order(
    Lengths, Order, LS: tl.constexpr, B: tl.constexpr, BLOCK: tl.constexpr
):
    batches = tl.arange(0, BLOCK)
    lengths = tl.load(Lengths + batches * LS, batches < B, 0)
    maximum = tl.max(lengths, 0)
    minimum = tl.min(tl.where(batches < B, lengths, 2147483647), 0)
    if maximum != minimum:
        keys = tl.where(batches < B, lengths * BLOCK + batches, -1)
        ordered = tl.sort(keys, descending=True) % BLOCK
    else:
        ordered = batches
    tl.store(Order + batches, ordered, batches < B)


@functools.lru_cache(maxsize=1)
def _compiler():
    path = Path(os.environ.get("MACA_PATH", "/opt/maca")) / "mxgpu_llvm/bin/mxcc"
    return path.resolve(), subprocess.check_output(
        [str(path), "--version"], stderr=subprocess.STDOUT
    )


@functools.lru_cache(maxsize=128)
def _library(stage, tile, spec):
    """Build one CTA-wide device function before CUDA Graph capture."""
    compiler, version = _compiler()
    source = Path(__file__).with_suffix(".cu")
    flags = [
        "-x",
        "maca",
        "--offload-arch=xcore1000",
        "-maca-device-only",
        "-fgpu-rdc",
        "-emit-llvm",
        "-c",
        "-O3",
        f"-DFMLA_STAGE_{stage.upper()}",
    ]
    if stage == "small":
        flags.append(f"-DHEADS={tile}")
    elif stage == "values":
        flags.append(f"-DVDIM={tile}")
    flags.extend(f"-DFMLA_{key}={value}" for key, value in spec)
    key = hashlib.sha256(
        source.read_bytes() + version + str(compiler).encode() + repr(flags).encode()
    ).hexdigest()
    cache = get_cache_manager(key)
    filename = f"mla_int8_{stage}.bc"
    cached = cache.get_file(filename)
    if cached is not None:
        return cached
    # A unique temporary directory also permits concurrent handle preparation.
    with tempfile.TemporaryDirectory(prefix="flaggems-mla-int8-") as temporary:
        output = Path(temporary) / filename
        subprocess.run(
            [str(compiler), *flags, str(source), "-o", str(output)], check=True
        )
        return cache.put(output.read_bytes(), filename)


def _scale_spec(handle):
    value = struct.unpack("f", struct.pack("f", handle.scale * 1.4426950408889634))[0]
    return (
        ("HQ", handle.heads),
        ("SQ", handle.queries),
        ("CAP", handle.tensors[6].shape[1] * 64),
        ("CAUSAL", int(handle.causal)),
        ("BF16", int(handle.tensors[1].dtype == torch.bfloat16)),
        ("SCALE", value.hex() + "f"),
    )


SMALL_SIGNATURE = (tl.int64,) * 15 + (tl.int32,) * 6 + (tl.float32,) + (tl.int32,) * 3


@extern
def _small_device(*args, _semantic=None):
    return tl.core.extern_elementwise(
        "",
        "",
        list(args),
        {SMALL_SIGNATURE: ("mla_int8_small", tl.float32)},
        is_pure=False,
        _semantic=_semantic,
    )


@triton.jit
def small_native(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Table,
    Lengths,
    Partial,
    Stats,
    Output,
    LSE,
    Q_STRIDES: tl.constexpr,
    QR_STRIDES: tl.constexpr,
    KV_STRIDES: tl.constexpr,
    KR_STRIDES: tl.constexpr,
    QS_STRIDES: tl.constexpr,
    KS_STRIDES: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    SQ: tl.constexpr,
    CAUSAL: tl.constexpr,
    BH: tl.constexpr,
    BatchOrder=None,
):
    _small_device(
        tl.cast(Q, tl.int64),
        tl.cast(QR, tl.int64),
        tl.cast(KV, tl.int64),
        tl.cast(KR, tl.int64),
        tl.cast(QS, tl.int64),
        tl.cast(KS, tl.int64),
        tl.cast(Table, tl.int64),
        tl.cast(Lengths, tl.int64),
        tl.cast(Partial, tl.int64),
        tl.cast(Stats, tl.int64),
        tl.cast(Output, tl.int64),
        tl.cast(LSE, tl.int64),
        tl.cast(TABLE_STRIDE, tl.int64),
        tl.cast(LENGTH_STRIDE, tl.int64),
        (
            tl.full((), 0, tl.int64)
            if BatchOrder is None
            else tl.cast(BatchOrder, tl.int64)
        ),
        tl.cast(B, tl.int32),
        tl.cast(H, tl.int32),
        tl.cast(SQ, tl.int32),
        tl.cast(SPLITS, tl.int32),
        tl.cast(CAUSAL, tl.int32),
        tl.cast(QR.dtype.element_ty == tl.bfloat16, tl.int32),
        tl.cast(SCALE * 1.4426950408889634, tl.float32),
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )


SCORES_SIGNATURE = (tl.int64,) * 14 + (tl.int32,) * 6 + (tl.float32,) + (tl.int32,) * 3


@extern
def _scores_device(*args, _semantic=None):
    return tl.core.extern_elementwise(
        "",
        "",
        list(args),
        {SCORES_SIGNATURE: ("mla_int8_scores", tl.float32)},
        is_pure=False,
        _semantic=_semantic,
    )


@triton.jit
def scores_native(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Table,
    Lengths,
    Quantized,
    PageStats,
    SplitStats,
    Q_STRIDES: tl.constexpr,
    QR_STRIDES: tl.constexpr,
    KV_STRIDES: tl.constexpr,
    KR_STRIDES: tl.constexpr,
    QS_STRIDES: tl.constexpr,
    KS_STRIDES: tl.constexpr,
    TS: tl.constexpr,
    LS: tl.constexpr,
    H: tl.constexpr,
    SQ: tl.constexpr,
    CAP: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BH: tl.constexpr = 64,
    SPLITS: tl.constexpr = 4,
    BatchOrder=None,
    COL_STATS: tl.constexpr = True,
    NATIVE_SYNC: tl.constexpr = True,
):
    _scores_device(
        tl.cast(Q, tl.int64),
        tl.cast(QR, tl.int64),
        tl.cast(KV, tl.int64),
        tl.cast(KR, tl.int64),
        tl.cast(QS, tl.int64),
        tl.cast(KS, tl.int64),
        tl.cast(Table, tl.int64),
        tl.cast(Lengths, tl.int64),
        tl.cast(Quantized, tl.int64),
        tl.cast(PageStats, tl.int64),
        tl.cast(SplitStats, tl.int64),
        (
            tl.full((), 0, tl.int64)
            if BatchOrder is None
            else tl.cast(BatchOrder, tl.int64)
        ),
        tl.cast(TS, tl.int64),
        tl.cast(LS, tl.int64),
        tl.cast(H, tl.int32),
        tl.cast(SQ, tl.int32),
        tl.cast(CAP, tl.int32),
        tl.cast(SPLITS, tl.int32),
        tl.cast(CAUSAL, tl.int32),
        tl.cast(QR.dtype.element_ty == tl.bfloat16, tl.int32),
        tl.cast(SCALE * 1.4426950408889634, tl.float32),
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )


VALUES_SIGNATURE = (tl.int64,) * 14 + (tl.int32,) * 11


@extern
def _values_device(*args, _semantic=None):
    return tl.core.extern_elementwise(
        "",
        "",
        list(args),
        {VALUES_SIGNATURE: ("mla_int8_pv", tl.float32)},
        is_pure=False,
        _semantic=_semantic,
    )


@triton.jit
def values_native_fragments(
    KV,
    Table,
    Lengths,
    Scores,
    CacheMetadata,
    Metadata,
    PS,
    Output,
    LSE,
    Partial,
    PartialD,
    SplitFlag,
    KV_STRIDES: tl.constexpr,
    TS: tl.constexpr,
    LS: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    SQ: tl.constexpr,
    CAP: tl.constexpr,
    BH: tl.constexpr = 64,
    SPLITS: tl.constexpr = 1,
    ADAPTIVE: tl.constexpr = False,
    QUANTIZED_CACHE: tl.constexpr = True,
    SORT_BATCHES: tl.constexpr = False,
    UNIFORM_SPLITS: tl.constexpr = 1,
    STAGED: tl.constexpr = False,
):
    tl.static_assert(BH == 64 and QUANTIZED_CACHE)
    _values_device(
        tl.cast(KV, tl.int64),
        tl.cast(Table, tl.int64),
        tl.cast(Lengths, tl.int64),
        tl.cast(Scores, tl.int64),
        tl.cast(CacheMetadata, tl.int64),
        tl.cast(Metadata, tl.int64),
        tl.cast(PS, tl.int64),
        tl.cast(Output, tl.int64),
        tl.cast(Partial, tl.int64),
        tl.cast(SplitFlag, tl.int64),
        tl.cast(KV_STRIDES[0], tl.int64),
        tl.cast(KV_STRIDES[1], tl.int64),
        tl.cast(TS, tl.int64),
        tl.cast(LS, tl.int64),
        tl.cast(B, tl.int32),
        tl.cast(H * SQ, tl.int32),
        tl.cast(CAP, tl.int32),
        tl.cast(SPLITS, tl.int32),
        tl.cast(ADAPTIVE, tl.int32),
        tl.cast(SORT_BATCHES, tl.int32),
        tl.cast(UNIFORM_SPLITS, tl.int32),
        tl.cast(Output.dtype.element_ty == tl.bfloat16, tl.int32),
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )


class SmallKernel:
    def __init__(self, heads, handle):
        spec = _scale_spec(handle) + (("B", handle.batch), ("SPLITS", handle.splits))
        self.bitcode = _library("small", heads, spec)

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            kwargs.update(
                num_warps=4,
                num_stages=2,
                scenario="mla",
                extern_libs={"mla_int8_small": self.bitcode},
            )
            return small_native[grid](*args, **kwargs)

        return launch


class ScoresKernel:
    def __init__(self, handle):
        spec = _scale_spec(handle) + (("SPLITS", handle.score_splits),)
        self.bitcode = _library("scores", 64, spec)

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            kwargs.update(
                num_warps=4,
                num_stages=1,
                scenario="mla",
                extern_libs={"mla_int8_scores": self.bitcode},
            )
            return scores_native[grid](*args, **kwargs)

        return launch


class ValuesKernel:
    def __init__(self, dims, handle):
        self.dims = dims
        spec = (
            ("B", handle.batch),
            ("HEADS", handle.heads * handle.queries),
            ("CAP", handle.capacity),
            ("SPLITS", handle.value_splits),
            ("ADAPTIVE", int(handle.adaptive)),
            ("SORT", int(handle.sort_batches)),
            ("UNIFORM", handle.uniform_value_splits),
            ("BF16", int(handle.output.dtype == torch.bfloat16)),
        )
        self.bitcode = _library("values", dims, spec)

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            kwargs.update(
                num_warps=self.dims // 64,
                num_stages=1,
                scenario="mla",
                extern_libs={"mla_int8_pv": self.bitcode},
            )
            return values_native_fragments[(512 // self.dims, *grid[1:])](
                *args, **kwargs
            )

        return launch


class NativeSmallHandle(FlashMLAInt8PreparedHandle):
    def __init__(self, tensors, scale, causal):
        capacity = tensors[6].shape[1] * 64
        splits = min(
            (
                4
                if tensors[0].shape[1] * tensors[0].shape[2] == 32 and capacity <= 8192
                else 8
            ),
            triton.next_power_of_2(tensors[6].shape[1]),
        )
        super().__init__(tensors, splits, scale, causal)
        self.head_tile = self.queries * self.heads
        self.normalized_partials = True
        if self.head_tile == 32 and 1 < self.batch <= 256:
            self.batch_order = torch.empty(
                (self.batch,), dtype=torch.int32, device=tensors[0].device
            )
            self.batch_order_kernel = dense_int8_batch_order
        self.kernel_arguments = {"BatchOrder": self.batch_order}
        self.grid = (1, self.batch, self.splits)
        self.kernel = SmallKernel(self.head_tile, self)


class NativeCachedHandle(FlashMLAInt8PreparedHandle):
    def __init__(self, tensors, scale, causal):
        super().__init__(tensors, 1, scale, causal)
        merged = self.queries * self.heads
        rows = self.batch * merged
        device = tensors[0].device
        self.capacity = tensors[6].shape[1] * 64
        self.head_tile = 64
        self.pv_head_tile = 64
        self.prequantized = False
        self.grouped = False
        self.quantized_cache = True
        self.qk_native_statistics = True
        self.native_barriers = True
        self.value_staging = True
        self.adaptive = merged <= 64
        self.sort_batches = self.batch <= 256
        self.value_splits = (
            2 if self.adaptive and self.capacity <= 32768 else 4 if self.adaptive else 1
        )
        self.uniform_value_splits = (
            2
            if self.adaptive
            and (self.capacity == 8192 or 16384 <= self.capacity <= 32768)
            else 1
        )
        self.value_merge_heads = 4 if self.adaptive else 0
        self.score_splits = (
            8 if merged == 64 or (merged <= 128 and self.capacity >= 16384) else 4
        )
        if 1 < self.batch <= 256:
            self.batch_order = torch.empty(
                (self.batch,), dtype=torch.int32, device=device
            )
            self.batch_order_kernel = dense_int8_batch_order
        self.scores = torch.empty(
            (self.batch, self.capacity // 64, merged, 64),
            dtype=torch.uint16,
            device=device,
        )
        self.cache_metadata = torch.empty(
            (self.batch, self.capacity // 64, 2, merged, 4),
            dtype=torch.float32,
            device=device,
        )
        self.stats = torch.empty(
            (self.batch, self.score_splits, 3, 4, merged),
            dtype=torch.float32,
            device=device,
        )
        self.probability_metadata = torch.empty(
            (rows, 2), dtype=torch.float32, device=device
        )
        self.probability_scale = torch.empty(
            (rows,), dtype=torch.float32, device=device
        )
        self.split_flag = torch.empty(
            (self.batch + 1,) if self.sort_batches else (),
            dtype=torch.int32,
            device=device,
        )
        workspace_splits = self.value_splits if self.adaptive else 0
        self.partial = torch.empty(
            (workspace_splits, rows, 512), dtype=torch.float32, device=device
        )
        self.partial_denominator = torch.empty(
            (workspace_splits, rows), dtype=torch.float32, device=device
        )
        self.score_kernel = ScoresKernel(self)
        self.value_kernel = ValuesKernel(256 if merged <= 64 else 512, self)

    def __call__(self):
        q, qr, kv, kr, qs, ks, table, lengths = self.tensors
        merged = self.queries * self.heads
        rows = self.batch * merged
        with torch_device_fn.device(q.device):
            self._prepare_batch_order()
            self.score_kernel[(self.score_splits, triton.cdiv(merged, 64), self.batch)](
                *self.tensors,
                self.scores,
                self.cache_metadata,
                self.stats,
                q.stride(),
                qr.stride(),
                kv.stride(),
                kr.stride(),
                qs.stride(),
                ks.stride(),
                table.stride(0),
                lengths.stride(0),
                self.heads,
                self.queries,
                self.capacity,
                self.scale,
                self.causal,
                BH=64,
                SPLITS=self.score_splits,
                COL_STATS=True,
                NATIVE_SYNC=True,
                BatchOrder=self.batch_order,
            )
            dense_int8_probability_merge[(triton.cdiv(rows, 32),)](
                self.stats,
                self.probability_metadata,
                self.probability_scale,
                self.lse,
                lengths,
                self.split_flag,
                lengths.stride(0),
                self.batch,
                triton.next_power_of_2(self.batch),
                self.heads,
                self.queries,
                rows,
                self.score_splits,
                self.adaptive,
                QUANTIZED_CACHE=True,
                SORT_BATCHES=self.sort_batches,
                num_warps=4,
                num_stages=1,
            )
            self.value_kernel[
                (2, triton.cdiv(merged, 64), self.batch * self.value_splits)
            ](
                kv,
                table,
                lengths,
                self.scores,
                self.cache_metadata,
                self.probability_metadata,
                self.probability_scale,
                self.output,
                self.lse,
                self.partial,
                self.partial_denominator,
                self.split_flag,
                kv.stride(),
                table.stride(0),
                lengths.stride(0),
                self.batch,
                self.heads,
                self.queries,
                self.capacity,
                BH=64,
                SPLITS=self.value_splits,
                ADAPTIVE=self.adaptive,
                UNIFORM_SPLITS=self.uniform_value_splits,
                QUANTIZED_CACHE=True,
                SORT_BATCHES=self.sort_batches,
                STAGED=True,
            )
            if self.adaptive:
                dense_int8_value_merge[(self.batch, triton.cdiv(merged, 4))](
                    self.partial,
                    self.partial_denominator,
                    self.probability_metadata,
                    self.probability_scale,
                    self.output,
                    self.lse,
                    self.split_flag,
                    self.batch,
                    self.heads,
                    self.queries,
                    self.value_splits,
                    QUANTIZED_CACHE=True,
                    UNIFORM_SPLITS=self.uniform_value_splits,
                    num_warps=4,
                    num_stages=1,
                )
        return self.output, self.lse


def prepare_flash_mla_with_kvcache_fwd_w8a8_int8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_lora: torch.Tensor,
    k_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int = 512,
    *,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    num_splits: Optional[int] = None,
) -> tuple[FlashMLAInt8PreparedHandle, tuple[torch.Tensor, torch.Tensor]]:
    """Decode 64-token pages; RoPE is divided by the content scale as in PR851.

    Active page IDs and lengths must be valid, scales finite and positive.
    Causal queries attend the aligned suffix of the valid cache.
    """
    if q_nope.ndim != 4 or q_nope.shape[1] not in (1, 2) or q_nope.shape[-1] != 512:
        raise NotImplementedError("requires Q [batch, 1 or 2, heads, 512]")
    if q_nope.device.type != "cuda":
        raise ValueError("requires MetaX device tensors")
    batch, queries, heads, _ = q_nope.shape
    if (
        k_lora.ndim not in (3, 4)
        or k_lora.shape[1] != 64
        or k_lora.shape[-1] != 512
        or (k_lora.ndim == 4 and k_lora.shape[2] != 1)
        or head_dim_v != 512
    ):
        raise NotImplementedError("requires KV [pages, 64, 512] and head_dim_v=512")
    expected_shapes = (
        (batch, queries, heads, 64),
        (*k_lora.shape[:-1], 64),
        (batch, queries, heads, 1),
        (*k_lora.shape[:-1], 1),
        (batch,),
    )
    for tensor, shape in zip(
        (q_rope, k_rope, q_scale, k_scale, cache_seqlens), expected_shapes
    ):
        if tensor.shape != shape:
            raise ValueError(f"expected shape {shape}, got {tuple(tensor.shape)}")
    if (
        block_table.ndim != 2
        or block_table.shape[0] != batch
        or block_table.shape[1] == 0
    ):
        raise ValueError("block_table must have shape [batch, positive page capacity]")
    if q_nope.dtype != torch.int8 or k_lora.dtype != torch.int8:
        raise TypeError("Q and KV content must be INT8")
    if (
        q_rope.dtype not in (torch.float16, torch.bfloat16)
        or k_rope.dtype != q_rope.dtype
    ):
        raise TypeError("RoPE tensors must have the same FP16 or BF16 dtype")
    if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32:
        raise TypeError("scales must be FP32")
    if block_table.dtype != torch.int32 or cache_seqlens.dtype != torch.int32:
        raise TypeError("block_table and cache_seqlens must be INT32")
    tensors = (
        q_nope,
        q_rope,
        k_lora,
        k_rope,
        q_scale,
        k_scale,
        block_table,
        cache_seqlens,
    )
    for tensor in tensors:
        if tensor.device != q_nope.device or (
            tensor.numel() != 0 and tensor.stride(-1) != 1
        ):
            raise ValueError(
                "all tensors must share a device and contiguous last dimension"
            )
    if not all(
        can_use_int32_index(tensor)
        for tensor in (q_nope, q_rope, q_scale, block_table, cache_seqlens)
    ):
        raise NotImplementedError("query and metadata offsets must fit signed INT32")
    if not isinstance(causal, bool):
        raise TypeError("causal must be a bool")
    if softmax_scale is not None and not isinstance(softmax_scale, (int, float)):
        raise TypeError("softmax_scale must be a Python number")
    scale = 576**-0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax_scale must be finite and positive")
    if num_splits is None:
        splits = min(
            16,
            triton.next_power_of_2(
                max(
                    1,
                    triton.cdiv(
                        4 * _get_num_sms(q_nope.device),
                        max(1, batch * triton.cdiv(queries * heads, 16)),
                    ),
                )
            ),
        )
        splits = min(splits, max(1, triton.next_power_of_2(block_table.shape[1])))
    else:
        splits = num_splits
    if (
        isinstance(splits, bool)
        or not isinstance(splits, int)
        or splits not in (1, 2, 4, 8, 16)
    ):
        raise ValueError("num_splits must be one of 1, 2, 4, 8, 16")
    output_bytes = batch * queries * heads * 512 * q_rope.element_size()
    partial_bytes = (splits if splits > 1 else 0) * batch * queries * heads * 512 * 4
    if max(output_bytes, partial_bytes) > 2**31 - 1:
        raise NotImplementedError(
            "output and partial workspaces must fit signed INT32 indexing"
        )
    capacity = block_table.shape[1] * 64
    merged_heads = queries * heads
    rows = batch * merged_heads
    # The native ABI uses contiguous rows, 64-token pages and aligned 128-bit copies.
    fast_path = (
        torch.cuda.get_device_capability(q_nope.device) == (8, 0)
        and "C550" in torch.cuda.get_device_name(q_nope.device)
        and num_splits is None
        and 0 < batch < 2**14
        and (
            merged_heads in (16, 32) or (merged_heads >= 64 and merged_heads % 64 == 0)
        )
        and triton.cdiv(merged_heads, 64) < 2**16
        and capacity <= 65536
        and batch * capacity < 2**31
        and all(tensor.is_contiguous() for tensor in tensors)
        and all(tensor.data_ptr() % 16 == 0 for tensor in tensors[:6])
        and not torch.cuda.is_current_stream_capturing()
    )
    if fast_path:
        native_splits = min(
            4 if merged_heads == 32 and capacity <= 8192 else 8,
            triton.next_power_of_2(block_table.shape[1]),
        )
        if merged_heads <= 32:
            workspace_bytes = (
                output_bytes + rows * 4 + native_splits * rows * (512 + 2) * 4
            )
            if merged_heads == 32 and 1 < batch <= 256:
                workspace_bytes += 4 * batch
        else:
            score_splits = (
                8
                if merged_heads == 64 or (merged_heads <= 128 and capacity >= 16384)
                else 4
            )
            workspace_bytes = (
                output_bytes
                + rows * capacity * 2
                + rows * (capacity // 64) * 8 * 4
                + rows * score_splits * 12 * 4
                + rows * 3 * 4
                + 4 * (batch + 1)
                + 4 * batch
            )
            if merged_heads <= 64:
                workspace_bytes += rows * 4 * (512 + 1) * 4
        free_bytes, _ = torch.cuda.mem_get_info(q_nope.device)
        free_bytes += torch.cuda.memory_reserved(
            q_nope.device
        ) - torch.cuda.memory_allocated(q_nope.device)
        fast_path = workspace_bytes <= min(8 * 1024**3, free_bytes // 2)
    if fast_path:
        handle_class = NativeSmallHandle if merged_heads <= 32 else NativeCachedHandle
        handle = handle_class(tensors, scale, causal)
    else:
        handle = FlashMLAInt8PreparedHandle(tensors, splits, scale, causal)
    return handle, handle()


def flash_mla_with_kvcache_fwd_w8a8_int8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_lora: torch.Tensor,
    k_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int = 512,
    *,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    num_splits: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, outputs = prepare_flash_mla_with_kvcache_fwd_w8a8_int8(
        q_nope,
        q_rope,
        k_lora,
        k_rope,
        q_scale,
        k_scale,
        block_table,
        cache_seqlens,
        head_dim_v,
        softmax_scale=softmax_scale,
        causal=causal,
        num_splits=num_splits,
    )
    return outputs
