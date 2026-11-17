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
import os
import struct
import subprocess
import tempfile
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import extern
from triton.runtime.cache import get_cache_manager

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry


@libentry()
@triton.jit
def merge_int8_partials(
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
    splits = tl.arange(0, triton.next_power_of_2(SPLITS))
    dims = tl.arange(0, 512)
    merged_heads = H * SQ
    maxima = tl.load(
        Stats + (splits * 2 * B + batch) * merged_heads + head,
        splits < SPLITS,
        -float("inf"),
    )
    maximum = tl.max(maxima, 0)
    safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp2(maxima - safe_maximum)
    sums = tl.load(
        Stats + ((splits * 2 + 1) * B + batch) * merged_heads + head, splits < SPLITS, 0
    )
    denominator = tl.sum(sums * weights, 0)
    values = tl.load(
        Partial
        + ((splits[:, None] * B + batch) * merged_heads + head) * 512
        + dims[None, :],
        splits[:, None] < SPLITS,
        0,
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
        self,
        tensors: tuple[torch.Tensor, ...],
        splits: int,
        scale: float,
        *,
        uses_raw_indices: bool = False,
    ):
        self.tensors = tensors
        self.uses_raw_indices = uses_raw_indices
        query, rope, _, _, _, _, _, _ = tensors
        self.batch, self.queries, self.heads, _ = query.shape
        self.head_tile = 16
        self.splits = splits
        self.scale = scale
        self.num_warps = 4
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
        # The single-split kernel only needs the grid's batch coordinate.
        batch_stride = self.batch if self.splits > 1 else 1
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
                batch_stride,
                self.heads,
                self.splits,
                self.scale,
                self.queries,
                False,
                BH=self.head_tile,
                num_warps=self.num_warps,
                num_stages=2,
                scenario="mla",
                **self.kernel_arguments,
            )
            if self.splits > 1:
                merge_int8_partials[(self.batch, self.queries * self.heads)](
                    self.partial,
                    self.stats,
                    self.output,
                    self.lse,
                    self.batch,
                    self.heads,
                    self.queries,
                    self.splits,
                    NORMALIZED=self.normalized_partials,
                    num_warps=2 if self.uses_raw_indices and rope.shape[-1] == 0 else 4,
                )
        return self.output, self.lse


@libentry()
@triton.jit
def merge_int8_probabilities(
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
def merge_int8_values(
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
def sort_batches_by_length(
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
    index_spec = ()
    if handle.uses_raw_indices:
        index_spec += (
            ("SPARSE_RAW", 1),
            ("CACHE_TOKENS", handle.tensors[2].shape[0] * 64),
        )
    return index_spec + (
        ("NO_ROPE", int(handle.tensors[1].shape[-1] == 0)),
        ("HQ", handle.heads),
        ("SQ", handle.queries),
        ("CAP", handle.tensors[6].shape[1] * 64),
        ("CAUSAL", 0),
        ("BF16", int(handle.tensors[1].dtype == torch.bfloat16)),
        ("SCALE", value.hex() + "f"),
    )


SMALL_SIGNATURE = (tl.int64,) * 16 + (tl.int32,) * 6 + (tl.float32,) + (tl.int32,) * 3


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
    SelectedScales=None,
    REVERSE_SPLITS: tl.constexpr = False,
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
        (
            tl.full((), 0, tl.int64)
            if SelectedScales is None
            else tl.cast(SelectedScales, tl.int64)
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
        SPLITS - 1 - tl.program_id(2) if REVERSE_SPLITS else tl.program_id(2),
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


@functools.lru_cache(maxsize=1)
def _bsm_offset_kernel():
    """Use BSM immediates for this specialization only on the validated toolchain."""
    from triton.backends.metax.compiler import get_lld_version
    from triton.compiler import CompiledKernel, compile
    from triton.runtime.jit import JITFunction

    compiler, version = _compiler()
    if b"(d9102a1572)" not in version or get_lld_version() != 19:
        return small_native

    def compile_with_offsets(src, target=None, options=None, **kwargs):
        standard = compile(src, target=target, options=options, **kwargs)
        if (
            "mcfatbin" not in standard.asm
            or "llir" not in standard.asm
            or target.backend != "maca"
            or options.get("scenario") != "mla"
            or options.get("pipeline") != "basic"
            or options.get("num_stages") != 2
            or options.get("debug")
            or options.get("instrumentation_mode")
            or os.environ.get("TRITON_COMPILER_OPT_PATH")
        ):
            return standard
        maca_path = os.environ.get("MACA_PATH", "/opt/maca")
        flags = (
            "-metaxgpu-mma-sched=true",
            "-map-use-pk-fma=1",
            "-metaxgpu-split-regalloc=true",
            "-metaxgpu-aggressive-fold=true",
            "-metaxgpu-disable-licm=true",
            "-metaxgpu-merge-copy-postra=false",
            "--vectorize-slp=false",
            "-metaxgpu-disable-bsm-offset=0",
            f"-inputFuncBC={maca_path}/lib/maca_kernellib.bc",
        )
        key = hashlib.sha256(
            standard.kernel + version + str(compiler).encode() + repr(flags).encode()
        ).hexdigest()
        entries = standard.metadata_group.copy()
        binary = next(name for name in entries if name.endswith(".mcfatbin"))
        cache = get_cache_manager(key)
        path = cache.get_file(binary)
        if path is None:
            llvm_flags = [flag for item in flags for flag in ("-mllvm", item)]
            with tempfile.TemporaryDirectory(prefix="flaggems-mla-bsm-") as work:
                llir = Path(work) / "kernel.ll"
                output = Path(work) / binary
                llir.write_text(standard.asm["llir"])
                subprocess.run(
                    [
                        str(compiler),
                        "--fatbin",
                        str(llir),
                        *llvm_flags,
                        "-maca-link",
                        "-input-is-device",
                        "--maca-path",
                        maca_path,
                        "-o",
                        str(output),
                    ],
                    check=True,
                )
                path = cache.put(output.read_bytes(), binary)
        entries[binary] = path
        return CompiledKernel(src, entries, key)

    class BsmOffsetKernel(JITFunction):
        def create_binder(self):
            binder = super().create_binder()
            self.compile = compile_with_offsets
            return binder

    return BsmOffsetKernel(small_native.fn)


class SmallKernel:
    def __init__(self, heads, handle):
        # Batch size only strides the split workspace when a merge is needed.
        batch_stride = handle.batch if handle.splits > 1 else 1
        spec = _scale_spec(handle) + (("B", batch_stride), ("SPLITS", handle.splits))
        if "SelectedScales" in handle.kernel_arguments:
            spec += (("SCALE_TABLE", 1),)
        pages = handle.tensors[6].shape[1]
        # With one remainder page, only the last partition is longer.
        self.reverse_splits = (
            handle.uses_raw_indices
            and handle.tensors[1].shape[-1] == 0
            and handle.batch <= 32
            and pages <= 16
            and 1 < handle.splits <= 6
            and pages % handle.splits == 1
        )
        self.fn = (
            _bsm_offset_kernel()
            if heads == 32
            and handle.uses_raw_indices
            and handle.tensors[1].shape[-1] == 0
            and torch.cuda.get_device_properties(handle.tensors[0].device).name
            == "MetaX C550"
            else small_native
        )
        if (
            self.fn is not small_native
            and "SelectedScales" in handle.kernel_arguments
            and handle.splits == 1
            and 3072 <= handle.batch <= 7168
            and 4096 <= pages * 64 <= 16384
        ):
            spec += (("PAGE_UNROLL", 3),)
        self.bitcode = _library("small", heads, spec)

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            kwargs.update(
                num_warps=4,
                num_stages=2,
                scenario="mla",
                extern_libs={"mla_int8_small": self.bitcode},
            )
            return self.fn[grid](*args, REVERSE_SPLITS=self.reverse_splits, **kwargs)

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
        if handle.uses_raw_indices:
            spec += (
                ("SPARSE_RAW", 1),
                ("CACHE_TOKENS", handle.tensors[2].shape[0] * 64),
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
    def __init__(self, tensors, scale, *, uses_raw_indices=False):
        capacity = tensors[6].shape[1] * 64
        splits = min(
            (
                4
                if tensors[0].shape[1] * tensors[0].shape[2] == 32 and capacity <= 8192
                else 8
            ),
            triton.next_power_of_2(tensors[6].shape[1]),
        )
        super().__init__(
            tensors,
            splits,
            scale,
            uses_raw_indices=uses_raw_indices,
        )
        self.head_tile = self.queries * self.heads
        self.normalized_partials = True
        if self.head_tile == 32 and 1 < self.batch <= 256:
            self.batch_order = torch.empty(
                (self.batch,), dtype=torch.int32, device=tensors[0].device
            )
            self.batch_order_kernel = sort_batches_by_length
        self.kernel_arguments = {"BatchOrder": self.batch_order}
        self.grid = (1, self.batch, self.splits)
        self.kernel = SmallKernel(self.head_tile, self)


class NativeCachedHandle(FlashMLAInt8PreparedHandle):
    def __init__(self, tensors, scale, *, uses_raw_indices=False):
        super().__init__(
            tensors,
            1,
            scale,
            uses_raw_indices=uses_raw_indices,
        )
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
            self.batch_order_kernel = sort_batches_by_length
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
                False,
                BH=64,
                SPLITS=self.score_splits,
                COL_STATS=True,
                NATIVE_SYNC=True,
                BatchOrder=self.batch_order,
            )
            merge_int8_probabilities[(triton.cdiv(rows, 32),)](
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
                merge_int8_values[(self.batch, triton.cdiv(merged, 4))](
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
