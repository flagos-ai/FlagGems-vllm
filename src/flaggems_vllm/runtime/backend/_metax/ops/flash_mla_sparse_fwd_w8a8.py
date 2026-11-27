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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.core import extern
from triton.runtime.cache import get_cache_manager

from flaggems_vllm.ops.flash_mla import _get_num_sms
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry

MAX_SPARSE_SPLITS = 24
LARGE_BATCH_ROWS = 256
MAX_SPARSE_ROWS = 16384
LENGTH_BLOCK = 256
COMPACT_BLOCK = 1024
MAX_WORKSPACE_BYTES = 8 * 1024**3


@dataclass(frozen=True)
class SparseMLATensors:
    query_nope: torch.Tensor
    query_rope: torch.Tensor
    kv_nope: torch.Tensor
    kv_rope: torch.Tensor
    query_scale: torch.Tensor
    kv_scale: torch.Tensor
    page_table: torch.Tensor
    lengths: torch.Tensor

    def as_kernel_args(self) -> tuple[torch.Tensor, ...]:
        return (
            self.query_nope,
            self.query_rope,
            self.kv_nope,
            self.kv_rope,
            self.query_scale,
            self.kv_scale,
            self.page_table,
            self.lengths,
        )


@functools.lru_cache(maxsize=None)
def is_metax_c550(device_index: int) -> bool:
    return "C550" in torch.cuda.get_device_name(device_index)


def device_index_of(tensor: torch.Tensor) -> int:
    index = tensor.device.index
    return torch.cuda.current_device() if index is None else index


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
    batch = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    splits = tl.arange(0, triton.next_power_of_2(SPLITS))
    splits_i64 = splits.to(tl.int64)
    dims = tl.arange(0, 512)
    merged_heads = H * SQ
    maxima = tl.load(
        Stats + (splits_i64 * 2 * B + batch) * merged_heads + head,
        splits < SPLITS,
        -float("inf"),
    )
    maximum = tl.max(maxima, 0)
    safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp2(maxima - safe_maximum)
    sums = tl.load(
        Stats + ((splits_i64 * 2 + 1) * B + batch) * merged_heads + head,
        splits < SPLITS,
        0,
    )
    denominator = tl.sum(sums * weights, 0)
    values = tl.load(
        Partial
        + ((splits_i64[:, None] * B + batch) * merged_heads + head) * 512
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
        tensors: SparseMLATensors,
        splits: int,
        scale: float,
        *,
        uses_raw_indices: bool = False,
        fp32_output: bool = False,
        allocate_split_workspace: bool = True,
    ):
        self.tensors = tensors
        self.uses_raw_indices = uses_raw_indices
        query = tensors.query_nope
        rope = tensors.query_rope
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
            dtype=torch.float32 if fp32_output else rope.dtype,
            device=query.device,
        )
        self.lse = torch.empty(
            (self.batch, self.heads, self.queries),
            dtype=torch.float32,
            device=query.device,
        )
        if allocate_split_workspace:
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
        else:
            self.partial = None
            self.stats = None

    def prepare_batch_order(self):
        if self.batch_order is not None:
            lengths = self.tensors.lengths
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
        query = self.tensors.query_nope
        rope = self.tensors.query_rope
        with torch_device_fn.device(query.device):
            self.prepare_batch_order()
            self.kernel[self.grid](
                *self.tensors.as_kernel_args(),
                self.partial,
                self.stats,
                self.output,
                self.lse,
                self.splits,
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
    SORT_BATCHES: tl.constexpr = False,
):
    rows = tl.program_id(0) * 32 + tl.arange(0, 32)
    positions = tl.arange(0, SPLITS * 4)
    row_batches = rows // (H * SQ)
    row_heads = rows % (H * SQ)
    offsets = (
        (row_batches[:, None].to(tl.int64) * SPLITS * 3 + (positions[None, :] // 4) * 3)
        * 4
        + positions[None, :] % 4
    ) * H * SQ + row_heads[:, None]
    maxima = tl.load(Stats + offsets, rows[:, None] < ROWS, -float("inf"))
    weighted_maxima = tl.load(Stats + offsets + 4 * H * SQ, rows[:, None] < ROWS, 0.0)
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
@triton.jit
def merge_int8_values(
    Partial,
    Scale,
    Output,
    SplitFlag,
    B: tl.constexpr,
    H: tl.constexpr,
    SQ: tl.constexpr,
    SPLITS: tl.constexpr,
    UNIFORM_SPLITS: tl.constexpr,
):
    is_variable = tl.load(SplitFlag) != 0
    if UNIFORM_SPLITS > 1 or is_variable:
        batch = tl.program_id(0)
        heads = tl.program_id(1) * 4 + tl.arange(0, 4)
        dims = tl.arange(0, 512)
        offsets = (batch.to(tl.int64) * H * SQ + heads[:, None]) * 512 + dims[None, :]
        split_count = tl.where(is_variable, SPLITS, UNIFORM_SPLITS)
        accumulator = tl.zeros((4, 512), tl.float32)
        # Native PV leaves inactive partition buffers unwritten.
        for split in tl.static_range(SPLITS):
            values = tl.load(
                Partial + split * B * H * SQ * 512 + offsets,
                (heads[:, None] < H * SQ) & (split < split_count),
                0.0,
            )
            accumulator += values
        scale = tl.load(Scale + batch * H * SQ + heads, heads < H * SQ, 0.0)
        tl.store(
            Output + offsets, accumulator * scale[:, None], heads[:, None] < H * SQ
        )


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
def compiler_info():
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "warm up flash_mla_sparse_fwd_w8a8 before CUDA graph capture"
        )
    path = Path(os.environ.get("MACA_PATH", "/opt/maca")) / "mxgpu_llvm/bin/mxcc"
    return path.resolve(), subprocess.check_output(
        [str(path), "--version"], stderr=subprocess.STDOUT
    )


@functools.lru_cache(maxsize=128)
def compile_stage_bitcode(stage, tile, spec):
    """Build one CTA-wide device function; cached results are safe during graph capture."""
    compiler, version = compiler_info()
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
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "warm up flash_mla_sparse_fwd_w8a8 before CUDA graph capture"
        )
    # A unique temporary directory also permits concurrent handle preparation.
    with tempfile.TemporaryDirectory(prefix="flaggems-mla-int8-") as temporary:
        output = Path(temporary) / filename
        subprocess.run(
            [str(compiler), *flags, str(source), "-o", str(output)], check=True
        )
        return cache.put(output.read_bytes(), filename)


def build_scale_spec(handle):
    value = struct.unpack("f", struct.pack("f", handle.scale * 1.4426950408889634))[0]
    index_spec = ()
    if handle.uses_raw_indices:
        index_spec += (
            ("SPARSE_RAW", 1),
            ("CACHE_TOKENS", handle.tensors.kv_nope.shape[0] * 64),
        )
    return index_spec + (
        ("NO_ROPE", int(handle.tensors.query_rope.shape[-1] == 0)),
        ("HQ", handle.heads),
        ("SQ", handle.queries),
        ("CAP", handle.tensors.page_table.shape[1] * 64),
        ("BF16", int(handle.tensors.query_rope.dtype == torch.bfloat16)),
        ("SCALE", value.hex() + "f"),
    )


SMALL_SIGNATURE = (tl.int64,) * 14 + (tl.int32,) * 3


@extern
def small_device_extern(*args, _semantic=None):
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
    SPLITS: tl.constexpr,
    BatchOrder=None,
    SelectedScales=None,
    REVERSE_SPLITS: tl.constexpr = False,
):
    small_device_extern(
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
        tl.program_id(0),
        tl.program_id(1),
        SPLITS - 1 - tl.program_id(2) if REVERSE_SPLITS else tl.program_id(2),
    )


SCORES_SIGNATURE = (tl.int64,) * 12 + (tl.int32,) * 3


@extern
def scores_device_extern(*args, _semantic=None):
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
    BatchOrder=None,
):
    scores_device_extern(
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
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )


VALUES_SIGNATURE = (tl.int64,) * 10 + (tl.int32,) * 3


@extern
def values_device_extern(*args, _semantic=None):
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
    Partial,
    SplitFlag,
):
    values_device_extern(
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
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )


@functools.lru_cache(maxsize=1)
def bsm_offset_kernel():
    """Use BSM immediates for this specialization only on the validated toolchain."""
    from triton.backends.metax.compiler import get_lld_version
    from triton.compiler import CompiledKernel, compile
    from triton.runtime.jit import JITFunction

    compiler, version = compiler_info()
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
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "warm up flash_mla_sparse_fwd_w8a8 before CUDA graph capture"
                )
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
        spec = build_scale_spec(handle) + (
            ("B", batch_stride),
            ("SPLITS", handle.splits),
        )
        if "SelectedScales" in handle.kernel_arguments:
            spec += (("SCALE_TABLE", 1),)
        if handle.output.dtype == torch.float32:
            spec += (("FP32_OUTPUT", 1),)
        pages = handle.tensors.page_table.shape[1]
        # With one remainder page, only the last partition is longer.
        self.reverse_splits = (
            handle.uses_raw_indices
            and handle.tensors.query_rope.shape[-1] == 0
            and handle.batch <= 32
            and pages <= 16
            and 1 < handle.splits <= 6
            and pages % handle.splits == 1
        )
        self.fn = (
            bsm_offset_kernel()
            if heads == 32
            and handle.uses_raw_indices
            and (
                handle.tensors.query_rope.shape[-1] == 0
                or "SelectedScales" in handle.kernel_arguments
            )
            and is_metax_c550(device_index_of(handle.tensors.query_nope))
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
        self.bitcode = compile_stage_bitcode("small", heads, spec)

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
        spec = build_scale_spec(handle) + (("SPLITS", handle.score_splits),)
        self.bitcode = compile_stage_bitcode("scores", 64, spec)

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
        if handle.output.dtype == torch.float32:
            spec += (("FP32_OUTPUT", 1),)
        if handle.uses_raw_indices:
            spec += (
                ("SPARSE_RAW", 1),
                ("CACHE_TOKENS", handle.tensors.kv_nope.shape[0] * 64),
            )
        self.bitcode = compile_stage_bitcode("values", dims, spec)

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
    def __init__(
        self,
        tensors: SparseMLATensors,
        scale: float,
        *,
        uses_raw_indices: bool = False,
        fp32_output: bool = False,
    ):
        capacity = tensors.page_table.shape[1] * 64
        splits = min(
            (
                4
                if tensors.query_nope.shape[1] * tensors.query_nope.shape[2] == 32
                and capacity <= 8192
                else 8
            ),
            triton.next_power_of_2(tensors.page_table.shape[1]),
        )
        super().__init__(
            tensors,
            splits,
            scale,
            uses_raw_indices=uses_raw_indices,
            fp32_output=fp32_output,
        )
        self.head_tile = self.queries * self.heads
        self.normalized_partials = True
        if self.head_tile == 32 and 1 < self.batch <= 256:
            self.batch_order = torch.empty(
                (self.batch,), dtype=torch.int32, device=tensors.query_nope.device
            )
            self.batch_order_kernel = sort_batches_by_length
        self.kernel_arguments = {"BatchOrder": self.batch_order}
        self.grid = (1, self.batch, self.splits)
        self.kernel = SmallKernel(self.head_tile, self)


class NativeCachedHandle(FlashMLAInt8PreparedHandle):
    def __init__(
        self,
        tensors: SparseMLATensors,
        scale: float,
        *,
        uses_raw_indices: bool = False,
        fp32_output: bool = False,
    ):
        super().__init__(
            tensors,
            1,
            scale,
            uses_raw_indices=uses_raw_indices,
            fp32_output=fp32_output,
            allocate_split_workspace=False,
        )
        merged = self.queries * self.heads
        rows = self.batch * merged
        device = tensors.query_nope.device
        self.capacity = tensors.page_table.shape[1] * 64
        self.head_tile = 64
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
        self.score_kernel = ScoresKernel(self)
        self.value_kernel = ValuesKernel(256 if merged <= 64 else 512, self)

    def __call__(self):
        tensors = self.tensors
        merged = self.queries * self.heads
        rows = self.batch * merged
        with torch_device_fn.device(tensors.query_nope.device):
            self.prepare_batch_order()
            self.score_kernel[(self.score_splits, triton.cdiv(merged, 64), self.batch)](
                *tensors.as_kernel_args(),
                self.scores,
                self.cache_metadata,
                self.stats,
                BatchOrder=self.batch_order,
            )
            merge_int8_probabilities[(triton.cdiv(rows, 32),)](
                self.stats,
                self.probability_metadata,
                self.probability_scale,
                self.lse,
                tensors.lengths,
                self.split_flag,
                tensors.lengths.stride(0),
                self.batch,
                triton.next_power_of_2(self.batch),
                self.heads,
                self.queries,
                rows,
                self.score_splits,
                self.adaptive,
                SORT_BATCHES=self.sort_batches,
                num_warps=4,
                num_stages=1,
            )
            self.value_kernel[
                (2, triton.cdiv(merged, 64), self.batch * self.value_splits)
            ](
                tensors.kv_nope,
                tensors.page_table,
                tensors.lengths,
                self.scores,
                self.cache_metadata,
                self.probability_metadata,
                self.probability_scale,
                self.output,
                self.partial,
                self.split_flag,
            )
            if self.adaptive:
                merge_int8_values[(self.batch, triton.cdiv(merged, 4))](
                    self.partial,
                    self.probability_scale,
                    self.output,
                    self.split_flag,
                    self.batch,
                    self.heads,
                    self.queries,
                    self.value_splits,
                    UNIFORM_SPLITS=self.uniform_value_splits,
                    num_warps=4,
                    num_stages=1,
                )
        return self.output, self.lse


def c550_small_route(rows, heads, capacity, rope, raw_indices):
    if not raw_indices or heads != 64 or rope.shape[-1] != 0:
        return None
    if rows == 16 and 512 <= capacity <= 1024:
        route = "prefetch_scales"
    elif rows == 8 and 4096 <= capacity <= 16384:
        route = "even_splits"
    else:
        return None
    return route if is_metax_c550(device_index_of(rope)) else None


@libentry()
@triton.jit
def initialize_sparse_lengths(
    Counts, ROWS: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Counts + offsets, K, offsets < ROWS)


@libentry()
@triton.jit
def gather_sparse_scales(
    Indices,
    Scales,
    Selected,
    ELEMENTS: tl.constexpr,
    TOKENS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    ids = tl.load(Indices + offsets, offsets < ELEMENTS, -1)
    valid = (ids >= 0) & (ids < TOKENS) & (offsets < ELEMENTS)
    weights = tl.load(Scales + ids, valid, -1.0)
    tl.store(Selected + offsets, weights, offsets < ELEMENTS)


@libentry()
@triton.jit
def compact_sparse_indices(
    Indices,
    Lengths,
    Table,
    Counts,
    K: tl.constexpr,
    CAP: tl.constexpr,
    TOKENS: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    length = tl.minimum(tl.maximum(tl.load(Lengths + row), 0), K) if HAS_LENGTH else K
    running_count = tl.zeros((), dtype=tl.int32)
    for tile_start in range(0, CAP, BLOCK):
        positions = tile_start + offsets
        selected = tl.load(Indices + row * K + positions, positions < K, -1)
        valid = (positions < length) & (selected >= 0) & (selected < TOKENS)
        ranks = running_count + tl.cumsum(valid.to(tl.int32)) - 1
        tl.store(Table + row * CAP + ranks, selected, valid)
        running_count += tl.sum(valid.to(tl.int32), 0)
    for tile_start in range(0, CAP, BLOCK):
        positions = tile_start + offsets
        tl.store(
            Table + row * CAP + positions,
            0,
            (positions >= running_count) & (positions < CAP),
        )
    tl.store(Counts + row, running_count)


@libentry()
@triton.jit
def sparse_output_epilogue(
    Source,
    LSE,
    Sink,
    Output,
    H: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    dims = tl.arange(0, 512)
    base = (row * H + head) * 512 + dims
    values = tl.load(Source + base)
    if HAS_SINK:
        lse = tl.load(LSE + row * H + head)
        sink = tl.load(Sink + head)
        # Empty rows use +inf LSE and must remain zero even for an infinite sink.
        delta = sink - lse
        ratio = tl.exp(-tl.abs(delta))
        weight = tl.where(delta > 0, ratio, 1.0) / (1.0 + ratio)
        values *= tl.where(lse == float("inf"), 0.0, weight)
    tl.store(Output + base, values)


class SparseMLAInt8PreparedHandle:
    """Reuse workspace; calls must be serialized on one stream. Indices may change."""

    def __init__(
        self, tensors: tuple[torch.Tensor, ...], lengths, sink, out, scale, single_split
    ):
        q, qr, kv, kr, qs, ks, indices = tensors
        self.batch, self.queries, self.heads, _ = q.shape
        self.indices, self.lengths, self.sink = indices, lengths, sink
        self.capacity = max(64, triton.cdiv(indices.shape[-1], 64) * 64)
        rows = self.batch * self.queries
        self.tokens = kv.shape[0] * 64
        self.uses_raw_indices = (
            indices.shape[-1] > 0
            and indices.shape[-1] % 64 == 0
            and lengths is None
            and self.tokens > 0
        )
        self.counts = torch.empty((rows,), dtype=torch.int32, device=q.device)
        if self.uses_raw_indices:
            self.table = indices.view(rows, self.capacity // 64, 64)
            if rows > 0 and self.heads > 0:
                with torch_device_fn.device(q.device):
                    initialize_sparse_lengths[(triton.cdiv(rows, LENGTH_BLOCK),)](
                        self.counts,
                        rows,
                        indices.shape[-1],
                        LENGTH_BLOCK,
                        num_warps=4,
                    )
        else:
            self.table = torch.empty(
                (rows, self.capacity // 64, 64), dtype=torch.int32, device=q.device
            )
        self.needs_epilogue = sink is not None or out is not None
        self.native = None
        self.selected_scales = None
        if rows > 0 and self.heads > 0:
            native_tensors = SparseMLATensors(
                query_nope=q.view(rows, 1, self.heads, 512),
                query_rope=qr.view(rows, 1, self.heads, qr.shape[-1]),
                kv_nope=kv,
                kv_rope=kr,
                query_scale=qs.view(rows, 1, self.heads, 1),
                kv_scale=ks,
                page_table=self.table,
                lengths=self.counts,
            )
            if self.heads <= 32:
                self.native = NativeSmallHandle(
                    native_tensors,
                    scale,
                    uses_raw_indices=self.uses_raw_indices,
                    fp32_output=sink is not None,
                )
            elif rows <= 32 or single_split:
                route = c550_small_route(
                    rows, self.heads, self.capacity, qr, self.uses_raw_indices
                )
                prefetch_scales = route == "prefetch_scales"
                if single_split:
                    # Each CTA completes its output tile without writing split partials.
                    tile, splits = 32, 1
                elif self.heads == 64 and self.capacity >= 4096:
                    tile = 32
                    # Fill two CTA slots per SM without excessive partial-buffer traffic.
                    resident_splits = max(
                        1,
                        2
                        * _get_num_sms(q.device)
                        // (rows * triton.cdiv(self.heads, tile)),
                    )
                    splits = min(
                        MAX_SPARSE_SPLITS, self.capacity // 64, resident_splits
                    )
                    if route == "even_splits" and splits > 1:
                        splits -= splits % 2
                elif self.heads == 64 and self.capacity <= 1024:
                    tile = 32 if prefetch_scales else 16
                    # Use one CTA slot per SM with the wider prefetched tile.
                    resident_splits = max(
                        1,
                        (1 if prefetch_scales else 2)
                        * _get_num_sms(q.device)
                        // (rows * triton.cdiv(self.heads, tile)),
                    )
                    splits = (
                        1
                        if self.capacity <= 128
                        else min(self.capacity // 64, resident_splits)
                    )
                else:
                    tile = 16 if rows <= 8 and self.capacity <= 1024 else 32
                    target_programs = (
                        256 if self.capacity >= 4096 else min(128, 16 * rows)
                    )
                    splits = min(
                        16,
                        triton.next_power_of_2(self.capacity // 64),
                        triton.next_power_of_2(
                            max(
                                1,
                                triton.cdiv(
                                    target_programs,
                                    rows * triton.cdiv(self.heads, tile),
                                ),
                            )
                        ),
                    )
                self.native = FlashMLAInt8PreparedHandle(
                    native_tensors,
                    splits,
                    scale,
                    uses_raw_indices=self.uses_raw_indices,
                    fp32_output=sink is not None,
                )
                self.native.head_tile = tile
                self.native.normalized_partials = True
                self.native.grid = (triton.cdiv(self.heads, tile), rows, splits)
                if (
                    self.uses_raw_indices
                    and (
                        qr.shape[-1] == 0
                        or (single_split and self.heads == 64 and self.capacity == 2048)
                    )
                    and (
                        single_split
                        or prefetch_scales
                        or (rows <= 32 and self.heads == 64 and self.capacity >= 4096)
                    )
                ):
                    self.selected_scales = torch.empty(
                        (rows, self.capacity), dtype=torch.float32, device=q.device
                    )
                    self.native.kernel_arguments["SelectedScales"] = (
                        self.selected_scales
                    )
                self.native.kernel = SmallKernel(tile, self.native)
            else:
                self.native = NativeCachedHandle(
                    native_tensors,
                    scale,
                    uses_raw_indices=self.uses_raw_indices,
                    fp32_output=sink is not None,
                )
            if out is not None:
                self.output = out
            elif sink is not None:
                self.output = torch.empty(q.shape, dtype=qr.dtype, device=q.device)
            else:
                self.output = self.native.output.view(
                    self.batch, self.queries, self.heads, 512
                )
            self.lse = self.native.lse.view(
                self.batch, self.queries, self.heads
            ).transpose(1, 2)
        else:
            self.output = (
                out
                if out is not None
                else torch.empty(
                    (self.batch, self.queries, self.heads, 512),
                    dtype=qr.dtype,
                    device=q.device,
                )
            )
            self.lse = torch.empty(
                (self.batch, self.heads, self.queries),
                dtype=torch.float32,
                device=q.device,
            )

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.native is None:
            return self.output, self.lse
        with torch_device_fn.device(self.indices.device):
            if not self.uses_raw_indices:
                compact_block = min(
                    COMPACT_BLOCK, triton.next_power_of_2(self.capacity)
                )
                compact_sparse_indices[(self.batch * self.queries,)](
                    self.indices,
                    self.lengths,
                    self.table,
                    self.counts,
                    self.indices.shape[-1],
                    self.capacity,
                    self.tokens,
                    self.lengths is not None,
                    compact_block,
                    num_warps=4,
                )
            if self.selected_scales is not None:
                count = self.indices.numel()
                # Prepared calls can receive updated indices and scale values.
                block = 128 if self.batch * self.queries <= 32 else 512
                gather_sparse_scales[(triton.cdiv(count, block),)](
                    self.indices,
                    self.native.tensors.kv_scale,
                    self.selected_scales,
                    count,
                    self.tokens,
                    block,
                    num_warps=2 if block == 128 else 4,
                )
            native_output, _ = self.native()
            if self.needs_epilogue:
                sparse_output_epilogue[(self.batch * self.queries, self.heads)](
                    native_output,
                    self.native.lse,
                    self.sink,
                    self.output,
                    self.heads,
                    self.sink is not None,
                    num_warps=4,
                )
        return self.output, self.lse


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
) -> tuple[SparseMLAInt8PreparedHandle, tuple[torch.Tensor, torch.Tensor]]:
    """Sparse INT8 MLA over physical page64 cache indices.

    RoPE is divided by the corresponding content scale.
    Negative/out-of-range indices are ignored; duplicates remain independent.
    Scales must be finite and positive. LSE excludes the optional attention sink.
    """
    if q_nope.ndim != 4 or q_nope.shape[1] not in (1, 2) or q_nope.shape[-1] != 512:
        raise NotImplementedError("requires Q [batch, 1 or 2, heads, 512]")
    batch, queries, heads, _ = q_nope.shape
    if heads not in (0, 16, 32) and (heads < 64 or heads % 64 != 0):
        raise NotImplementedError("requires 16, 32, or a multiple of 64 query heads")
    if (
        k_nope.ndim not in (3, 4)
        or k_nope.shape[1] != 64
        or k_nope.shape[-1] != 512
        or (k_nope.ndim == 4 and k_nope.shape[2] != 1)
    ):
        raise NotImplementedError("requires KV [pages, 64, 512] or [pages, 64, 1, 512]")
    if q_nope.dtype != torch.int8 or k_nope.dtype != torch.int8:
        raise TypeError("Q and KV content must be INT8")
    if (
        q_rope.dtype not in (torch.bfloat16, torch.float16)
        or k_rope.dtype != q_rope.dtype
    ):
        raise TypeError("RoPE must have matching BF16 or FP16 dtype")
    if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32:
        raise TypeError("scales must be FP32")
    if (
        indices.ndim != 3
        or indices.shape[:2] != (batch, queries)
        or indices.shape[-1] > 65536
    ):
        raise ValueError("indices must have shape [batch, queries, topk <= 65536]")
    if indices.dtype != torch.int32:
        raise TypeError("indices must be INT32")
    if (
        q_rope.ndim != 4
        or q_rope.shape[-1] not in (0, 64)
        or q_rope.shape != (batch, queries, heads, q_rope.shape[-1])
        or k_rope.shape != (*k_nope.shape[:-1], q_rope.shape[-1])
        or q_scale.shape != (batch, queries, heads, 1)
        or k_scale.shape != (*k_nope.shape[:-1], 1)
    ):
        raise ValueError(
            "RoPE/scales must match content rows with trailing dimension 0 or 64 / 1"
        )
    tensors = (q_nope, q_rope, k_nope, k_rope, q_scale, k_scale, indices)
    optional = tuple(t for t in (topk_length, attn_sink, out) if t is not None)
    if q_nope.device.type != "cuda" or any(
        t.device != q_nope.device for t in tensors + optional
    ):
        raise ValueError("all inputs must share a MetaX device")
    if any(not t.is_contiguous() for t in tensors + optional):
        raise NotImplementedError("requires contiguous inputs and output")
    if any(t.numel() and t.data_ptr() % 16 for t in tensors[:6]):
        raise NotImplementedError("requires 16-byte aligned content, RoPE and scales")
    if topk_length is not None and queries == 1 and topk_length.shape == (batch,):
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
    if (
        out is not None
        and out.numel()
        and any(
            out.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
            for tensor in tensors + optional[:-1]
        )
    ):
        raise ValueError("out must not alias an input")
    if softmax_scale is not None and (
        isinstance(softmax_scale, bool) or not isinstance(softmax_scale, (int, float))
    ):
        raise TypeError("softmax_scale must be a Python number")
    scale = (
        (512 + q_rope.shape[-1]) ** -0.5
        if softmax_scale is None
        else float(softmax_scale)
    )
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax_scale must be finite and positive")
    if not is_metax_c550(device_index_of(q_nope)):
        raise NotImplementedError("native sparse INT8 MLA requires MetaX C550")
    rows = batch * queries
    capacity = max(64, triton.cdiv(indices.shape[-1], 64) * 64)
    uses_raw_indices = (
        indices.shape[-1] > 0
        and indices.shape[-1] % 64 == 0
        and topk_length is None
        and k_nope.shape[0] > 0
    )
    output_bytes = (
        (6 if out is None else 4)
        if attn_sink is not None
        else (2 if out is None else 0)
    )
    base_workspace = rows * (
        4 + (0 if uses_raw_indices else capacity * 4) + heads * (512 * output_bytes + 4)
    )
    if heads <= 32:
        small_splits = min(
            4 if heads == 32 and capacity <= 8192 else 8,
            triton.next_power_of_2(capacity // 64),
        )
        split_workspace = (
            (small_splits if small_splits > 1 else 0) * rows * heads * (512 * 4 + 2 * 4)
        )
        single_split = False
        fits = base_workspace + split_workspace <= MAX_WORKSPACE_BYTES
    else:
        cached_workspace = rows * (capacity * (3 * heads + 4) + heads * 8192)
        cached_fits = (
            rows < MAX_SPARSE_ROWS
            and rows * capacity * max(heads, 1) < 2**30
            and cached_workspace <= MAX_WORKSPACE_BYTES
        )
        # Fusing QK and PV reuses the gathered K tile and avoids cached-score traffic.
        aligned_rope = (
            heads == 64
            and q_rope.shape[-1] == 64
            and indices.shape[-1] == 2048
            and topk_length is None
        )
        single_split = (rows > 32 and aligned_rope) or (
            rows > LARGE_BATCH_ROWS
            and heads >= 64
            and (q_rope.shape[-1] == 0 or not cached_fits)
        )
        if single_split:
            scale_table_bytes = rows * capacity * 4 if uses_raw_indices else 0
            fits = base_workspace + scale_table_bytes <= MAX_WORKSPACE_BYTES
        elif rows <= 32:
            small_batch_workspace = (
                base_workspace
                + MAX_SPARSE_SPLITS * rows * heads * (512 * 4 + 2 * 4)
                + rows * capacity * 4
            )
            fits = small_batch_workspace <= MAX_WORKSPACE_BYTES
        else:
            fits = cached_fits
    if rows > MAX_SPARSE_ROWS or k_nope.shape[0] * 64 >= 2**31 or not fits:
        raise NotImplementedError(
            "native sparse workspace exceeds its indexing or 8 GiB budget"
        )
    handle = SparseMLAInt8PreparedHandle(
        tensors, topk_length, attn_sink, out, scale, single_split
    )
    return handle, handle()


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse MLA on MetaX C550 with signed INT8 NoPE and FP16/BF16 RoPE."""
    _, output = prepare_flash_mla_sparse_fwd_w8a8(
        q_nope,
        q_rope,
        k_cache_lora,
        k_cache_rope,
        q_scale,
        k_scale,
        indices,
        softmax_scale=softmax_scale,
        topk_length=topk_length,
        attn_sink=attn_sink,
    )
    return output
