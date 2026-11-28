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

"""PPU MegaMHC host dispatch, validation, and lazy SDK bindings."""

import ctypes
import functools
import hashlib
import math
import os
import subprocess
import tempfile
from pathlib import Path

import torch
import triton
from triton.runtime.cache import get_cache_manager

from flaggems_vllm.runtime import device, torch_device_fn
from flaggems_vllm.runtime.backend._thead.ops.megamhc_kernel import mega_mhc_kernel
from flaggems_vllm.utils.triton_version_utils import has_triton_tle_attrs


@functools.lru_cache(maxsize=1)
def counter_memset():
    """Bind the stream-ordered PPU counter initialization."""
    try:
        memset = ctypes.CDLL("libhggc.so").hgMemsetD8Async
    except (OSError, AttributeError):
        return None
    memset.argtypes = [
        ctypes.c_uint64,
        ctypes.c_ubyte,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    memset.restype = ctypes.c_int

    return memset


@functools.lru_cache(maxsize=1)
def device_bitcode():
    """Compile the optional PPU device helpers for the fused pipeline."""
    sdk = Path(os.environ.get("PPU_SDK", "/usr/local/PPU_SDK"))
    compiler = sdk / "bin" / "clang++"
    headers = sdk / "CUDA_SDK" / "include"
    if not compiler.is_file() or not (headers / "hggc_runtime.h").is_file():
        return None
    source = Path(__file__).with_name("megamhc_device.cu")
    compiler_stat = compiler.stat()
    digest = hashlib.sha256(
        source.read_bytes()
        + f"{compiler.resolve()}:{compiler_stat.st_size}:{compiler_stat.st_mtime_ns}".encode()
    )
    version_file = sdk / "VERSION.txt"
    if version_file.is_file():
        digest.update(version_file.read_bytes())
    cache = get_cache_manager(digest.hexdigest())
    name = "megamhc-ppu001.bc"
    path = cache.get_file(name)
    if path is None:
        with tempfile.TemporaryDirectory(dir=cache.cache_dir) as temporary:
            bitcode = Path(temporary) / name
            try:
                subprocess.run(
                    [
                        str(compiler),
                        "-x",
                        "hggc",
                        "--hggc-device-only",
                        "--cuda-gpu-arch=ppu001",
                        "-std=c++17",
                        "-O2",
                        "-ffp-contract=off",
                        "-fgpu-rdc",
                        "-emit-llvm",
                        "-c",
                        "-I",
                        str(sdk / "include"),
                        "-I",
                        str(headers),
                        "-Xclang",
                        "-fno-cuda-host-device-constexpr",
                        "-D___HGGC_CG_HELPER_H___",
                        "-D__CUDACC_RDC__",
                        str(source),
                        "-o",
                        str(bitcode),
                    ],
                    check=True,
                    capture_output=True,
                )
            except (OSError, subprocess.CalledProcessError):
                return None
            path = cache.put(bitcode.read_bytes(), name, binary=True)
    return path


def clear_counters(counters, memset):
    stream = torch.cuda.current_stream(counters.device)
    status = memset(
        counters.data_ptr(),
        0,
        counters.numel() * counters.element_size(),
        stream.cuda_stream,
    )
    if status:
        raise RuntimeError(
            f"MegaMHC counter initialization failed with status {status}"
        )


_PROJECTION_BLOCK_K = 64
_PROJECTION_MAX_SPLITS = 8
# Native finish caches eight BF16x8 vectors per lane in a 32-lane warp.
_SIMT_CACHED_H_PER_WARP = 8 * 32 * 8


def _simt_launch_config(tokens, hidden, mixes, multiprocessors):
    """Choose complete and half row tiles from occupancy and cache capacity."""
    block_mix = triton.next_power_of_2(mixes)
    if tokens <= 2 * multiprocessors:
        rows, warps = 2, max(8, block_mix)
    elif tokens <= 4 * multiprocessors:
        rows, warps = 2, max(8, block_mix // 2)
    else:
        rows, warps = 4, max(8, block_mix // 4)
    blocks = triton.cdiv(tokens, rows)
    large_blocks = 0
    # Cached row tiles can fill incomplete waves without increasing
    # register use. Wider tiles retain their FP32 weight reuse.
    target_blocks = 2 * multiprocessors
    cached_hidden = _SIMT_CACHED_H_PER_WARP * (warps // rows)
    if (
        rows > 2
        and warps < mixes
        and hidden <= cached_hidden
        and blocks < target_blocks
    ):
        half_rows = rows // 2
        large_blocks = triton.cdiv(tokens - target_blocks * half_rows, rows - half_rows)
        blocks = target_blocks
    return rows, warps, blocks, large_blocks


def _projection_splits(k):
    return min(_PROJECTION_MAX_SPLITS, triton.cdiv(k, _PROJECTION_BLOCK_K))


def _packed_weight_shape(q, k):
    splits = _projection_splits(k)
    if splits > 1 and k % (splits * _PROJECTION_BLOCK_K) == 0:
        return (3, splits, q, k // splits)
    return (3, q, k)


def pack_mega_mhc_weights(fn):
    """Prepare the three BF16 projection components before Graph capture."""
    if device.vendor_name != "thead" or fn.device.type != "cuda":
        raise NotImplementedError("MegaMHC requires T-Head PPU")
    if (
        fn.ndim != 2
        or not all(fn.shape)
        or fn.dtype != torch.float32
        or not fn.is_contiguous()
        or fn.requires_grad
    ):
        raise ValueError("expected a nonempty contiguous FP32 inference weight matrix")

    from flaggems_vllm.runtime.backend._thead.ops.megamhc_kernel import (
        pack_mega_mhc_weights_kernel,
    )

    p, k = fn.shape
    q = max(16, triton.next_power_of_2(p))
    shape = _packed_weight_shape(q, k)
    packed = torch.empty(shape, dtype=torch.bfloat16, device=fn.device)
    packed_splits = shape[1] if len(shape) == 4 else 1
    block = 2048
    with torch_device_fn.device(fn.device):
        pack_mega_mhc_weights_kernel[(triton.cdiv(q * k, block),)](
            fn,
            packed,
            k,
            p,
            q,
            block,
            packed_splits,
            num_warps=2,
            enable_fp_fusion=False,
        )
    return packed


def mega_mhc(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    tile_n=1,
    norm_weight=None,
    norm_eps=1e-6,
    *,
    packed_fn=None,
    implementation="auto",
):
    """Fuse the normal mHC post-to-pre boundary for BF16 inference on PPU.

    Inputs may have arbitrary leading batch dimensions. Combination weights
    index [..., source stream, destination stream]. Tiling arguments are retained
    for vLLM compatibility; dispatch selects its own tiling.
    Returns the BF16 updated residual, FP32 post and combination coefficients,
    and the BF16 collapsed layer input with optional RMS normalization.

    For fixed weights, pass the result of pack_mega_mhc_weights(fn) as packed_fn
    to the large-input pipeline. Keep fn and its packed weights consistent and
    keep packed_fn alive for the lifetime of any graph that captures it.
    Small grids read fn directly. The default implementation reuses FP32
    weights across multiple rows in a SIMD tile. Omit packed_fn when updating fn
    between graph replays. Use implementation="triton" to run the fused
    pipeline with Triton postprocessing and CTA synchronization.
    """
    if implementation not in ("auto", "triton"):
        raise ValueError("MegaMHC implementation must be 'auto' or 'triton'")
    if device.vendor_name != "thead" or residual.device.type != "cuda":
        raise NotImplementedError("MegaMHC requires T-Head PPU")
    if residual.ndim < 2:
        raise ValueError("residual must have shape [..., streams, hidden]")
    outer = residual.shape[:-2]
    streams, hidden = residual.shape[-2:]
    tokens = math.prod(outer)
    if tokens >= 2**31:
        raise NotImplementedError("MegaMHC token count exceeds the PPU grid limit")
    if streams not in (1, 2, 4, 8) or not 0 < hidden <= 16384:
        raise NotImplementedError(
            "MegaMHC supports 1/2/4/8 streams and hidden sizes up to 16384"
        )
    if not isinstance(sinkhorn_repeat, int) or sinkhorn_repeat < 1:
        raise ValueError("sinkhorn_repeat must be positive")
    if not all(
        math.isfinite(v) and v >= 0
        for v in (rms_eps, hc_pre_eps, hc_sinkhorn_eps, norm_eps)
    ):
        raise ValueError("MegaMHC epsilons must be finite and nonnegative")
    if not math.isfinite(hc_post_mult_value):
        raise ValueError("hc_post_mult_value must be finite")

    mixes = streams * (streams + 2)
    specs = [
        (x, (*outer, hidden), torch.bfloat16),
        (residual, (*outer, streams, hidden), torch.bfloat16),
        (fn, (mixes, streams * hidden), torch.float32),
        (hc_scale, (3,), torch.float32),
        (hc_base, (mixes,), torch.float32),
    ]
    if post_layer_mix.shape not in ((*outer, streams), (*outer, streams, 1)):
        raise ValueError("invalid MegaMHC post weights")
    specs.append((post_layer_mix, post_layer_mix.shape, torch.float32))
    for tensor, shape, dtype in specs:
        if tensor.shape != shape or tensor.dtype != dtype or not tensor.is_contiguous():
            raise ValueError(
                f"MegaMHC expected a contiguous {dtype} tensor of shape {shape}"
            )
        if tensor.device != residual.device or tensor.requires_grad:
            raise ValueError("MegaMHC requires inference tensors on one PPU")
    if (
        comb_res_mix.shape != (*outer, streams, streams)
        or comb_res_mix.dtype != torch.float32
        or comb_res_mix.device != residual.device
        or comb_res_mix.requires_grad
    ):
        raise ValueError("invalid MegaMHC combination weights")

    if norm_weight is not None:
        if (
            norm_weight.shape != (hidden,)
            or not norm_weight.is_floating_point()
            or norm_weight.device != residual.device
            or norm_weight.requires_grad
        ):
            raise ValueError("invalid MegaMHC normalization weight")
        # vLLM rounds normalization weights to BF16 before multiplication.
        # Triton performs the cast while loading; native SIMD needs BF16 input.
        norm_weight = norm_weight.contiguous()

    k = streams * hidden
    block_p = triton.next_power_of_2(mixes)
    q = max(16, block_p)
    if packed_fn is not None and (
        packed_fn.shape not in ((3, q, k), _packed_weight_shape(q, k))
        or packed_fn.dtype != torch.bfloat16
        or packed_fn.device != x.device
        or not packed_fn.is_contiguous()
        or packed_fn.requires_grad
    ):
        raise ValueError("invalid MegaMHC packed projection weights")

    new_post = torch.empty((*outer, streams, 1), device=x.device, dtype=torch.float32)
    new_comb = torch.empty(
        (*outer, streams, streams), device=x.device, dtype=torch.float32
    )
    y = torch.empty_like(x)
    if not tokens:
        return torch.empty_like(residual), new_post, new_comb, y

    block_h = triton.next_power_of_2(hidden)
    comb_res_mix = comb_res_mix.view(tokens, streams, streams)
    properties = torch.cuda.get_device_properties(x.device)
    multiprocessors = properties.multi_processor_count
    norm = norm_weight if norm_weight is not None else x
    inputs = (x, residual, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base, norm)
    outputs = (new_post, new_comb, y)
    scalars = (
        *comb_res_mix.stride(),
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_eps,
    )
    has_norm = norm_weight is not None
    with torch_device_fn.device(x.device):
        # Small grids and wide projections favor the per-token path. The
        # pipeline keeps its projection width within one narrow AIU tile.
        if (
            (properties.major, properties.minor) == (8, 0)
            and tokens > 8 * multiprocessors
            and k >= 256
            and block_p <= 32
            and has_triton_tle_attrs(("load",))
        ):
            memset = counter_memset()
            if memset is not None:
                new_residual = _launch_pipeline(
                    inputs,
                    outputs,
                    scalars,
                    tokens,
                    streams,
                    hidden,
                    mixes,
                    q,
                    has_norm,
                    multiprocessors,
                    memset,
                    packed_fn=packed_fn,
                    bitcode=device_bitcode() if implementation == "auto" else None,
                )
                return new_residual.view(residual.shape), *outputs

        new_residual = torch.empty_like(residual)
        # Reuse large FP32 weight matrices across a few rows. SIMD keeps
        # small queues on the full compute grid without the MMA mode limit.
        native_simt = (
            implementation == "auto"
            and (properties.major, properties.minor) == (8, 0)
            and multiprocessors < tokens <= 8 * multiprocessors
            and block_p <= 32
            and k * block_p >= 4096 * multiprocessors
            and hidden % 8 == 0
            and 2 * tokens * k <= 2**32
            and sinkhorn_repeat < 2**32
            and norm.dtype == torch.bfloat16
            and all(t.data_ptr() % 16 == 0 for t in (x, residual, fn, norm))
            and has_triton_tle_attrs(("gpu.alloc", "gpu.local_ptr"))
        )
        if native_simt:
            bitcode = device_bitcode()
            if bitcode is not None:
                from flaggems_vllm.runtime.backend._thead.ops.megamhc_tle import (
                    mega_mhc_simt_kernel,
                )

                rows, warps, blocks, large_blocks = _simt_launch_config(
                    tokens, hidden, mixes, multiprocessors
                )
                mega_mhc_simt_kernel[(blocks,)](
                    *inputs,
                    new_residual,
                    *outputs,
                    *scalars,
                    tokens,
                    hidden,
                    streams,
                    has_norm,
                    rows,
                    warps,
                    LARGE_BLOCKS=large_blocks,
                    num_warps=warps,
                    num_stages=1,
                    enable_fp_fusion=False,
                    extern_libs={"megamhc": bitcode},
                )
                return new_residual, *outputs

        # With only one CTA per token, use a wider K tile at low occupancy.
        # More waves favor fewer registers per CTA. Bound the accumulator for
        # larger stream counts as its size is proportional to the mix width.
        if tokens <= 2 * multiprocessors and block_p <= 32:
            block_k, warps = 256, 8
        elif tokens <= 4 * multiprocessors and block_p <= 32:
            block_k, warps = 256, 4
        else:
            block_k, warps = 128, 8 if block_p > 32 else 2
        mega_mhc_kernel[(tokens,)](
            *inputs,
            new_residual,
            *outputs,
            *scalars,
            hidden,
            streams,
            mixes,
            block_p,
            min(block_k, block_h),
            block_h,
            has_norm,
            num_warps=warps,
            enable_fp_fusion=False,
        )
    return new_residual, *outputs


def _launch_pipeline(
    inputs,
    outputs,
    scalars,
    tokens,
    streams,
    hidden,
    mixes,
    q,
    has_norm,
    multiprocessors,
    memset,
    *,
    packed_fn=None,
    bitcode=None,
):
    from flaggems_vllm.runtime.backend._thead.ops.megamhc_tle import mega_mhc_pipeline

    device = inputs[0].device
    m, bk, pack_block = 64, _PROJECTION_BLOCK_K, 2048
    groups = triton.cdiv(tokens, m)
    padded = groups * m
    k = streams * hidden
    splits = _projection_splits(k)
    prepacked = packed_fn is not None
    split_packed = prepacked and packed_fn.ndim == 4
    aligned_prepacked = prepacked and hidden % bk == 0
    async_post = (
        bitcode is not None
        and prepacked
        and streams in (2, 4)
        and hidden % 256 == 0
        and inputs[0].data_ptr() % 16 == 0
        and inputs[1].data_ptr() % 16 == 0
        and has_triton_tle_attrs(("gpu.alloc", "gpu.local_ptr"))
    )
    # Long queues can amortize the drain between staggered stages.
    wave, lag = (8, 2) if groups >= 4 * multiprocessors else (min(groups, 32), 0)
    short_wave = max(1, multiprocessors // 4)
    if aligned_prepacked and groups >= 2 * multiprocessors:
        # A separate packing stage no longer precedes the post and GEMM waves.
        wave, lag = max(1, multiprocessors // 8), 3
    if (
        not prepacked
        and groups <= short_wave + short_wave // 4
        and k * q >= 4096 * multiprocessors
    ):
        # A deep packing queue can dominate dispatch when there are only one
        # or two token waves. Combine packing jobs and stagger the compute.
        wave, lag = min(groups, short_wave), 1
        pack_block = 8192 if groups <= short_wave else 4096
    waves = triton.cdiv(groups, wave)
    compact_waves = (
        lag > 0
        and groups // wave >= 2 * lag
        and (aligned_prepacked or groups % wave == 0)
    )
    scheduled_groups = groups if compact_waves else (waves + 2 * lag) * wave
    jobs = scheduled_groups * (2 * (m // 8) + splits)
    if not prepacked:
        jobs += triton.cdiv(k * q, pack_block)
        packed_fn = torch.empty((3, q, k), dtype=torch.bfloat16, device=device)

    residual = torch.empty(
        (padded, streams, hidden), dtype=torch.bfloat16, device=device
    )
    # Long queues benefit from streaming residual stores and packing all
    # split-K partials for one token group together.
    long_queue = groups >= (1 if aligned_prepacked else 2) * multiprocessors
    partial_shape = (groups, splits, m, q) if long_queue else (splits, padded, q)
    partial = torch.empty(partial_shape, dtype=torch.float32, device=device)
    sum_sq = torch.empty((padded,), dtype=torch.float32, device=device)
    # Ticket, optional packing counter and two counters per group. Allocation in the
    # caller's stream gives eager invocations and captured graphs private storage.
    counters = torch.empty(
        (1 + int(not prepacked) + 2 * groups,), dtype=torch.int64, device=device
    )
    clear_counters(counters, memset)
    mega_mhc_pipeline[(jobs,)](
        *inputs,
        residual,
        *outputs,
        counters,
        packed_fn,
        partial,
        sum_sq,
        *scalars,
        tokens,
        hidden,
        streams,
        mixes,
        q,
        m,
        bk,
        splits,
        pack_block,
        wave,
        lag,
        has_norm,
        long_queue,
        compact_waves,
        PREPACKED=prepacked,
        SPLIT_PACKED=split_packed,
        ASYNC_POST=async_post,
        NATIVE_WAIT=bitcode is not None,
        num_warps=2,
        num_stages=1,
        enable_fp_fusion=False,
        extern_libs={"megamhc": bitcode} if bitcode is not None else {},
    )
    return residual[:tokens]
