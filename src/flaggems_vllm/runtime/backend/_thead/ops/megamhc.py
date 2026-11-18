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

import math

import torch
import triton

from flaggems_vllm.runtime import device, torch_device_fn
from flaggems_vllm.utils.triton_version_utils import has_triton_tle_attrs

from .megamhc_kernel import mega_mhc_kernel
from .megamhc_runtime import clear_counters, counter_memset, device_bitcode

_PROJECTION_BLOCK_K = 64
_PROJECTION_MAX_SPLITS = 8


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

    from .megamhc_pipeline import pack_mega_mhc_weights_kernel

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

    Combination weights index [token, source stream, destination stream].
    Returns the BF16 updated residual, FP32 post and combination coefficients,
    and the BF16 collapsed layer input with optional RMS normalization.

    For fixed weights, pass the result of pack_mega_mhc_weights(fn) as packed_fn
    to the large-input pipeline. Keep fn and its packed weights consistent and
    keep packed_fn alive for the lifetime of any graph that captures it.
    The per-token path reads fn directly. Omit packed_fn when updating fn
    between graph replays. Use implementation="triton" to run the fused
    pipeline with Triton postprocessing and CTA synchronization.
    """
    if implementation not in ("auto", "triton"):
        raise ValueError("MegaMHC implementation must be 'auto' or 'triton'")
    if device.vendor_name != "thead" or residual.device.type != "cuda":
        raise NotImplementedError("MegaMHC requires T-Head PPU")
    if residual.ndim != 3:
        raise ValueError("residual must have shape [tokens, streams, hidden]")
    tokens, streams, hidden = residual.shape
    if streams not in (1, 2, 4, 8) or not 0 < hidden <= 16384:
        raise NotImplementedError(
            "MegaMHC supports 1/2/4/8 streams and hidden sizes up to 16384"
        )
    if n_splits != 1 or tile_n != 1:
        raise NotImplementedError("MegaMHC selects its tiling internally")
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
        (x, (tokens, hidden), torch.bfloat16),
        (residual, (tokens, streams, hidden), torch.bfloat16),
        (post_layer_mix, (tokens, streams, 1), torch.float32),
        (fn, (mixes, streams * hidden), torch.float32),
        (hc_scale, (3,), torch.float32),
        (hc_base, (mixes,), torch.float32),
    ]
    if norm_weight is not None:
        specs.append((norm_weight, (hidden,), torch.bfloat16))
    for tensor, shape, dtype in specs:
        if tensor.shape != shape or tensor.dtype != dtype or not tensor.is_contiguous():
            raise ValueError(
                f"MegaMHC expected a contiguous {dtype} tensor of shape {shape}"
            )
        if tensor.device != residual.device or tensor.requires_grad:
            raise ValueError("MegaMHC requires inference tensors on one PPU")
    if (
        comb_res_mix.shape != (tokens, streams, streams)
        or comb_res_mix.dtype != torch.float32
        or comb_res_mix.device != residual.device
        or comb_res_mix.requires_grad
    ):
        raise ValueError("invalid MegaMHC combination weights")

    new_post = torch.empty_like(post_layer_mix)
    new_comb = torch.empty(
        (tokens, streams, streams), device=x.device, dtype=torch.float32
    )
    y = torch.empty_like(x)
    if not tokens:
        return torch.empty_like(residual), new_post, new_comb, y

    block_p = triton.next_power_of_2(mixes)
    block_h = triton.next_power_of_2(hidden)
    k = streams * hidden
    q = max(16, block_p)
    if packed_fn is not None and (
        packed_fn.shape not in ((3, q, k), _packed_weight_shape(q, k))
        or packed_fn.dtype != torch.bfloat16
        or packed_fn.device != x.device
        or not packed_fn.is_contiguous()
        or packed_fn.requires_grad
    ):
        raise ValueError("invalid MegaMHC packed projection weights")
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
            and tokens < 2**31
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
                return new_residual, *outputs

        new_residual = torch.empty_like(residual)
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
    from .megamhc_pipeline import mega_mhc_pipeline

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
