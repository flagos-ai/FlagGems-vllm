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

"""Triton prenorm GEMM specialized for the fused mHC decode path."""

from __future__ import annotations

import weakref
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except ImportError:
    TensorDescriptor = None


_HC_HIDDEN_SIZE = 16384
_MIX_COUNT = 24
_PADDED_MIX_COUNT = 32
_PACK_BLOCK_K = 256


class _PrenormConfig(NamedTuple):
    block_m: int
    block_k: int
    split_k: int
    num_warps: int
    num_stages: int


class _PackedFnEntry(NamedTuple):
    source: weakref.ReferenceType[torch.Tensor]
    version: int | None
    packed: torch.Tensor


# Static specializations are intentional. These are fixed decode shapes, and
# the choices below were validated on NVIDIA H20 (SM90). Other SM90 devices
# still need their own performance validation. Keeping the split count static
# also lets the downstream epilogue fully unroll its reduction. Pipeline
# parameters are not interchangeable: validate the complete producer/consumer
# CUDA Graph whenever changing them, not only isolated GEMM output.
_PRENORM_CONFIGS = {
    64: _PrenormConfig(64, 128, 64, 4, 3),
    # S=32 is within 0.11 us of the isolated S=128 GEMM while cutting the
    # downstream partial-reduction traffic by 4x.
    96: _PrenormConfig(64, 128, 32, 4, 3),
    # S=32 minimizes the complete prenorm + epilogue chain: reducing partial
    # traffic outweighs the small isolated GEMM cost at this token count.
    128: _PrenormConfig(64, 128, 32, 4, 3),
}


# ``WeakKeyDictionary`` cannot safely use ``torch.Tensor`` directly because
# weak-reference equality delegates to Tensor's elementwise ``__eq__``. Key by
# object identity and keep an explicit weak reference for automatic cleanup.
_PACKED_FN_CACHE: dict[int, _PackedFnEntry] = {}


@triton.jit
def _pack_mhc_fn_kernel(
    fn_ptr,
    packed_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    PADDED_N: tl.constexpr,
    LAUNCH_PDL: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()

    block = tl.program_id(0)
    offsets_k = block * BLOCK_K + tl.arange(0, BLOCK_K)[:, None]
    offsets_n = tl.arange(0, PADDED_N)[None, :]
    values = tl.load(
        fn_ptr + offsets_n * K + (offsets_k % K),
        mask=offsets_n < N,
        other=0.0,
    )
    # A single BF16 cast loses small FP32 components, including weights that
    # remain after cancellation. Keep a second BF16 component in a separate
    # contiguous plane and accumulate both tensor-core products in FP32.
    high = values.to(tl.bfloat16)
    low = (values - high.to(tl.float32)).to(tl.bfloat16)
    tl.store(
        packed_ptr + offsets_k * PADDED_N + offsets_n,
        tl.where(offsets_k < K, high, low),
    )

    if LAUNCH_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _mhc_prenorm_gemm_kernel(
    residual_desc,
    packed_fn_desc,
    partial_ptr,
    sqrsum_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    PADDED_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    LAUNCH_PDL: tl.constexpr,
):
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()

    token_block = tl.program_id(0)
    split = tl.program_id(1)
    token_start = (token_block * BLOCK_M).to(tl.int32)
    k_per_split: tl.constexpr = K // SPLIT_K
    k_start = (split * k_per_split).to(tl.int32)

    accumulator = tl.zeros((BLOCK_M, PADDED_N), dtype=tl.float32)
    square_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k_delta in range(0, k_per_split, BLOCK_K):
        current_k = k_start + k_delta
        residual = residual_desc.load([token_start, current_k])
        packed_fn = packed_fn_desc.load([current_k, 0])
        packed_fn_low = packed_fn_desc.load([K + current_k, 0])
        accumulator = tl.dot(
            residual,
            packed_fn_low,
            acc=accumulator,
            allow_tf32=False,
        )
        accumulator = tl.dot(
            residual,
            packed_fn,
            acc=accumulator,
            allow_tf32=False,
        )
        residual_f32 = residual.to(tl.float32)
        square_sum += tl.sum(residual_f32 * residual_f32, axis=1)

    offsets_m = token_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = tl.arange(0, PADDED_N)
    partial_offsets = (
        split * M * PADDED_N + offsets_m[:, None] * PADDED_N + offsets_n[None, :]
    )
    valid_tokens = offsets_m < M
    tl.store(
        partial_ptr + partial_offsets,
        accumulator,
        mask=valid_tokens[:, None],
    )
    tl.store(
        sqrsum_ptr + split * M + offsets_m,
        square_sum,
        mask=valid_tokens,
    )

    if LAUNCH_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _supports_pdl(tensor: torch.Tensor) -> bool:
    return (
        torch.version.hip is None
        and torch.cuda.get_device_capability(tensor.device)[0] >= 9
    )


def _validate_inputs(residual: torch.Tensor, fn: torch.Tensor) -> _PrenormConfig:
    if residual.ndim != 2 or residual.shape[1] != _HC_HIDDEN_SIZE:
        raise NotImplementedError(
            f"mHC prenorm requires residual[M, {_HC_HIDDEN_SIZE}]"
        )
    config = _PRENORM_CONFIGS.get(residual.shape[0])
    if config is None:
        raise NotImplementedError(
            f"mHC prenorm supports token counts {tuple(_PRENORM_CONFIGS)}"
        )
    if fn.shape != (_MIX_COUNT, _HC_HIDDEN_SIZE):
        raise ValueError(f"fn must have shape ({_MIX_COUNT}, {_HC_HIDDEN_SIZE})")
    if residual.dtype != torch.bfloat16:
        raise NotImplementedError("mHC prenorm residual must use bfloat16")
    if fn.dtype != torch.float32:
        raise NotImplementedError("mHC prenorm fn must use float32")
    if not residual.is_cuda or not fn.is_cuda:
        raise NotImplementedError("mHC prenorm requires CUDA tensors")
    if residual.device != fn.device:
        raise ValueError("mHC prenorm inputs must be on the same device")
    if not residual.is_contiguous() or not fn.is_contiguous():
        raise NotImplementedError("mHC prenorm requires contiguous inputs")
    if residual.requires_grad or fn.requires_grad:
        raise NotImplementedError("mHC prenorm is an inference-only path")
    if residual.device.index != torch.cuda.current_device():
        raise NotImplementedError(
            "mHC prenorm requires its input device to be the current CUDA device"
        )
    if torch.cuda.get_device_capability(residual.device)[0] < 9:
        raise NotImplementedError("mHC prenorm TMA path requires SM90 or newer")
    if TensorDescriptor is None:
        raise NotImplementedError("mHC prenorm requires Triton TensorDescriptor")
    return config


def _get_packed_fn(fn: torch.Tensor) -> tuple[torch.Tensor, bool]:
    current_stream = torch.cuda.current_stream(fn.device)
    cache_key = id(fn)
    try:
        version = fn._version
    except RuntimeError:
        # Inference tensors intentionally have no version counter. Serving
        # weights are immutable, so identity is the cache validity contract
        # for that tensor class.
        version = None
    cached = _PACKED_FN_CACHE.get(cache_key)
    if cached is not None and cached.source() is fn and cached.version == version:
        return cached.packed, True

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "mHC weights must be packed by one warm-up call before CUDA Graph capture"
        )

    packed = torch.empty(
        (2 * _HC_HIDDEN_SIZE, _PADDED_MIX_COUNT),
        dtype=torch.bfloat16,
        device=fn.device,
    )
    _pack_mhc_fn_kernel[(triton.cdiv(2 * _HC_HIDDEN_SIZE, _PACK_BLOCK_K),)](
        fn,
        packed,
        K=_HC_HIDDEN_SIZE,
        N=_MIX_COUNT,
        PADDED_N=_PADDED_MIX_COUNT,
        # Packing is a one-time cache fill. Do not release a dependent grid
        # until every packing CTA has made the complete weight visible.
        LAUNCH_PDL=False,
        BLOCK_K=_PACK_BLOCK_K,
        num_warps=8,
        num_stages=1,
        launch_pdl=False,
    )

    ready_event = torch.cuda.Event(blocking=False, interprocess=False)
    ready_event.record(current_stream)
    # Publish only completed immutable weights. This one-time initialization
    # synchronization makes every later cache hit safe on any CUDA stream and
    # lets graph capture avoid all external-event query/wait operations.
    ready_event.synchronize()

    def remove_cached(reference, key=cache_key):
        cached_entry = _PACKED_FN_CACHE.get(key)
        if cached_entry is not None and cached_entry.source is reference:
            _PACKED_FN_CACHE.pop(key, None)

    fn_reference = weakref.ref(fn, remove_cached)
    _PACKED_FN_CACHE[cache_key] = _PackedFnEntry(
        fn_reference,
        version,
        packed,
    )
    return packed, False


def mhc_prepare_weights(fn: torch.Tensor) -> None:
    """Prepare immutable FP32 mHC weights before CUDA Graph capture.

    This is a host preparation helper, not a dispatchable operator. The first
    call launches a Triton packing kernel and waits for its completion, making
    subsequent use safe on every stream. Keep ``fn`` alive and unchanged for
    the lifetime of captured graphs. Ordinary tensors are repacked after an
    in-place update on the next eager call; inference tensors have no version
    counter and must remain immutable.
    """
    if fn.shape != (_MIX_COUNT, _HC_HIDDEN_SIZE):
        raise ValueError(f"fn must have shape ({_MIX_COUNT}, {_HC_HIDDEN_SIZE})")
    if fn.dtype != torch.float32 or not fn.is_cuda:
        raise NotImplementedError("mHC weights must be float32 CUDA tensors")
    if not fn.is_contiguous() or fn.requires_grad:
        raise NotImplementedError("mHC weights must be contiguous inference tensors")
    if fn.device.index != torch.cuda.current_device():
        raise NotImplementedError("mHC weights must be on the current CUDA device")
    if torch.cuda.get_device_capability(fn.device)[0] < 9 or TensorDescriptor is None:
        raise NotImplementedError(
            "mHC weights require SM90+ and Triton TensorDescriptor"
        )
    _get_packed_fn(fn)


def mhc_prenorm_gemm(
    residual: torch.Tensor,
    fn: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute padded split-K mHC mix logits and residual square sums.

    The returned tensors have shapes ``[S, M, 32]`` and ``[S, M]``. Only the
    first 24 columns contain semantic mix logits; the remaining columns are
    zero-padding that keeps TMA and tensor-core accesses contiguous. ``S`` is
    selected from the static token-count specialization.

    ``fn`` is split into high/low BF16 components and transposed by a Triton
    kernel, then cached by the source tensor and its in-place version. The
    one-time pack completes before its cache entry is published. CUDA Graph
    capture requires a warm-up call or ``mhc_prepare_weights`` for each
    immutable ``fn`` identity. No PyTorch compute or copy kernel is used by
    this production path.
    """
    config = _validate_inputs(residual, fn)

    with torch_device_fn.device(residual.device):
        packed_fn, packed_cache_hit = _get_packed_fn(fn)
        # A cache miss launches the packing kernel immediately before this
        # GEMM. Keep ordinary stream ordering for that first invocation; later
        # steady-state invocations can overlap with their upstream producer.
        launch_pdl = packed_cache_hit and _supports_pdl(residual)
        partial = torch.empty(
            (config.split_k, residual.shape[0], _PADDED_MIX_COUNT),
            dtype=torch.float32,
            device=residual.device,
        )
        sqrsum = torch.empty(
            (config.split_k, residual.shape[0]),
            dtype=torch.float32,
            device=residual.device,
        )

        residual_desc = TensorDescriptor(
            residual,
            list(residual.shape),
            list(residual.stride()),
            [config.block_m, config.block_k],
            padding="zero",
        )
        packed_fn_desc = TensorDescriptor(
            packed_fn,
            list(packed_fn.shape),
            list(packed_fn.stride()),
            [config.block_k, _PADDED_MIX_COUNT],
            padding="zero",
        )
        grid = (triton.cdiv(residual.shape[0], config.block_m), config.split_k)
        _mhc_prenorm_gemm_kernel[grid](
            residual_desc,
            packed_fn_desc,
            partial,
            sqrsum,
            M=residual.shape[0],
            K=_HC_HIDDEN_SIZE,
            PADDED_N=_PADDED_MIX_COUNT,
            BLOCK_M=config.block_m,
            BLOCK_K=config.block_k,
            SPLIT_K=config.split_k,
            LAUNCH_PDL=launch_pdl,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
            launch_pdl=launch_pdl,
        )

    return partial, sqrsum


def mhc_prenorm_split_count(num_tokens: int) -> int:
    """Return the static split count consumed by the Triton epilogue."""
    config = _PRENORM_CONFIGS.get(num_tokens)
    if config is None:
        raise NotImplementedError(
            f"mHC prenorm supports token counts {tuple(_PRENORM_CONFIGS)}"
        )
    return config.split_k


__all__ = ["mhc_prepare_weights", "mhc_prenorm_gemm", "mhc_prenorm_split_count"]
