# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stateless Gumbel sampling aligned with vLLM 0ef2a95cc861.

The Murmur3 and uniform transforms follow vLLM's Model Runner V2 sampler.
All backends share the sampling arithmetic. Capability-gated integer helpers
handle verified compiler lowering issues without changing the RNG algorithm.
"""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

_BLOCK_SIZE = 1024
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_INDEX_DTYPES = (torch.int32, torch.int64)
_ASCEND_351 = (
    runtime.device.vendor_name == "ascend"
    and triton.__version__.split("+", 1)[0] == "3.5.1"
)
_ASCEND_SIGNED_BITOPS = tl.constexpr(_ASCEND_351)
_ASCEND_GRID_LIMIT = 65535
if _ASCEND_351:
    from triton.runtime import driver as _ascend_driver


@triton.jit
def _source_offset(row, indices, stride_row, stride_col):
    # A noncontiguous source may span more than 2**31 elements.
    return row.to(tl.int64) * stride_row + indices.to(tl.int64) * stride_col


@triton.jit
def _rng_token_offsets(
    indices, block, BLOCK_SIZE: tl.constexpr, USE_SCAN: tl.constexpr
):
    if USE_SCAN:
        # Iluvatar Triton 3.1.0 miscompiles hashing arithmetic on make_range.
        # An exact integer scan supplies the same global uint32 token keys.
        local = tl.cumsum(tl.full((BLOCK_SIZE,), 1, tl.int32), axis=0) - 1
        return (block * BLOCK_SIZE + local).to(tl.uint32)
    return indices.to(tl.uint32)


@triton.jit
def _logical_rshift32(value, shift: tl.constexpr):
    if _ASCEND_SIGNED_BITOPS:
        # Verified Ascend 3.5.1 SIMD lowering treats shrui as signed shift.
        return (value.to(tl.int32) >> shift) & (0xFFFFFFFF >> shift)
    else:
        return value >> shift


@triton.jit
def _murmur3_rotl32(value, shift: tl.constexpr):
    return (value << shift) | _logical_rshift32(value, 32 - shift)


@triton.jit
def _murmur3_mix(h, key):
    key = key * 0xCC9E2D51
    key = _murmur3_rotl32(key, 15)
    key = key * 0x1B873593
    h = _murmur3_rotl32(h ^ key, 13)
    return h * 5 + 0xE6546B64


@triton.jit
def _murmur3_fmix32(h, ZERO_EXTEND: tl.constexpr = True):
    h = (h ^ _logical_rshift32(h, 16)) * 0x85EBCA6B
    h = (h ^ _logical_rshift32(h, 13)) * 0xC2B2AE35
    h = h ^ _logical_rshift32(h, 16)
    if _ASCEND_SIGNED_BITOPS and ZERO_EXTEND:
        # The same stack also loses unsigned zero-extension when widening.
        return h.to(tl.int64) & 0xFFFFFFFF
    else:
        return h


@triton.jit
def murmur3_hash32(
    seed, pos, offset, domain: tl.constexpr = 0, ZERO_EXTEND: tl.constexpr = True
):
    seed = seed.to(tl.int64)
    pos = pos.to(tl.int64)
    h = (seed ^ seed).to(tl.uint32) ^ domain
    h = _murmur3_mix(h, (seed & 0xFFFFFFFF).to(tl.uint32))
    h = _murmur3_mix(h, ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32))
    h = _murmur3_mix(h, (pos & 0xFFFFFFFF).to(tl.uint32))
    h = _murmur3_mix(h, offset.to(tl.uint32))
    return _murmur3_fmix32(h ^ 16, ZERO_EXTEND)


@triton.jit
def _uniform32_from_random32(random32):
    random32 = random32.to(tl.uint32)
    hi = _logical_rshift32(random32, 16).to(tl.int32)
    lo = (random32 & 0xFFFF).to(tl.int32)
    return (
        hi.to(tl.float32) * 1.52587890625e-05
        + (lo.to(tl.float32) + 0.5) * 2.3283064365386963e-10
    )


@triton.jit
def murmur3_uniform32(seed, pos, offset):
    if _ASCEND_SIGNED_BITOPS:
        # Uniform32 consumes the original 32 bits without a widened return value.
        return _uniform32_from_random32(
            murmur3_hash32(seed, pos, offset, ZERO_EXTEND=False)
        )
    return _uniform32_from_random32(murmur3_hash32(seed, pos, offset))


@triton.jit
def _uniform64_from_random53(random53):
    uniform = (random53.to(tl.float64) + 0.5) * 1.1102230246251565e-16
    return tl.where(uniform == 1.0, uniform - 1.1102230246251565e-16, uniform)


@triton.jit
def murmur3_uniform64(seed, pos, offset):
    lo = murmur3_hash32(seed, pos, offset).to(tl.uint64)
    hi = murmur3_hash32(seed, pos, offset, domain=0x9E3779B9).to(tl.uint64)
    return _uniform64_from_random53(((hi << 32) | lo) >> 11)


@triton.jit
def _log1p_neg_stable(value, FINITE_INPUT: tl.constexpr = False):
    polynomial = 1.0 / 8.0
    polynomial = 1.0 / 7.0 + value * polynomial
    polynomial = 1.0 / 6.0 + value * polynomial
    polynomial = 1.0 / 5.0 + value * polynomial
    polynomial = 1.0 / 4.0 + value * polynomial
    polynomial = 1.0 / 3.0 + value * polynomial
    polynomial = 1.0 / 2.0 + value * polynomial
    polynomial = 1.0 + value * polynomial
    series = -value * polynomial
    if FINITE_INPUT:
        # Internal uniform32 is finite; ALL avoids maxnum NaN repair on Ascend.
        # Keep epsilon because FP32 uniform32 can round to 1.0.
        direct = tl.log(
            tl.maximum(
                1.0 - value,
                5.960464477539063e-08,
                propagate_nan=tl.PropagateNan.ALL,
            )
        )
    else:
        direct = tl.log(tl.maximum(1.0 - value, 5.960464477539063e-08))
    return tl.where(value < 0.25, series, direct)


@triton.jit
def _gumbel_from_uniform(
    uniform, USE_FP64: tl.constexpr, FINITE_INPUT: tl.constexpr = False
):
    if USE_FP64:
        return -tl.log(-tl.log(uniform))
    return -tl.log(-_log1p_neg_stable(uniform, FINITE_INPUT))


@triton.jit
def _sample_scores(
    logits, noise, temperature, APPLY_TEMPERATURE: tl.constexpr, USE_FP64: tl.constexpr
):
    """Pointwise computation also usable by private fixed-noise tests."""
    logits = logits.to(tl.float32)
    if APPLY_TEMPERATURE:
        if temperature != 0.0:
            logits = logits / temperature
    if USE_FP64:
        logits = logits.to(tl.float64)
    if temperature != 0.0:
        logits = logits + noise
    return logits


@triton.jit
def _masked_argmax(scores, indices, mask):
    scores = tl.where(mask, scores, float("-inf"))
    maximum = tl.max(scores, axis=0)
    index = tl.min(tl.where(mask & (scores == maximum), indices, 0x7FFFFFFF), axis=0)
    return maximum, index


@triton.jit(do_not_specialize=["grid_row_start", "grid_block_start"])
def _gumbel_sample_kernel(
    LOGITS,
    MAPPING,
    TEMPERATURE,
    SEEDS,
    POSITIONS,
    CACHE,
    COLS,
    CACHE_SOURCE,
    ROW_MAPPING,
    BIAS,
    LOCAL_IDS,
    LOCAL_MAX,
    SAMPLED,
    stride_logits,
    stride_mapping,
    stride_temperature,
    stride_seeds,
    stride_positions,
    stride_cache_row,
    stride_cache_col,
    stride_cols,
    stride_source_row,
    stride_source_col,
    stride_row_mapping,
    stride_bias,
    NUM_BLOCKS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    N_BUCKET: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    IS_DRAFTING: tl.constexpr,
    USE_FP64: tl.constexpr,
    USE_SCAN_OFFSETS: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    HAS_SOURCE: tl.constexpr,
    HAS_ROW_MAPPING: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    grid_row_start=0,
    grid_block_start=0,
    USE_GRID_OFFSETS: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    if USE_GRID_OFFSETS:
        row += grid_row_start.to(tl.int64)
        block += grid_block_start
    indices = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = indices < VOCAB_SIZE
    logits = tl.load(LOGITS + row * stride_logits + indices, mask, float("-inf"))
    if HAS_BIAS:
        bias = tl.load(BIAS + row * stride_bias + indices, mask, 0.0)
        logits = (logits.to(tl.float32) + bias.to(tl.float32)).to(logits.dtype)

    req = tl.load(MAPPING + row * stride_mapping).to(tl.int64)
    valid_req = req >= 0
    temperature = tl.load(TEMPERATURE + req * stride_temperature, valid_req, 0.0).to(
        tl.float32
    )
    seed = tl.load(SEEDS + req * stride_seeds, valid_req, 0).to(tl.int64)
    position = tl.load(POSITIONS + row * stride_positions).to(tl.int64)
    if HAS_CACHE:
        if PER_TOKEN_COL:
            col = tl.load(COLS + row * stride_cols).to(tl.int64)
        else:
            col = tl.load(COLS).to(tl.int64)
        cache_row = req
        if HAS_ROW_MAPPING:
            cache_row = tl.load(
                ROW_MAPPING + req * stride_row_mapping, valid_req, -1
            ).to(tl.int64)
        if HAS_SOURCE:
            cached = tl.load(
                CACHE_SOURCE
                + _source_offset(row, indices, stride_source_row, stride_source_col),
                mask,
                0.0,
            )
        else:
            cached = logits
        tl.store(
            CACHE + cache_row * stride_cache_row + col * stride_cache_col + indices,
            cached,
            mask & (cache_row >= 0),
        )
        seed = seed ^ (col << 32)

    if USE_FP64:
        noise = tl.full((BLOCK_SIZE,), 0.0, tl.float64)
    else:
        noise = tl.full((BLOCK_SIZE,), 0.0, tl.float32)
    if temperature != 0.0:
        rng_indices = _rng_token_offsets(indices, block, BLOCK_SIZE, USE_SCAN_OFFSETS)
        if IS_DRAFTING:
            position = position + (1 << 30)
        if USE_FP64:
            uniform = murmur3_uniform64(seed, position, rng_indices)
        else:
            uniform = murmur3_uniform32(seed, position, rng_indices)
        noise = _gumbel_from_uniform(uniform, USE_FP64)
    scores = _sample_scores(logits, noise, temperature, APPLY_TEMPERATURE, USE_FP64)
    maximum, index = _masked_argmax(scores, indices, mask)
    if NUM_BLOCKS == 1:
        tl.store(SAMPLED + row, index.to(tl.int64))
    else:
        partial = row * NUM_BLOCKS + block
        tl.store(LOCAL_IDS + partial, index.to(tl.int64))
        tl.store(LOCAL_MAX + partial, maximum)


@triton.jit(do_not_specialize=["grid_row_start"])
def _gumbel_combine_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    sampled_ptr,
    num_blocks,
    PADDED_NUM_BLOCKS: tl.constexpr,
    grid_row_start=0,
    USE_GRID_OFFSETS: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    if USE_GRID_OFFSETS:
        row += grid_row_start.to(tl.int64)
    blocks = tl.arange(0, PADDED_NUM_BLOCKS)
    mask = blocks < num_blocks
    values = tl.load(
        local_max_ptr + row * local_max_stride + blocks, mask, float("-inf")
    )
    # Preserve the source's helper behavior even though NaN logits are outside
    # the public input domain. Lowest block implies lowest global token on ties.
    nan_block = tl.min(
        tl.where(mask & (values != values), blocks, PADDED_NUM_BLOCKS), axis=0
    )
    if nan_block < PADDED_NUM_BLOCKS:
        winner = nan_block
    else:
        maximum = tl.max(values, axis=0)
        winner = tl.min(
            tl.where(mask & (values == maximum), blocks, PADDED_NUM_BLOCKS), axis=0
        )
    token = tl.load(local_argmax_ptr + row * local_argmax_stride + winner)
    tl.store(sampled_ptr + row, token)


_raw_sample_kernel = libentry()(_gumbel_sample_kernel)
_combine_kernel = libentry()(_gumbel_combine_kernel)
if runtime.device.vendor_name == "nvidia":
    _sample_configs = runtime.get_tuned_config("gumbel_sample")
    _tuned_sample_kernel = libentry()(
        libtuner(
            configs=_sample_configs,
            key=[
                "VOCAB_SIZE",
                "N_BUCKET",
                "USE_FP64",
                "HAS_BIAS",
                "APPLY_TEMPERATURE",
                "IS_DRAFTING",
            ],
            use_cuda_graph=False,
        )(_gumbel_sample_kernel)
    )
else:
    _tuned_sample_kernel = _raw_sample_kernel


class _CompiledLauncher(triton.KernelInterface):
    """Reuse a tuned kernel without rebuilding LibEntry's argument dictionaries.

    The full-argument CompiledKernel ABI is verified on NVIDIA/Hygon Triton 3.6.0
    and the Ascend Triton 3.5.1 SIMD stack.
    Only code and scalar metadata are retained, never tensors or device values.
    LibEntry still owns compilation and configuration selection on every miss.
    """

    def __init__(self, entry, pointer_indices):
        self.entry = entry
        self.pointer_indices = frozenset(pointer_indices)
        self.params = entry.jit_function.params
        self.cache = {}

    def run(self, *args, grid, **kwargs):
        key = (
            runtime.torch_device_fn.current_device(),
            tuple(
                (
                    (arg.dtype, arg.data_ptr() % 16 == 0)
                    if i in self.pointer_indices and arg is not None
                    else arg
                )
                for i, arg in enumerate(args)
            ),
            tuple(kwargs.items()),
            (
                tuple(
                    _hygon_tensor_specialization(arg)
                    for i, arg in enumerate(args)
                    if i in self.pointer_indices and arg is not None
                )
                if _hygon_tensor_specialization is not None
                else None
            ),
        )
        cached = self.cache.get(key)
        if cached is None:
            kernel, constants = self.entry.run(*args, grid=grid, **kwargs)
            tail = tuple(
                kwargs.get(p.name, constants.get(p.name, p.default))
                for p in self.params[len(args) :]
            )
            self.cache[key] = kernel, tail
            return kernel, constants
        kernel, tail = cached
        # CompiledKernel obtains the current stream on each invocation.
        kernel[(grid + (1, 1))[:3]](*args, *tail)
        return kernel, None


_hygon_tensor_specialization = None
if (
    runtime.device.vendor_name == "hygon"
    and triton.__version__ == "3.6.0"
    and hasattr(triton.backends, "hcu")
):
    from triton.backends.hcu.compiler import HIPBackend

    _hygon_tensor_specialization = getattr(
        HIPBackend, "get_tensor_specialization", None
    )

_COMPILED_FAST_LAUNCH = (
    (
        runtime.device.vendor_name == "nvidia"
        and triton.__version__ == "3.6.0"
        and not _tuned_sample_kernel._has_flagtune_tuner
        and all(config.pre_hook is None for config in _sample_configs)
    )
    or (
        _hygon_tensor_specialization is not None
        and not _tuned_sample_kernel._has_flagtune_tuner
    )
    or (_ASCEND_351 and not _tuned_sample_kernel._has_flagtune_tuner)
)
_ascend_sample_config = (
    runtime.get_tuned_config("gumbel_sample")[0] if _ASCEND_351 else None
)
if _COMPILED_FAST_LAUNCH:
    _tuned_sample_kernel = _CompiledLauncher(_tuned_sample_kernel, range(13))
    _combine_kernel = _CompiledLauncher(_combine_kernel, (0, 2, 4))


_nocache_plans = {}


def _compile_plan_part(entry, grid, args, meta):
    kernel, constants = entry.run(*args, grid=grid, warmup=False, **meta)
    tail = tuple(
        meta.get(p.name, constants.get(p.name, p.default))
        for p in entry.jit_function.params[len(args) :]
    )
    return kernel[(grid + (1, 1))[:3]], tail


def _sample_nocache(
    logits,
    mapping,
    temperature,
    seed,
    pos,
    sampled,
    metadata,
    apply_temperature,
    is_drafting,
    use_fp64,
    block_size,
    blocks,
):
    """A code-only launch plan for the validated no-cache/no-bias scope."""
    n, vocab = metadata[:2]
    local_ids, local_max = None, None
    if blocks > 1:
        local_ids = torch.empty((n, blocks), dtype=torch.int64, device=logits.device)
        local_max = torch.empty(
            (n, blocks),
            dtype=torch.float64 if use_fp64 else torch.float32,
            device=logits.device,
        )
    args = (
        logits,
        mapping,
        temperature,
        seed,
        pos,
        None,
        None,
        None,
        None,
        None,
        local_ids,
        local_max,
        sampled,
        *metadata[2:],
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    combine_args = (local_ids, blocks, local_max, blocks, sampled, blocks)
    tensor_device = logits.device
    # Real NPU tensors carry a device index. Reuse that live metadata while
    # retaining the device guard and the original path for an unspecified index.
    device_index = (
        tensor_device.index if _ASCEND_351 and tensor_device.type == "npu" else None
    )
    with runtime.torch_device_fn.device(
        tensor_device if device_index is None else device_index
    ):
        device = (
            runtime.torch_device_fn.current_device()
            if device_index is None
            else device_index
        )
        key = (
            device,
            metadata,
            block_size,
            logits.dtype,
            mapping.dtype,
            tuple(
                None if tensor is None else tensor.data_ptr() % 16 == 0
                for tensor in (
                    logits,
                    mapping,
                    temperature,
                    seed,
                    pos,
                    local_ids,
                    local_max,
                    sampled,
                )
            ),
            apply_temperature,
            is_drafting,
            use_fp64,
            (
                tuple(
                    _hygon_tensor_specialization(tensor) if tensor is not None else None
                    for tensor in (
                        logits,
                        mapping,
                        temperature,
                        seed,
                        pos,
                        local_ids,
                        local_max,
                        sampled,
                    )
                )
                if _hygon_tensor_specialization is not None
                else None
            ),
        )
        plan = _nocache_plans.get(key)
        if plan is None:
            # Cold calls retain the original libtuner configuration discovery.
            launch = {}
            if _ASCEND_351:
                launch = dict(
                    num_warps=_ascend_sample_config.num_warps,
                    num_stages=_ascend_sample_config.num_stages,
                    enable_auto_blockify=False,
                )
            sample_runner, sample_tail = _compile_plan_part(
                _tuned_sample_kernel.entry,
                (n, blocks),
                args,
                dict(
                    NUM_BLOCKS=blocks,
                    VOCAB_SIZE=vocab,
                    N_BUCKET=triton.next_power_of_2(n),
                    APPLY_TEMPERATURE=apply_temperature,
                    IS_DRAFTING=is_drafting,
                    USE_FP64=use_fp64,
                    USE_SCAN_OFFSETS=False,
                    HAS_CACHE=False,
                    HAS_SOURCE=False,
                    HAS_ROW_MAPPING=False,
                    HAS_BIAS=False,
                    PER_TOKEN_COL=False,
                    BLOCK_SIZE=block_size,
                    **launch,
                ),
            )
            combine_runner, combine_tail = None, ()
            if blocks > 1:
                combine_runner, combine_tail = _compile_plan_part(
                    _combine_kernel.entry,
                    (n,),
                    combine_args,
                    dict(
                        PADDED_NUM_BLOCKS=triton.next_power_of_2(blocks),
                        num_warps=4,
                        num_stages=1,
                        **({"enable_auto_blockify": False} if _ASCEND_351 else {}),
                    ),
                )
            _nocache_plans[key] = (
                sample_runner,
                sample_tail,
                combine_runner,
                combine_tail,
            )
        else:
            sample_runner, sample_tail, combine_runner, combine_tail = plan
            # The Ascend runner accepts an explicit current stream. Reuse it only
            # within this call; the runner still handles launch metadata and hooks.
            if _ASCEND_351:
                stream = _ascend_driver.active.get_current_stream(device)
                sample_runner(*args, *sample_tail, stream=stream)
                if combine_runner is not None:
                    combine_runner(*combine_args, *combine_tail, stream=stream)
            else:
                sample_runner(*args, *sample_tail)
                if combine_runner is not None:
                    combine_runner(*combine_args, *combine_tail)
    return sampled


def _check_tensor(name, tensor, device, dtypes, ndim=None):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"gumbel_sample {name} must be a tensor")
    if tensor.layout != torch.strided:
        raise NotImplementedError(f"gumbel_sample {name} requires strided layout")
    if tensor.device != device:
        raise ValueError(f"gumbel_sample {name} must be on the logits device")
    if tensor.dtype not in dtypes:
        raise NotImplementedError(f"gumbel_sample {name} dtype is unsupported")
    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(f"gumbel_sample {name} must have rank {ndim}")
    if tensor.requires_grad:
        raise NotImplementedError("gumbel_sample supports inference only")
    if _COMPILED_FAST_LAUNCH and ndim in (1, 2):
        # Snapshot live metadata once; do not cache validation across calls.
        shape, strides = tensor.shape, tensor.stride()
        if strides[0] < 0 or (ndim == 2 and strides[1] < 0):
            raise NotImplementedError(f"gumbel_sample {name} has negative strides")
        if ndim == 1:
            extent = (shape[0] - 1) * strides[0] if shape[0] else 0
        elif shape[0] and shape[1]:
            extent = (shape[0] - 1) * strides[0] + (shape[1] - 1) * strides[1]
        else:
            extent = 0
        if extent > (1 << 63) - 1:
            raise NotImplementedError(f"gumbel_sample {name} offsets exceed int64")
        return shape, strides
    if any(stride < 0 for stride in tensor.stride()):
        raise NotImplementedError(f"gumbel_sample {name} has negative strides")
    if (
        tensor.numel()
        and sum(
            (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
        )
        > (1 << 63) - 1
    ):
        raise NotImplementedError(f"gumbel_sample {name} offsets exceed int64")


def _check_vector(name, tensor, device, dtypes, length):
    metadata = _check_tensor(name, tensor, device, dtypes, ndim=1)
    shape = metadata[0] if _COMPILED_FAST_LAUNCH else tensor.shape
    if shape[0] != length:
        raise ValueError(f"gumbel_sample {name} must have length {length}")
    if _COMPILED_FAST_LAUNCH:
        return metadata[1][0]


def _nonoverlapping(tensor):
    """Conservative metadata-only check, including padded/permuted layouts."""
    extent = 1
    for stride, size in sorted(
        (stride, size)
        for size, stride in zip(tensor.shape, tensor.stride())
        if size > 1
    ):
        if stride < extent:
            return False
        extent += (size - 1) * stride
    return True


def _validate_inputs(
    logits,
    expanded_idx_mapping,
    temperature,
    seed,
    pos,
    apply_temperature,
    is_drafting,
    logits_cache,
    logits_cache_col,
    use_fp64,
    logits_cache_source,
    logits_cache_row_mapping,
    bias,
):
    if not isinstance(logits, torch.Tensor):
        raise TypeError("gumbel_sample logits must be a tensor")
    device = logits.device
    if device.type in ("cpu", "meta"):
        raise NotImplementedError("gumbel_sample requires an accelerator device")
    for name, flag in (
        ("apply_temperature", apply_temperature),
        ("is_drafting", is_drafting),
        ("use_fp64", use_fp64),
    ):
        if not isinstance(flag, bool):
            raise TypeError(f"gumbel_sample {name} must be bool")
    logits_metadata = _check_tensor("logits", logits, device, _FLOAT_DTYPES, ndim=2)
    n, vocab = logits_metadata[0] if _COMPILED_FAST_LAUNCH else logits.shape
    if not 1 <= vocab <= 0x7FFFFFFF:
        raise ValueError("gumbel_sample vocab size must be in [1, 2**31-1]")
    if (logits_metadata[1][1] if _COMPILED_FAST_LAUNCH else logits.stride(1)) != 1:
        raise NotImplementedError("gumbel_sample logits last stride must be 1")
    mapping_stride = _check_vector(
        "expanded_idx_mapping", expanded_idx_mapping, device, _INDEX_DTYPES, n
    )
    temperature_metadata = _check_tensor(
        "temperature", temperature, device, (torch.float32,), ndim=1
    )
    requests = (
        temperature_metadata[0][0] if _COMPILED_FAST_LAUNCH else temperature.shape[0]
    )
    seed_stride = _check_vector("seed", seed, device, (torch.int64,), requests)
    pos_stride = _check_vector("pos", pos, device, (torch.int64,), n)
    checked_metadata = (
        (
            n,
            vocab,
            logits_metadata[1][0],
            mapping_stride,
            temperature_metadata[1][0],
            seed_stride,
            pos_stride,
        )
        if _COMPILED_FAST_LAUNCH
        else None
    )
    if bias is not None:
        _check_tensor("bias", bias, device, (logits.dtype,), ndim=2)
        if bias.shape != logits.shape:
            raise ValueError("gumbel_sample bias shape must match logits")
        if bias.stride(1) != 1:
            raise NotImplementedError("gumbel_sample bias last stride must be 1")
    if logits_cache_source is not None:
        if logits_cache is None:
            raise ValueError("logits_cache_source requires logits_cache")
        if bias is not None:
            raise ValueError("gumbel_sample bias and logits_cache_source are exclusive")
    if logits_cache is None:
        return checked_metadata
    _check_tensor("logits_cache", logits_cache, device, _FLOAT_DTYPES, ndim=3)
    if logits_cache.shape[2] < vocab:
        raise ValueError("gumbel_sample logits_cache is narrower than logits")
    if logits_cache.stride(2) != 1 or not _nonoverlapping(logits_cache):
        raise NotImplementedError(
            "gumbel_sample logits_cache layout overlaps or is strided"
        )
    _check_tensor("logits_cache_col", logits_cache_col, device, _INDEX_DTYPES)
    if logits_cache_col.ndim > 1 or (
        logits_cache_col.ndim == 1 and logits_cache_col.shape[0] != n
    ):
        raise ValueError(
            "gumbel_sample logits_cache_col must be scalar or [num_tokens]"
        )
    if logits_cache_source is None:
        if logits_cache.dtype != logits.dtype:
            raise ValueError("gumbel_sample logits and logits_cache dtype must match")
    else:
        _check_tensor(
            "logits_cache_source", logits_cache_source, device, (logits_cache.dtype,), 2
        )
        if logits_cache_source.shape != logits.shape:
            raise ValueError(
                "gumbel_sample logits_cache_source shape must match logits"
            )
    if logits_cache_row_mapping is not None:
        _check_vector(
            "logits_cache_row_mapping",
            logits_cache_row_mapping,
            device,
            _INDEX_DTYPES,
            requests,
        )
    if logits_cache.numel():
        cache_storage = logits_cache.untyped_storage().data_ptr()
        for tensor in (
            logits,
            expanded_idx_mapping,
            temperature,
            seed,
            pos,
            logits_cache_col,
            logits_cache_source,
            logits_cache_row_mapping,
            bias,
        ):
            if tensor is not None and tensor.numel():
                if tensor.untyped_storage().data_ptr() == cache_storage:
                    raise ValueError("gumbel_sample logits_cache must not alias inputs")
    return checked_metadata


def gumbel_sample(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    apply_temperature: bool,
    is_drafting: bool,
    logits_cache: torch.Tensor = None,
    logits_cache_col: torch.Tensor = None,
    use_fp64: bool = False,
    logits_cache_source: torch.Tensor = None,
    logits_cache_row_mapping: torch.Tensor = None,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """Sample one int64 token per logits row without advancing any RNG state.

    Inputs use FP16/BF16/FP32 logits, FP32 temperatures, int64 seeds/positions
    and int32/int64 request mappings. Temperature zero selects the first maximum;
    negative request mappings are greedy padding rows. Positive temperatures
    must be finite. NaN logits (including NaNs produced by bias) are unsupported.
    All -inf rows select zero, and +inf ties select the lowest token index.

    The noise is keyed by request seed, low 32 position bits, and token index.
    Drafting adds 2**30 to position; caching mixes column into the seed high bits.
    FP32 and FP64 modes need not produce identical samples. FP64 requires backend
    compiler/device FP64 support; no precision downgrade is performed.
    Nonempty FP64 calls on the verified Iluvatar Triton 3.1.0 stack raise
    NotImplementedError because its FP64 log/reduction primitives are incorrect.
    That stack derives token keys using an exact integer scan to avoid its
    incorrect lowering of hash arithmetic on range expressions.
    The verified Ascend Triton 3.5.1 SIMD stack also rejects nonempty FP64
    calls: its FP64 conversion, log and reduction primitives do not compile.
    Its FP32 path uses explicit signed shifts and masks to preserve unsigned
    Murmur3 semantics despite incorrect unsigned shift/extension lowering.

    Cache stores pre-temperature logits, or the explicit cache source. Its
    destinations must be unique (row, column) pairs, with valid request/column
    indices. Negative cache row mappings suppress writes. Cache and read inputs
    must not share storage. These value-based bounds/uniqueness preconditions
    belong to the caller; no device-to-host validation or copy is performed.
    Cache calls use a fixed launch, never autotuning the caller's live cache.
    """
    checked_metadata = _validate_inputs(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature,
        is_drafting,
        logits_cache,
        logits_cache_col,
        use_fp64,
        logits_cache_source,
        logits_cache_row_mapping,
        bias,
    )
    n, vocab = checked_metadata[:2] if _COMPILED_FAST_LAUNCH else logits.shape
    legacy_iluvatar = (
        runtime.device.vendor_name == "iluvatar"
        and triton.__version__.split("+", 1)[0] == "3.1.0"
    )
    if n and use_fp64 and legacy_iluvatar:
        raise NotImplementedError(
            "gumbel_sample FP64 log/reduction is unsupported on the verified "
            "Iluvatar Triton 3.1.0 stack"
        )
    if n and use_fp64 and _ASCEND_351:
        raise NotImplementedError(
            "gumbel_sample FP64 is unsupported on the verified Ascend Triton "
            "3.5.1 SIMD stack: FP64 conversion/log/reduction do not compile"
        )
    sampled = torch.empty((n,), dtype=torch.int64, device=logits.device)
    if n == 0:
        return sampled
    if (
        _ASCEND_351
        and _COMPILED_FAST_LAUNCH
        and logits.dtype == torch.float32
        and not use_fp64
        and logits_cache is None
        and logits_cache_col is None
        and logits_cache_source is None
        and logits_cache_row_mapping is None
        and bias is None
        and 64 <= n <= _ASCEND_GRID_LIMIT
        and vocab >= 32768
    ):
        from flaggems_vllm.runtime.backend._ascend.ops.gumbel_sample import (
            _sample_streaming_row,
        )

        return _sample_streaming_row(
            logits,
            expanded_idx_mapping,
            temperature,
            seed,
            pos,
            sampled,
            checked_metadata,
            apply_temperature,
            is_drafting,
        )
    block_size = _BLOCK_SIZE
    if (
        _ASCEND_351
        and apply_temperature
        and logits.dtype == torch.float32
        and logits_cache is None
        and bias is None
        and vocab > _BLOCK_SIZE
    ):
        # The pre-divided (APPLY_TEMPERATURE=False) 4096 variant exceeds UB.
        # Keep its original tile and apply this offline config only as verified.
        block_size = _ascend_sample_config.kwargs["BLOCK_SIZE"]
    blocks = (vocab + block_size - 1) // block_size
    if (
        _COMPILED_FAST_LAUNCH
        and logits_cache is None
        and logits_cache_col is None
        and logits_cache_source is None
        and logits_cache_row_mapping is None
        and bias is None
        and (not _ASCEND_351 or n * blocks <= _ASCEND_GRID_LIMIT)
    ):
        return _sample_nocache(
            logits,
            expanded_idx_mapping,
            temperature,
            seed,
            pos,
            sampled,
            checked_metadata,
            apply_temperature,
            is_drafting,
            use_fp64,
            block_size,
            blocks,
        )
    if blocks == 1:
        local_ids, local_max = None, None
    else:
        local_ids = torch.empty((n, blocks), dtype=torch.int64, device=logits.device)
        local_max = torch.empty(
            (n, blocks),
            dtype=torch.float64 if use_fp64 else torch.float32,
            device=logits.device,
        )
    has_cache = logits_cache is not None
    per_token_col = has_cache and logits_cache_col.ndim == 1
    kernel = _raw_sample_kernel if has_cache else _tuned_sample_kernel
    launch = {}
    if has_cache or runtime.device.vendor_name != "nvidia":
        # Stateful-cache autotune exemption: reuse the exact same JIT body once.
        # Non-NVIDIA configurations start conservatively pending backend evidence.
        launch = {"num_warps": 4, "num_stages": 1}
        if has_cache and runtime.device.vendor_name == "nvidia":
            launch = {
                "num_warps": _sample_configs[0].num_warps,
                "num_stages": _sample_configs[0].num_stages,
            }
    sample_args = (
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        logits_cache,
        logits_cache_col if has_cache else None,
        logits_cache_source,
        logits_cache_row_mapping if has_cache else None,
        bias,
        local_ids,
        local_max,
        sampled,
        *(
            checked_metadata[2:]
            if _COMPILED_FAST_LAUNCH
            else (
                logits.stride(0),
                expanded_idx_mapping.stride(0),
                temperature.stride(0),
                seed.stride(0),
                pos.stride(0),
            )
        ),
        logits_cache.stride(0) if has_cache else 0,
        logits_cache.stride(1) if has_cache else 0,
        logits_cache_col.stride(0) if per_token_col else 0,
        logits_cache_source.stride(0) if logits_cache_source is not None else 0,
        logits_cache_source.stride(1) if logits_cache_source is not None else 0,
        (
            logits_cache_row_mapping.stride(0)
            if has_cache and logits_cache_row_mapping is not None
            else 0
        ),
        bias.stride(0) if bias is not None else 0,
    )
    sample_meta = dict(
        NUM_BLOCKS=blocks,
        VOCAB_SIZE=vocab,
        N_BUCKET=triton.next_power_of_2(n),
        APPLY_TEMPERATURE=apply_temperature,
        IS_DRAFTING=is_drafting,
        USE_FP64=use_fp64,
        USE_SCAN_OFFSETS=legacy_iluvatar,
        HAS_CACHE=has_cache,
        HAS_SOURCE=logits_cache_source is not None,
        HAS_ROW_MAPPING=has_cache and logits_cache_row_mapping is not None,
        HAS_BIAS=bias is not None,
        PER_TOKEN_COL=per_token_col,
        BLOCK_SIZE=block_size,
        **launch,
    )
    combine_args = (local_ids, blocks, local_max, blocks, sampled, blocks)
    combine_meta = dict(
        PADDED_NUM_BLOCKS=triton.next_power_of_2(blocks), num_warps=4, num_stages=1
    )
    with runtime.torch_device_fn.device(logits.device):
        if _ASCEND_351:
            # This stack caps coreDim at 65535, and its auto-blockify option
            # truncates work instead of inserting a loop. Keep global IDs in
            # the shared body and split only the launch, including wide vocab.
            sample_meta.update(USE_GRID_OFFSETS=True, enable_auto_blockify=False)
            combine_meta.update(USE_GRID_OFFSETS=True, enable_auto_blockify=False)
            block_chunk = min(blocks, _ASCEND_GRID_LIMIT)
            row_chunk = max(1, _ASCEND_GRID_LIMIT // block_chunk)
            for row_start in range(0, n, row_chunk):
                rows = min(row_chunk, n - row_start)
                for block_start in range(0, blocks, block_chunk):
                    columns = min(block_chunk, blocks - block_start)
                    kernel[(rows, columns)](
                        *sample_args,
                        **sample_meta,
                        grid_row_start=row_start,
                        grid_block_start=block_start,
                    )
                if blocks > 1:
                    _combine_kernel[(rows,)](
                        *combine_args, **combine_meta, grid_row_start=row_start
                    )
        else:
            kernel[(n, blocks)](*sample_args, **sample_meta)
            if blocks > 1:
                _combine_kernel[(n,)](*combine_args, **combine_meta)
    return sampled
