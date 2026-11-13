# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# INT8 state arithmetic follows the public contracts from PRs #1037 and #1040.

import hashlib
import logging
import math
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.backends.ascend.driver import NPUUtils
from triton.runtime.cache import get_cache_manager

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._ascend.ops.chunk_gated_delta_rule_fwd import (
    SINGLE_KERNEL_MAX_TOTAL_CHUNKS,
    _chunk_gated_delta_rule_fwd_stage5_6_with_intermediates,
    _get_aic_num,
    _map_flattened_varlen_block,
    _transpose_gate_beta_kernel,
    chunk_gated_delta_rule_fwd_stage0_4,
    chunk_local_cumsum,
    merge_pair_16x16_to_64x64_inverse_kernel,
    safe_exp,
    solve_tril_16x16_kernel,
    stage4_chunk_pair_combined128,
)

try:
    import triton.experimental.tle.language as tle
except ImportError:
    tle = None


logger = logging.getLogger(__name__)


@triton.jit(do_not_specialize=["TOTAL"])
def prepare_decode_kernel(
    q,
    k,
    a,
    b,
    a_log,
    bias,
    q_out,
    k_out,
    decay_out,
    beta_out,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    STRIDE_Q: tl.constexpr,
    STRIDE_K: tl.constexpr,
    OFFSET_K: tl.constexpr,
    STRIDE_A: tl.constexpr,
    STRIDE_B: tl.constexpr,
    NORMALIZE: tl.constexpr,
    BK: tl.constexpr,
    TOTAL,
):
    query_rows = (TOTAL // HV) * H
    norm_blocks = tl.cdiv(query_rows, 16)
    gate_blocks = tl.cdiv(TOTAL, 128)
    for task in range(
        tl.program_id(0), tl.maximum(norm_blocks, gate_blocks), tl.num_programs(0)
    ):
        if task < norm_blocks:
            rows = task * 16 + tl.arange(0, 16)
            token, head = rows // H, rows % H
            reduction = tl.arange(0, BK)
            mask = (rows[:, None] < query_rows) & (reduction[None, :] < K)
            query = tl.load(
                q + token[:, None] * STRIDE_Q + head[:, None] * K + reduction[None, :],
                mask,
                0,
            ).to(tl.float32)
            key = tl.load(
                k
                + token[:, None] * STRIDE_K
                + OFFSET_K
                + head[:, None] * K
                + reduction[None, :],
                mask,
                0,
            ).to(tl.float32)
            if NORMALIZE:
                query *= tl.rsqrt(tl.sum(query * query, 1) + 1e-6)[:, None]
                key *= tl.rsqrt(tl.sum(key * key, 1) + 1e-6)[:, None]
            tl.store(q_out + rows[:, None] * K + reduction[None, :], query, mask)
            tl.store(k_out + rows[:, None] * K + reduction[None, :], key, mask)
        if task < gate_blocks:
            rows = task * 128 + tl.arange(0, 128)
            token, head = rows // HV, rows % HV
            gate = tl.load(a + token * STRIDE_A + head, rows < TOTAL, 0).to(tl.float32)
            gate += tl.load(bias + head).to(tl.float32)
            softplus = tl.where(gate <= 20.0, tl.log(1.0 + tl.exp(gate)), gate)
            decay = tl.exp(-tl.exp(tl.load(a_log + head).to(tl.float32)) * softplus)
            beta = tl.sigmoid(
                tl.load(b + token * STRIDE_B + head, rows < TOTAL, 0).to(tl.float32)
            )
            tl.store(decay_out + rows, decay, rows < TOTAL)
            tl.store(beta_out + rows, beta, rows < TOTAL)


@triton.jit(do_not_specialize=["TOTAL"])
def transform_state_kernel(
    source,
    target,
    scales,
    K: tl.constexpr,
    V: tl.constexpr,
    TOTAL,
    BK: tl.constexpr,
    BV: tl.constexpr,
    DEQUANTIZE: tl.constexpr,
    TRANSPOSE: tl.constexpr,
):
    for task in range(tl.program_id(0), TOTAL, tl.num_programs(0)):
        group = (task // tl.cdiv(V, BV)).to(tl.int64)
        column = (task % tl.cdiv(V, BV)) * BV + tl.arange(0, BV)
        reduction = tl.arange(0, BK)
        offsets = (group * V + column[:, None]) * K + reduction[None, :]
        transposed = (group * K + reduction[None, :]) * V + column[:, None]
        mask = (column[:, None] < V) & (reduction[None, :] < K)
        source_offsets = transposed if TRANSPOSE else offsets
        values = tl.load(source + source_offsets, mask, 0).to(tl.float32)
        if DEQUANTIZE:
            scale = tl.load(scales + group * V + column, column < V, 0)
            tl.store(target + transposed, values * scale[:, None], mask)
        else:
            store_int8_state(
                values, target, scales, offsets, group * V + column, mask, column < V
            )


@triton.jit(do_not_specialize=["ROWS"])
def normalize_qk_kernel(q, k, q_out, k_out, ROWS, K: tl.constexpr, BK: tl.constexpr):
    for block in range(tl.program_id(0), tl.cdiv(ROWS, 32), tl.num_programs(0)):
        rows = block * 32 + tl.arange(0, 32)
        offsets = rows[:, None] * K + tl.arange(0, BK)[None, :]
        mask = (rows[:, None] < ROWS) & (tl.arange(0, BK)[None, :] < K)
        query = tl.load(q + offsets, mask, 0).to(tl.float32)
        key = tl.load(k + offsets, mask, 0).to(tl.float32)
        query *= tl.rsqrt(tl.sum(query * query, 1) + 1e-6)[:, None]
        key *= tl.rsqrt(tl.sum(key * key, 1) + 1e-6)[:, None]
        tl.store(q_out + offsets, query, mask)
        tl.store(k_out + offsets, key, mask)


@triton.jit
def cast_metadata_kernel(source, target, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(
        target + offsets,
        tl.load(source + offsets, offsets < N, 0).to(tl.int64),
        offsets < N,
    )


@lru_cache(maxsize=1)
def vector_cores() -> int:
    return NPUUtils().get_aivector_core_num()


def quantize_gdn_state_int8(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [slots,HV,V,K] with nearest-even rounding and FP32 scales per V row."""
    logger.debug("GEMS_ASCEND QUANTIZE_GDN_STATE_INT8")
    if state.ndim != 4 or not state.is_contiguous():
        raise ValueError("state must be contiguous [slots,HV,V,K]")
    if state.device.type != "npu" or state.requires_grad:
        raise ValueError("state must be an inference-only NPU tensor")
    if state.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError("state must be FP32, FP16 or BF16")
    slots, heads, columns, reduction = state.shape
    if min(heads, columns, reduction) < 1:
        raise ValueError("state head and feature dimensions must be positive")
    output = torch.empty(state.shape, device=state.device, dtype=torch.int8)
    scales = torch.empty(
        (slots, heads, columns), device=state.device, dtype=torch.float32
    )
    if state.numel() == 0:
        return output, scales
    total = slots * heads * triton.cdiv(columns, 16)
    with torch_device_fn.device(state.device):
        transform_state_kernel[(min(vector_cores(), total),)](
            state,
            output,
            scales,
            reduction,
            columns,
            total,
            triton.next_power_of_2(reduction),
            16,
            False,
            False,
        )
    return output, scales


@tl.core.builtin
def cast_integral_half_to_int8(value, _semantic=None):
    # Ascend's standard FP16 cast promotes to FP32 before converting to INT8.
    # Emit the direct conversion; this helper only receives exact INT8 integers.
    dtype = value.type.with_element_ty(tl.int8)
    handle = _semantic.builder.create_fp_to_si(
        value.handle, dtype.to_ir(_semantic.builder)
    )
    return tl.tensor(handle, dtype)


# The generic integer cast preserves overflow with several extra vector passes.
# This private conversion only receives rounded integers in [-127,127].
_GDN_CAST_SOURCE = r"""
// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include <stdint.h>
using namespace __cce_scalar;

template <typename T> struct MemRef1D {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[1];
  int64_t strides[1];
};

// Input values are integral FP16 values in [-127,127].
extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_gdn_bounded_fp16_to_words(MemRef1D<__ubuf__ half> *src,
                                    int32_t count,
                                    MemRef1D<__ubuf__ uint32_t> *dst) {
  set_mask_count();
  set_vector_mask(0, static_cast<uint64_t>(count));
  auto *bytes = reinterpret_cast<__ubuf__ int8_t *>(dst->aligned + dst->offset);
  vconv_f162s8z(bytes, src->aligned + src->offset, 1, 1, 1, 4, 8);
  set_mask_norm();
  set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
}
"""
_GDN_CAST_FLAGS = (
    "-O2",
    "-x",
    "cce",
    "--cce-auto-sync=off",
    "--cce-aicore-only",
    "--cce-generic-addrspace=off",
    "-mllvm",
    "-disable-llvm-optzns",
    "--cce-aicore-arch=dav-c220-vec",
    "-std=c++17",
    "-emit-llvm",
    "-c",
)
_GDN_CAST_DIGEST = tl.constexpr(
    hashlib.sha256((_GDN_CAST_SOURCE + repr(_GDN_CAST_FLAGS)).encode()).hexdigest()
)


@lru_cache(maxsize=1)
def native_word_cast_available():
    return (
        tle is not None
        and NPUUtils().get_arch().startswith("Ascend910B")
        and shutil.which("ccec") is not None
    )


@lru_cache(maxsize=1)
def native_word_cast_bitcode():
    compiler = shutil.which("ccec")
    if compiler is None:
        raise NotImplementedError("The native GDN cast requires the CANN ccec compiler")
    version = subprocess.check_output([compiler, "--version"], stderr=subprocess.STDOUT)
    key = hashlib.sha256(
        _GDN_CAST_DIGEST.value.encode() + compiler.encode() + version
    ).hexdigest()
    cache = get_cache_manager(key)
    filename = "gdn_bounded_int8_words.bc"
    cached = cache.get_file(filename)
    if cached:
        return cached
    with tempfile.TemporaryDirectory(prefix="gdn-int8-cast-") as directory:
        source = Path(directory) / "cast.cpp"
        output = Path(directory) / filename
        source.write_text(_GDN_CAST_SOURCE)
        try:
            subprocess.run(
                [compiler, *_GDN_CAST_FLAGS, str(source), "-o", str(output)],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(error.stderr.decode(errors="replace")) from error
        return cache.put(output.read_bytes(), filename, binary=True)


@al.register_custom_op
class _GdnBoundedInt8Words:
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD

    def __init__(self, source, count, out=None):
        assert out is not None
        self.symbol = "gdn_bounded_fp16_to_words"
        self.bitcode = native_word_cast_bitcode()


if tle is not None:

    @triton.jit
    def bounded_int8_words(value):
        tl.static_assert(value.dtype == tl.float16 and value.numel % 4 == 0)
        # Include native source changes in the enclosing Triton kernel cache key.
        tl.static_assert(_GDN_CAST_DIGEST != "")
        # The conversion writes every output byte, so initialization is unnecessary.
        target = tle.dsa.alloc([value.numel // 4], tl.uint32, tle.dsa.ascend.UB)
        return al.custom(
            "_GdnBoundedInt8Words",
            tl.reshape(value, (value.numel,)),
            value.numel,
            out=tle.dsa.to_tensor(target),
        )

else:

    @triton.jit
    def bounded_int8_words(value):
        tl.static_assert(False, "The native GDN conversion requires FlagTree TLE")


@triton.jit
def store_int8_state(
    values,
    state,
    scales,
    offsets,
    scale_offsets,
    mask,
    mask_v,
    WORD_CAST: tl.constexpr = False,
    word_state=None,
):
    magnitude = tl.max(tl.abs(values), axis=1)
    safe_magnitude = tl.where(magnitude > 0, magnitude, 127.0)
    scale = safe_magnitude * (1.0 / 127.0)
    reciprocal_scale = 127.0 / safe_magnitude
    # FP32 bias rounding produces exact integers in [-127,127]. They remain
    # exact in FP16, which avoids the expensive overflow-preserving INT32 cast.
    rounded = (values * reciprocal_scale[:, None] + 12582912.0) - 12582912.0
    if WORD_CAST:
        tl.static_assert(values.shape[1] == 128)
        packed = tl.reshape(
            bounded_int8_words(rounded.to(tl.float16)), (values.shape[0], 32)
        )
        word_offsets = scale_offsets[:, None] * 32 + tl.arange(0, 32)[None, :]
        tl.store(
            word_state + word_offsets,
            packed.to(tl.int32, bitcast=True),
            mask_v[:, None],
        )
    else:
        tl.store(
            state + offsets, cast_integral_half_to_int8(rounded.to(tl.float16)), mask
        )
    tl.store(scales + scale_offsets, scale, mask_v)


@triton.jit(do_not_specialize=["TOTAL"])
def gdn_decode_int8_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    a,
    b,
    A_log,
    dt_bias,
    state,
    scales,
    output,
    indices,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    STRIDE_Q: tl.constexpr,
    STRIDE_K: tl.constexpr,
    STRIDE_V: tl.constexpr,
    OFFSET_K: tl.constexpr,
    OFFSET_V: tl.constexpr,
    STRIDE_A: tl.constexpr,
    STRIDE_B: tl.constexpr,
    SCALE: tl.constexpr,
    NORMALIZE: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    TOTAL=1,
    PREPARED: tl.constexpr = False,
    WORD_CAST: tl.constexpr = False,
    word_state=None,
):
    for task in range(tl.program_id(0), TOTAL, tl.num_programs(0)):
        sequence_head = (task // tl.cdiv(V, BV)).to(tl.int64)
        sequence, head = sequence_head // HV, sequence_head % HV
        query_head = head // (HV // H)
        column = (task % tl.cdiv(V, BV)) * BV + tl.arange(0, BV)
        reduction = tl.arange(0, BK)
        slot = tl.load(indices + sequence).to(tl.int64)
        output_ptr = output + (sequence * HV + head) * V + column
        # Slot zero is the null block in vLLM packed decode.
        if slot <= 0:
            tl.store(output_ptr, 0.0, column < V)
        else:
            group = slot * HV + head
            scale_offsets = group * V + column
            offsets = (group * V + column[:, None]) * K + reduction[None, :]
            mask = (column[:, None] < V) & (reduction[None, :] < K)
            state_values = tl.load(state + offsets, mask, 0).to(tl.float32)
            state_scale = tl.load(scales + scale_offsets, column < V, 0)
            state_values *= state_scale[:, None]
            query = tl.load(
                query_ptr + sequence * STRIDE_Q + query_head * K + reduction,
                reduction < K,
                0,
            ).to(tl.float32)
            key = tl.load(
                key_ptr + sequence * STRIDE_K + OFFSET_K + query_head * K + reduction,
                reduction < K,
                0,
            ).to(tl.float32)
            value = tl.load(
                value_ptr + sequence * STRIDE_V + OFFSET_V + head * V + column,
                column < V,
                0,
            ).to(tl.float32)
            if NORMALIZE and not PREPARED:
                query *= tl.rsqrt(tl.sum(query * query, 0) + 1e-6)
                key *= tl.rsqrt(tl.sum(key * key, 0) + 1e-6)
            query *= SCALE
            if PREPARED:
                decay = tl.load(a + sequence * STRIDE_A + head)
                beta = tl.load(b + sequence * STRIDE_B + head)
            else:
                gate = tl.load(a + sequence * STRIDE_A + head).to(tl.float32)
                gate += tl.load(dt_bias + head).to(tl.float32)
                softplus = tl.where(gate <= 20.0, tl.log(1.0 + tl.exp(gate)), gate)
                decay = tl.exp(-tl.exp(tl.load(A_log + head).to(tl.float32)) * softplus)
                beta = tl.sigmoid(
                    tl.load(b + sequence * STRIDE_B + head).to(tl.float32)
                )
            state_values *= decay
            residual = (value - tl.sum(state_values * key[None, :], 1)) * beta
            state_values = tl.fma(residual[:, None], key[None, :], state_values)
            result = tl.sum(state_values * query[None, :], 1)
            tl.store(output_ptr, result, column < V)
            store_int8_state(
                state_values,
                state,
                scales,
                offsets,
                scale_offsets,
                mask,
                column < V,
                WORD_CAST,
                word_state,
            )


def gdn_w8a16_int8(
    mixed_qkv: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    state_int8: torch.Tensor,
    state_scale: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-token-per-sequence GDN with in-place INT8 state.

    State layout follows vLLM: [slots, HV, V, K]. Scales are [slots, HV, V].
    Positive indices must be distinct and within the state allocation; nonpositive
    indices are null slots. Callers retain ownership of state and scale buffers.
    Output must not overlap read-only inputs.
    QKV can be packed [T,2*H*K+HV*V] or a tuple of contiguous Q/K[1,T,H,K]
    and V[1,T,HV,V] tensors. The tuple route does not allocate a packed buffer.
    """
    logger.debug("GEMS_ASCEND GDN_W8A16_INT8")
    if not isinstance(scale, (float, int)) or not isinstance(
        use_qk_l2norm_in_kernel, bool
    ):
        raise TypeError("scale and normalization must be Python numeric/bool scalars")
    if state_int8.ndim != 4:
        raise ValueError("expected four-dimensional state")
    slots, heads, columns, reduction = state_int8.shape
    if columns < 1 or reduction < 1 or heads < 1 or slots < 1:
        raise ValueError("state dimensions must be positive")
    if isinstance(mixed_qkv, tuple):
        if len(mixed_qkv) != 3:
            raise ValueError("separate QKV must contain three tensors")
        query, key, value = mixed_qkv
        activation_tensors = mixed_qkv
        if query.ndim != 4 or query.shape != key.shape or query.shape[0] != 1:
            raise ValueError("separate Q/K must have matching [1,T,H,K] shapes")
        tokens, query_heads = query.shape[1:3]
        if query.shape[3] != reduction or value.shape != (1, tokens, heads, columns):
            raise ValueError("separate QKV dimensions do not match the state")
        stride_q, stride_k, stride_v = query.stride(1), key.stride(1), value.stride(1)
        offset_k, offset_v = 0, 0
    else:
        if mixed_qkv.ndim != 2:
            raise ValueError("packed QKV must be two-dimensional")
        tokens, packed = mixed_qkv.shape
        query_dim = packed - heads * columns
        if query_dim <= 0 or query_dim % (2 * reduction):
            raise ValueError("packed QKV dimensions do not match the state")
        query_heads = query_dim // (2 * reduction)
        query = key = value = mixed_qkv
        activation_tensors = (mixed_qkv,)
        stride_q = stride_k = stride_v = mixed_qkv.stride(0)
        offset_k, offset_v = query_heads * reduction, 2 * query_heads * reduction
    if query_heads < 1 or heads % query_heads:
        raise ValueError("value heads must be divisible by query heads")
    if query.dtype not in (torch.bfloat16, torch.float16) or any(
        tensor.dtype != query.dtype for tensor in activation_tensors
    ):
        raise TypeError("QKV must share a BF16 or FP16 dtype")
    if state_int8.dtype != torch.int8 or state_scale.dtype != torch.float32:
        raise TypeError("state and scales must be INT8 and FP32")
    if state_scale.shape != (slots, heads, columns):
        raise ValueError("state scales must have shape [slots, HV, V]")
    if a.shape != (tokens, heads) or b.shape != a.shape:
        raise ValueError("a and b must have shape [tokens, HV]")
    if a.dtype != query.dtype or b.dtype != query.dtype:
        raise TypeError("gate activations must match QKV dtype")
    if A_log.shape != (heads,) or dt_bias.shape != (heads,):
        raise ValueError("A_log and dt_bias must have shape [HV]")
    if A_log.dtype != torch.float32 or dt_bias.dtype != torch.float32:
        raise TypeError("A_log and dt_bias must be FP32")
    if out.shape != (tokens, 1, heads, columns) or out.dtype != query.dtype:
        raise ValueError("out must match activation dtype and shape [tokens, 1, HV, V]")
    if ssm_state_indices.shape != (tokens,) or ssm_state_indices.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("state indices must be an integer vector of length tokens")
    tensors = (
        *activation_tensors,
        a,
        b,
        A_log,
        dt_bias,
        state_int8,
        state_scale,
        out,
        ssm_state_indices,
    )
    if query.device.type != "npu" or any(
        tensor.device != query.device for tensor in tensors
    ):
        raise ValueError("all tensors must be on the same accelerator")
    if any(tensor.requires_grad for tensor in tensors):
        raise NotImplementedError("GDN INT8 state is forward-only")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all buffers must be contiguous")
    if not tokens:
        return out, state_int8
    block_v = 128 if reduction == columns == 128 else 16
    total = tokens * heads * triton.cdiv(columns, block_v)
    # Shared normalization/gating amortizes its extra launch on larger batches.
    prepared = reduction == columns == 128 and tokens >= 16
    with torch_device_fn.device(query.device):
        # Word stores preserve the INT8 cache layout; byte-unaligned caches use
        # the pure Triton conversion instead of reinterpreting their pointers.
        word_cast = (
            reduction == columns == 128
            and state_int8.data_ptr() % 4 == 0
            and state_int8.storage_offset() % 4 == 0
            and native_word_cast_available()
        )
        if prepared:
            normalized_q = torch.empty(
                (tokens, query_heads, reduction),
                device=query.device,
                dtype=torch.float32,
            )
            normalized_k = torch.empty_like(normalized_q)
            decay = torch.empty(
                (tokens, heads), device=query.device, dtype=torch.float32
            )
            beta = torch.empty_like(decay)
            prepare_blocks = max(
                triton.cdiv(tokens * query_heads, 16), triton.cdiv(tokens * heads, 128)
            )
            prepare_decode_kernel[(min(vector_cores(), prepare_blocks),)](
                query,
                key,
                a,
                b,
                A_log,
                dt_bias,
                normalized_q,
                normalized_k,
                decay,
                beta,
                query_heads,
                heads,
                reduction,
                stride_q,
                stride_k,
                offset_k,
                a.stride(0),
                b.stride(0),
                use_qk_l2norm_in_kernel,
                triton.next_power_of_2(reduction),
                tokens * heads,
            )
            query, key, a, b = normalized_q, normalized_k, decay, beta
            stride_q = stride_k = query_heads * reduction
            offset_k = 0
        gdn_decode_int8_kernel[(min(vector_cores(), total),)](
            query,
            key,
            value,
            a,
            b,
            A_log,
            dt_bias,
            state_int8,
            state_scale,
            out,
            ssm_state_indices,
            query_heads,
            heads,
            reduction,
            columns,
            stride_q,
            stride_k,
            stride_v,
            offset_k,
            offset_v,
            a.stride(0),
            b.stride(0),
            float(scale),
            use_qk_l2norm_in_kernel,
            triton.next_power_of_2(reduction),
            block_v,
            TOTAL=total,
            PREPARED=prepared,
            WORD_CAST=word_cast,
            # A dtype view only reinterprets storage; it launches no torch kernel.
            word_state=state_int8.view(torch.int32) if word_cast else None,
        )
    return out, state_int8


# INT8-only preprocessing; shared GDN kernels and dispatch remain untouched.
@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_G": lambda args: args["g_cumsum"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "B"])
def _int8_prefill_kkt_kernel(
    k,
    beta,  # [H, B, T]
    g_cumsum,  # [H, B, T]
    A,
    cu_seqlens,
    T,
    B,
    N,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
):
    bt_stride = B * T
    i_t_i, _ = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        valid, bos, T, i_t = _map_flattened_varlen_block(
            i_t_i,
            cu_seqlens,
            N,
            BT,
        )
    else:
        valid, bos, i_t = True, 0, i_t_i

    if valid:
        HEAD_PAIR: tl.constexpr = 2 if H == 2 * Hg else 1
        for i_group in range(B * H // HEAD_PAIR):
            i_b = i_group // (H // HEAD_PAIR)
            first_head = (i_group % (H // HEAD_PAIR)) * HEAD_PAIR
            i_kh = first_head // (H // Hg)
            if not IS_VARLEN:
                bos = i_b * T
            o_t = tl.arange(0, BT)
            o_t_fp32 = o_t.to(tl.float32)

            b_gram = tl.zeros([BT, BT], dtype=tl.float32)
            for i_k in range(tl.cdiv(K, BK)):
                p_k = tl.make_block_ptr(
                    k + (bos * Hg + i_kh) * K,
                    (T, K),
                    (Hg * K, 1),
                    (i_t * BT, i_k * BK),
                    (BT, BK),
                    (1, 0),
                )
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_gram += tl.dot(b_k, tl.trans(b_k))

            # Value heads in a GQA group share the same K @ K.T tile.
            for group_head in tl.static_range(HEAD_PAIR):
                i_h = first_head + group_head
                b_A = b_gram
                p_beta = tl.make_block_ptr(
                    beta + i_h * bt_stride + bos, (T,), (1,), (i_t * BT,), (BT,), (0,)
                )
                b_beta = tl.load(p_beta, boundary_check=(0,))

                if USE_G:
                    p_g = tl.make_block_ptr(
                        g_cumsum + i_h * bt_stride + bos,
                        (T,),
                        (1,),
                        (i_t * BT,),
                        (BT,),
                        (0,),
                    )
                    b_g = tl.load(p_g, boundary_check=(0,))
                    b_g_diff = b_g[:, None] - b_g[None, :]
                    b_A *= safe_exp(b_g_diff)

                b_A *= b_beta[:, None]
                b_A = tl.where(o_t_fp32[:, None] > o_t_fp32[None, :], b_A, 0)
                p_A = tl.make_block_ptr(
                    A + (bos * H + i_h) * BT,
                    (T, BT),
                    (BT * H, 1),
                    (i_t * BT, 0),
                    (BT, BT),
                    (1, 0),
                )
                tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


def _int8_prefill_stage0_4(k, v, g, beta, cu_seqlens):
    _, tokens, query_heads, reduction = k.shape
    heads = v.shape[2]
    sequences = 1 if cu_seqlens is None else cu_seqlens.numel() - 1
    chunks = triton.cdiv(tokens, 64) + sequences - 1
    if (
        cu_seqlens is None
        or heads != 2 * query_heads
        or _get_aic_num() != 20
        or chunks <= SINGLE_KERNEL_MAX_TOTAL_CHUNKS
    ):
        return chunk_gated_delta_rule_fwd_stage0_4(
            k, v, g, beta, cu_seqlens=cu_seqlens, chunk_size=64
        )
    g_cumsum = chunk_local_cumsum(
        g, chunk_size=64, cu_seqlens=cu_seqlens, output_dtype=torch.float32
    )
    g_perm = torch.empty((heads, 1, tokens), device=g.device, dtype=torch.float32)
    beta_perm = torch.empty_like(g_perm, dtype=beta.dtype)
    _transpose_gate_beta_kernel[(triton.cdiv(tokens, 128), 1)](
        g_cumsum,
        beta,
        g_perm,
        beta_perm,
        tokens,
        1,
        H=heads,
        BLOCK_T=128,
        num_warps=4,
        num_stages=1,
    )
    a = torch.empty((1, tokens, heads, 64), device=k.device, dtype=torch.float32)
    _int8_prefill_kkt_kernel[(chunks, 1)](
        k,
        beta_perm,
        g_perm,
        a,
        cu_seqlens,
        tokens,
        1,
        sequences,
        H=heads,
        Hg=query_heads,
        K=reduction,
        BT=64,
        BK=128,
        num_warps=8,
        num_stages=3,
        multibuffer=True,
    )
    # Retain FP32 inversion; only this INT8 call uses the 1024-row geometry.
    ad = torch.empty((1, tokens, heads, 16), device=k.device, dtype=torch.float32)
    solve_tril_16x16_kernel[(triton.cdiv(tokens, 1024) + sequences - 1, heads)](
        a,
        ad,
        cu_seqlens,
        tokens,
        heads,
        sequences,
        BT=64,
        LARGE_BLOCK_T=1024,
        num_warps=1,
        num_stages=4,
    )
    a_inv = torch.empty_like(a, dtype=k.dtype)
    merge_pair_16x16_to_64x64_inverse_kernel[
        (triton.cdiv(tokens, 128) + sequences - 1, heads)
    ](
        a,
        ad,
        a_inv,
        cu_seqlens,
        tokens,
        heads,
        sequences,
        BT=64,
        num_warps=8,
        num_stages=1,
    )
    w, u = stage4_chunk_pair_combined128(k, v, beta_perm, g_perm, a_inv, cu_seqlens)
    return w, u, g_cumsum


def chunk_gated_delta_rule_w8a16_int8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_int8: torch.Tensor,
    state_scale: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefill with INT8 [N,HV,V,K] state and FP32 per-channel scales.

    Packed variable-length calls provide consistent 64-token chunk metadata.
    State is dequantized once, accumulated across all chunks, and requantized
    once at the end. This entry is forward-only and mutates state and scales.
    """
    logger.debug("GEMS_ASCEND CHUNK_GATED_DELTA_RULE_W8A16_INT8")
    if scale is not None and not isinstance(scale, (float, int)):
        raise TypeError("scale must be a Python numeric scalar or None")
    if not isinstance(use_qk_l2norm_in_kernel, bool):
        raise TypeError("normalization must be a Python bool")
    if q.ndim != 4 or q.shape != k.shape or v.ndim != 4:
        raise ValueError("expected Q/K[B,T,H,K] and V[B,T,HV,V]")
    batch, tokens, query_heads, reduction = q.shape
    heads, columns = v.shape[-2:]
    if batch != 1 or reduction != 128 or columns != 128:
        raise NotImplementedError("prefill supports packed B=1 and K=V=128")
    if query_heads < 1 or heads % query_heads or v.shape[:2] != q.shape[:2]:
        raise ValueError("invalid QKV head or token dimensions")
    if (
        q.dtype not in (torch.bfloat16, torch.float16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise TypeError("QKV must share a BF16 or FP16 dtype")
    if g.shape != (batch, tokens, heads) or beta.shape != g.shape:
        raise ValueError("g and beta must have shape [1,T,HV]")
    if g.dtype != torch.float32 or beta.dtype != q.dtype:
        raise TypeError("log gates must be FP32 and beta must match QKV dtype")
    sequences = batch if cu_seqlens is None else cu_seqlens.numel() - 1
    if sequences < 1 or state_int8.shape != (sequences, heads, columns, reduction):
        raise ValueError("state must have shape [N,HV,V,K]")
    if state_int8.dtype != torch.int8 or state_scale.dtype != torch.float32:
        raise TypeError("state and scales must be INT8 and FP32")
    if state_scale.shape != (sequences, heads, columns):
        raise ValueError("state scales must have shape [N,HV,V]")
    if cu_seqlens is None:
        if chunk_indices is not None or chunk_offsets is not None:
            raise ValueError("chunk metadata requires cu_seqlens")
        metadata = ()
    else:
        if chunk_indices is None or chunk_offsets is None:
            raise ValueError("varlen prefill requires chunk_indices and chunk_offsets")
        if (
            cu_seqlens.ndim != 1
            or chunk_indices.ndim != 2
            or chunk_indices.shape[1] != 2
            or chunk_offsets.shape != (sequences + 1,)
        ):
            raise ValueError("invalid variable-length metadata shapes")
        metadata = (cu_seqlens, chunk_indices, chunk_offsets)
        if any(tensor.dtype not in (torch.int32, torch.int64) for tensor in metadata):
            raise TypeError("chunk metadata must be integer tensors")
    tensors = (q, k, v, g, beta, state_int8, state_scale, *metadata)
    if q.device.type != "npu" or any(
        tensor.device != q.device or not tensor.is_contiguous() for tensor in tensors
    ):
        raise ValueError("all inputs must be contiguous and on one accelerator")
    if any(tensor.requires_grad for tensor in tensors):
        raise NotImplementedError("INT8-state GDN is forward-only")
    if tokens == 0:
        return torch.empty(v.shape, device=q.device, dtype=q.dtype), state_int8
    launch_scale = reduction**-0.5 if scale is None else float(scale)
    if not math.isfinite(launch_scale):
        raise ValueError("scale must be finite")
    with torch_device_fn.device(q.device):
        initial = torch.empty(
            (sequences, heads, reduction, columns), device=q.device, dtype=torch.float32
        )
        total = sequences * heads * triton.cdiv(columns, 16)
        transform_state_kernel[(min(vector_cores(), total),)](
            state_int8,
            initial,
            state_scale,
            reduction,
            columns,
            total,
            reduction,
            16,
            True,
            False,
        )
        if use_qk_l2norm_in_kernel:
            q_norm, k_norm = torch.empty_like(q), torch.empty_like(k)
            normalize_qk_kernel[
                (min(vector_cores(), triton.cdiv(tokens * query_heads, 32)),)
            ](
                q,
                k,
                q_norm,
                k_norm,
                tokens * query_heads,
                reduction,
                reduction,
            )
        else:
            q_norm, k_norm = q, k
        if cu_seqlens is not None and cu_seqlens.dtype != torch.int64:
            cu_long = torch.empty(cu_seqlens.shape, device=q.device, dtype=torch.int64)
            cast_metadata_kernel[(triton.cdiv(cu_seqlens.numel(), 256),)](
                cu_seqlens,
                cu_long,
                cu_seqlens.numel(),
                256,
            )
        else:
            cu_long = cu_seqlens
        w, u, g_cumsum = _int8_prefill_stage0_4(k_norm, v, g, beta, cu_long)
        output, final, _, _ = _chunk_gated_delta_rule_fwd_stage5_6_with_intermediates(
            q_norm,
            k_norm,
            w,
            u,
            g_cumsum,
            scale=launch_scale,
            initial_state=initial,
            output_final_state=True,
            cu_seqlens=cu_long,
        )
        transform_state_kernel[(min(vector_cores(), total),)](
            final,
            state_int8,
            state_scale,
            reduction,
            columns,
            total,
            reduction,
            16,
            False,
            True,
        )
    return output, state_int8
