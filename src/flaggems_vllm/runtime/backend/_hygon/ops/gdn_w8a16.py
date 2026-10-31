# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# INT8 state contracts adapted from FlagGems-vllm #1037.

import logging
from collections import OrderedDict
from threading import Lock

import flag_gems
import torch
import triton
import triton.language as tl
from triton.compiler import CompiledKernel

from flaggems_vllm.ops.FLA.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
)
from flaggems_vllm.ops.FLA.chunk_o import chunk_fwd_kernel_o
from flaggems_vllm.ops.FLA.fused_cumsum_kkt_solve_tril import (
    chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril_kernel,
)
from flaggems_vllm.ops.FLA.triton_ops_helper import exp
from flaggems_vllm.ops.FLA.wy_fast import recompute_w_u_fwd_kernel
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.utils import libentry

logger = logging.getLogger(__name__)
CHUNK_SIZE = 64
DECODE_CACHE_SIZE = 32
decode_launch_cache: OrderedDict[tuple, CompiledKernel] = OrderedDict()
decode_launch_lock = Lock()

# Bypass runtime autotuning: these launches share mutable recurrence workspaces.
chunk_prepare = chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril_kernel.jit_function
chunk_wy = recompute_w_u_fwd_kernel.jit_function
chunk_state = chunk_gated_delta_rule_fwd_kernel_h_blockdim64.jit_function
chunk_output = chunk_fwd_kernel_o.jit_function


@triton.jit
def store_int8_state(values, state, scales, offsets, scale_offsets, mask, mask_v):
    magnitude = tl.max(tl.abs(values), axis=1)
    safe_magnitude = tl.where(magnitude > 0, magnitude, 127.0)
    scale = safe_magnitude * (1.0 / 127.0)
    reciprocal_scale = 127.0 / safe_magnitude
    # Share one division across the channel instead of dividing every element.
    quantized = tl.minimum(
        tl.maximum(tl.floor(values * reciprocal_scale[:, None] + 0.5), -127), 127
    )
    tl.store(state + offsets, quantized.to(tl.int8), mask)
    tl.store(scales + scale_offsets, scale, mask_v)


@libentry()
@triton.jit
def quantize_prefill_state(
    source,
    state,
    scales,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_MAJOR: tl.constexpr = False,
):
    group = tl.program_id(1).to(tl.int64)
    columns = tl.program_id(0) * BV + tl.arange(0, BV)
    reduction = tl.arange(0, BK)
    source_offsets = (
        ((group * V + columns[:, None]) * K + reduction[None, :])
        if STATE_V_MAJOR
        else ((group * K + reduction[None, :]) * V + columns[:, None])
    )
    target_offsets = (group * V + columns[:, None]) * K + reduction[None, :]
    mask = (columns[:, None] < V) & (reduction[None, :] < K)
    values = tl.load(source + source_offsets, mask, 0).to(tl.float32)
    store_int8_state(
        values, state, scales, target_offsets, group * V + columns, mask, columns < V
    )


def quantize_gdn_state_int8(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize contiguous [slots, value_heads, V, K] state per V channel."""
    logger.debug("GEMS_HYGON QUANTIZE_GDN_STATE_INT8")
    if state.ndim != 4 or not state.is_contiguous():
        raise ValueError("state must be contiguous [slots, value_heads, V, K]")
    if state.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError("state must be floating point")
    if state.device.type != "cuda" or state.requires_grad:
        raise ValueError("state must be a forward-only accelerator tensor")
    slots, heads, columns, reduction = state.shape
    if reduction < 1 or columns < 1:
        raise ValueError("state dimensions V and K must be positive")
    output = torch.empty(state.shape, device=state.device, dtype=torch.int8)
    scales = torch.empty(
        (slots, heads, columns), device=state.device, dtype=torch.float32
    )
    if output.numel():
        with torch_device_fn.device(state.device):
            quantize_prefill_state[(triton.cdiv(columns, 16), slots * heads)](
                state,
                output,
                scales,
                reduction,
                columns,
                triton.next_power_of_2(reduction),
                16,
                STATE_V_MAJOR=True,
                num_warps=4,
            )
    return output, scales


@libentry()
@triton.jit
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
):
    sequence_head = tl.program_id(1).to(tl.int64)
    sequence, head = sequence_head // HV, sequence_head % HV
    query_head = head // (HV // H)
    column = tl.program_id(0) * BV + tl.arange(0, BV)
    reduction = tl.arange(0, BK)
    slot = tl.load(indices + sequence).to(tl.int64)
    output_ptr = output + (sequence * HV + head) * V + column
    # Slot zero is the null block in vLLM packed decode.
    if slot <= 0:
        tl.store(output_ptr, 0.0, column < V)
        return
    group = slot * HV + head
    offsets = (group * V + column[:, None]) * K + reduction[None, :]
    mask = (column[:, None] < V) & (reduction[None, :] < K)
    scale_offsets = group * V + column
    state_values = tl.load(state + offsets, mask, 0).to(tl.float32)
    state_values *= tl.load(scales + scale_offsets, column < V, 0)[:, None]
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
    if NORMALIZE:
        query *= tl.rsqrt(tl.sum(query * query, 0) + 1e-6)
        key *= tl.rsqrt(tl.sum(key * key, 0) + 1e-6)
    query *= SCALE
    gate = tl.load(a + sequence * STRIDE_A + head).to(tl.float32)
    gate += tl.load(dt_bias + head).to(tl.float32)
    softplus = tl.where(gate <= 20.0, tl.log(1.0 + tl.exp(gate)), gate)
    decay = exp(-tl.exp(tl.load(A_log + head).to(tl.float32)) * softplus)
    beta = tl.sigmoid(tl.load(b + sequence * STRIDE_B + head).to(tl.float32))
    state_values *= decay
    residual = (value - tl.sum(state_values * key[None, :], 1)) * beta
    state_values += residual[:, None] * key[None, :]
    result = tl.sum(state_values * query[None, :], 1)
    tl.store(output_ptr, result, column < V)
    store_int8_state(
        state_values, state, scales, offsets, scale_offsets, mask, column < V
    )


def gdn_w8a16(
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
    logger.debug("GEMS_HYGON GDN_W8A16_INT8")
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
    if query.device.type != "cuda" or any(
        tensor.device != query.device for tensor in tensors
    ):
        raise ValueError("all tensors must be on the same accelerator")
    if any(tensor.requires_grad for tensor in tensors):
        raise NotImplementedError("GDN INT8 state is forward-only")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all buffers must be contiguous")
    if not tokens:
        return out, state_int8
    # Autotuning must not replay in-place recurrence on a live state cache.
    block_v, warps = (8 if tokens in (1, 3) else 16), 1
    launch_args = (
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
        scale,
        use_qk_l2norm_in_kernel,
        triton.next_power_of_2(reduction),
        block_v,
    )
    # HCU specializes pointers on alignment and backing-storage range. Cache
    # those facts and compiled metadata only; never retain tensors or streams.
    cache_key = (
        query.device.index,
        query.dtype,
        ssm_state_indices.dtype,
        launch_args[11:],
        tuple(
            (tensor.data_ptr() % 16 == 0, tensor.untyped_storage().size() < 2**31)
            for tensor in tensors
        ),
    )
    with decode_launch_lock:
        compiled = decode_launch_cache.get(cache_key)
        if compiled is not None:
            decode_launch_cache.move_to_end(cache_key)
    grid = (triton.cdiv(columns, block_v), tokens * heads, 1)
    with torch_device_fn.device(query.device):
        if compiled is None:
            compiled = gdn_decode_int8_kernel.jit_function[grid](
                *launch_args,
                num_warps=warps,
                num_stages=1,
            )
            # Concurrent cold calls may compile independently. Publish after
            # launch and keep compilation/device work outside the cache lock.
            with decode_launch_lock:
                decode_launch_cache.setdefault(cache_key, compiled)
                decode_launch_cache.move_to_end(cache_key)
                if len(decode_launch_cache) > DECODE_CACHE_SIZE:
                    decode_launch_cache.popitem(last=False)
        else:
            compiled[grid](*launch_args)
    return out, state_int8


@libentry()
@triton.jit
def normalize_prefill_qk(
    q,
    k,
    q_out,
    k_out,
    ROWS: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
    row_block=None,
):
    if row_block is None:
        row_block = tl.program_id(0)
    rows = row_block * BM + tl.arange(0, BM)
    columns = tl.arange(0, BK)
    offsets = rows[:, None] * K + columns[None, :]
    mask = (rows[:, None] < ROWS) & (columns[None, :] < K)
    query = tl.load(q + offsets, mask, 0).to(tl.float32)
    key = tl.load(k + offsets, mask, 0).to(tl.float32)
    query *= tl.rsqrt(tl.sum(query * query, 1) + 1e-6)[:, None]
    key *= tl.rsqrt(tl.sum(key * key, 1) + 1e-6)[:, None]
    tl.store(q_out + offsets, query, mask)
    tl.store(k_out + offsets, key, mask)


@libentry()
@triton.jit
def dequantize_prefill_state(
    state,
    scales,
    output,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_MAJOR: tl.constexpr = False,
    value_block=None,
    group=None,
):
    if group is None:
        group = tl.program_id(1).to(tl.int64)
    if value_block is None:
        value_block = tl.program_id(0)
    columns = value_block * BV + tl.arange(0, BV)
    reduction = tl.arange(0, K)
    source = (group * V + columns[:, None]) * K + reduction[None, :]
    target = (
        source
        if STATE_V_MAJOR
        else (group * K + reduction[None, :]) * V + columns[:, None]
    )
    values = tl.load(state + source, columns[:, None] < V, 0).to(tl.float32)
    scale = tl.load(scales + group * V + columns, columns < V, 0)
    tl.store(output + target, values * scale[:, None], columns[:, None] < V)


def chunk_gated_delta_rule_w8a16(
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
        chunks = triton.cdiv(tokens, CHUNK_SIZE)
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
        chunks = chunk_indices.shape[0]
    tensors = (q, k, v, g, beta, state_int8, state_scale, *metadata)
    if q.device.type != "cuda" or any(
        tensor.device != q.device or not tensor.is_contiguous() for tensor in tensors
    ):
        raise ValueError("all inputs must be contiguous and on one accelerator")
    if any(tensor.requires_grad for tensor in tensors):
        raise NotImplementedError("INT8-state GDN is forward-only")
    output = torch.empty(v.shape, device=q.device, dtype=q.dtype)
    if tokens == 0:
        return output, state_int8
    launch_scale = reduction**-0.5 if scale is None else float(scale)
    initial = torch.empty(
        (sequences, heads, reduction, columns), device=q.device, dtype=torch.float32
    )
    final = torch.empty_like(initial)
    gate_sum = torch.empty_like(g)
    matrix = torch.empty(
        (batch, tokens, heads, CHUNK_SIZE), device=q.device, dtype=torch.float32
    )
    inverse = flag_gems.zeros(matrix.shape, device=q.device, dtype=q.dtype)
    # WY has one row per value head, including grouped-query attention.
    w = torch.empty((batch, tokens, heads, reduction), device=q.device, dtype=q.dtype)
    u, v_new = torch.empty_like(v), torch.empty_like(v)
    history = torch.empty(
        (chunks, heads, reduction, columns), device=q.device, dtype=q.dtype
    )
    variable = cu_seqlens is not None
    q_norm, k_norm = q, k
    with torch_device_fn.device(q.device):
        if use_qk_l2norm_in_kernel:
            q_norm, k_norm = torch.empty_like(q), torch.empty_like(k)
            normalize_prefill_qk[(triton.cdiv(tokens * query_heads, 16),)](
                q,
                k,
                q_norm,
                k_norm,
                tokens * query_heads,
                reduction,
                reduction,
                16,
                num_warps=4,
            )
        dequantize_prefill_state[(triton.cdiv(columns, 32), sequences * heads)](
            state_int8,
            state_scale,
            initial,
            reduction,
            columns,
            32,
            num_warps=4,
        )
        chunk_prepare[(chunks, heads)](
            g_in=g,
            g_out=gate_sum,
            k=k_norm,
            beta=beta,
            A=matrix,
            A_inv=inverse,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=tokens,
            H=heads,
            Hg=query_heads,
            K=reduction,
            BT=CHUNK_SIZE,
            BK=128,
            IS_VARLEN=variable,
            USE_G=True,
            USE_TMA=False,
            DOT_PRECISION="ieee",
            num_warps=4,
            num_stages=1,
        )
        chunk_wy[(chunks, heads)](
            k=k_norm,
            v=v,
            beta=beta,
            w=w,
            u=u,
            A=inverse,
            g=gate_sum,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=tokens,
            H=heads,
            Hg=query_heads,
            K=reduction,
            V=columns,
            BT=CHUNK_SIZE,
            BK=128,
            BV=64,
            IS_VARLEN=variable,
            num_warps=4,
            num_stages=1,
        )
        chunk_state[(triton.cdiv(columns, 32), sequences * heads)](
            k=k_norm,
            v=u,
            w=w,
            v_new=v_new,
            g=gate_sum,
            gk=None,
            h=history,
            h0=initial,
            ht=final,
            cu_seqlens=cu_seqlens,
            chunk_offsets=chunk_offsets,
            T=tokens,
            H=heads,
            Hg=query_heads,
            K=reduction,
            V=columns,
            BT=CHUNK_SIZE,
            BV=32,
            USE_G=True,
            USE_GK=False,
            USE_INITIAL_STATE=True,
            STORE_FINAL_STATE=True,
            SAVE_NEW_VALUE=True,
            IS_VARLEN=variable,
            num_warps=4,
            num_stages=1,
        )
        chunk_output[(1, chunks, heads)](
            q=q_norm,
            k=k_norm,
            v=v_new,
            h=history,
            g=gate_sum,
            o=output,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            scale=launch_scale,
            T=tokens,
            H=heads,
            Hg=query_heads,
            K=reduction,
            V=columns,
            BT=CHUNK_SIZE,
            BK=128,
            BV=128,
            USE_G=True,
            IS_VARLEN=variable,
            num_warps=4,
            num_stages=1,
        )
        quantize_prefill_state[(triton.cdiv(columns, 16), sequences * heads)](
            final,
            state_int8,
            state_scale,
            reduction,
            columns,
            reduction,
            16,
            num_warps=4,
        )
    return output, state_int8
