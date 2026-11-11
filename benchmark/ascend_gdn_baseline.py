# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Standalone latest Ascend GDN baseline adapter for the INT8 benchmark."""

import os
from functools import lru_cache
from types import SimpleNamespace

import torch
import triton
import triton.language as tl


# L2 kernel and launch parameters from vLLM 65711a16, ops/l2norm.py.
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang; vLLM contributors.
@triton.jit
def _ascend_gdn_baseline_l2norm_kernel(
    X, Y, eps, M, N: tl.constexpr, BD: tl.constexpr, MBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * MBLOCK
    row_idx = xoffset + tl.arange(0, MBLOCK)[:, None]
    xmask = row_idx < M
    rindex = tl.arange(0, BD)[None, :]
    cmask = rindex < N
    mask = xmask & cmask
    xs = tl.load(X + (rindex + N * row_idx), mask, other=0.0).to(tl.float32)
    square = tl.broadcast_to(xs * xs, [MBLOCK, BD])
    square_sum = tl.sum(tl.where(xmask, square, 0), 1)[:, None]
    rsqrt = tl.rsqrt(square_sum + eps)
    tl.store(Y + (rindex + N * row_idx), xs * rsqrt, mask)


def _ascend_gdn_baseline_l2norm(x, eps=1e-6):
    shape = x.shape
    x = x.view(-1, shape[-1])
    output = torch.empty_like(x)
    rows, columns = x.shape
    block = min(65536 // x.element_size(), triton.next_power_of_2(columns))
    if columns > block:
        raise NotImplementedError("GDN baseline normalization exceeds 64KB per row")
    _ascend_gdn_baseline_l2norm_kernel[(triton.cdiv(rows, 32),)](
        x, output, eps, rows, columns, block, 32
    )
    return output.view(shape)


@lru_cache(maxsize=1)
def ascend_gdn_native_runtime():
    """Load the standalone latest kernels without the 0.26 model worker."""
    if os.environ.get("VLLM_PLUGINS") != "":
        raise RuntimeError(
            "The standalone GDN baseline requires VLLM_PLUGINS='' before Python starts"
        )
    import torch_npu
    from fla_npu.ops.ascendc import (
        chunk_gated_delta_rule_fwd,
        recurrent_gated_delta_rule,
    )
    from vllm_ascend.ops.gdn_attn_builder import (
        _build_non_spec_chunked_prefill_metadata,
    )
    from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    torch_npu.npu.set_compile_mode(jit_compile=False)
    init_device_properties_triton()
    return (
        chunk_gated_delta_rule_fwd,
        _ascend_gdn_baseline_l2norm,
        fused_gdn_gating_patch,
        _build_non_spec_chunked_prefill_metadata,
        recurrent_gated_delta_rule,
    )


def prepare_ascend_gdn_decode_baseline(reference):
    """Native Ascend BF16 activations / FP32 state, including gating and QK norm."""
    _, normalize_qk, gating, _, recurrent = ascend_gdn_native_runtime()
    mixed, a, b, a_log, bias, scale, state, out, indices, normalize = reference
    a, b = (tensor.to(torch.bfloat16) for tensor in (a, b))
    is_packed = not isinstance(mixed, tuple)
    if is_packed:
        mixed = mixed.to(torch.bfloat16)
        out = torch.empty(out.shape, device=out.device, dtype=torch.bfloat16)
        tokens, heads, columns, reduction = mixed.shape[0], *state.shape[1:]
        query_heads = (mixed.shape[1] - heads * columns) // (2 * reduction)
        query, key, value = mixed.split(
            (query_heads * reduction, query_heads * reduction, heads * columns),
            dim=-1,
        )
        query = query.view(1, tokens, query_heads, reduction)
        key = key.view_as(query)
        value = value.view(tokens, heads, columns)
    else:
        query, key, value = (tensor.to(torch.bfloat16) for tensor in mixed)
        tokens = query.shape[1]
        value = value.squeeze(0)
    lengths = torch.tensor([0] + [1] * tokens, device=query.device, dtype=torch.int32)
    indices = indices.to(torch.int32)

    def _run():
        gate, beta = gating(a_log, a, b, bias)
        if normalize:
            q_norm, k_norm = normalize_qk(query.contiguous()), normalize_qk(
                key.contiguous()
            )
        else:
            q_norm, k_norm = query, key
        result = recurrent(
            q_norm.squeeze(0),
            k_norm.squeeze(0),
            value.contiguous(),
            state,
            beta=beta.squeeze(0),
            g=gate.squeeze(0),
            scale=scale,
            actual_seq_lengths=lengths,
            ssm_state_indices=indices,
        )
        if is_packed:
            out.copy_(result.view_as(out))
            return out, state
        return result.unsqueeze(0), state

    return _run, (state, out) if is_packed else (state,)


def prepare_ascend_gdn_prefill_baseline(reference):
    """Native chunk metadata is constructed outside the timed public call."""
    prefill, normalize_qk, _, build_metadata, _ = ascend_gdn_native_runtime()
    q, k, v, beta = (
        reference[name].to(torch.bfloat16) for name in ("q", "k", "v", "beta")
    )
    state = reference["initial_state"]
    boundaries = torch.tensor(reference["raw_cu_seqlens"], dtype=torch.int32)
    builder = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(linear_num_value_heads=v.shape[2])
            ),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
        )
    )
    plan = build_metadata(builder, boundaries, q.device)

    # Port the A2/A3 DeviceOperator.fla_gdn_prefill call from Ascend 35d8014c.
    # Metadata is built by 0.26; all native kernels and launch options are current.
    def _run():
        q_norm = normalize_qk(q.contiguous()).contiguous()
        k_norm = normalize_qk(k.contiguous()).contiguous()
        initial = state.transpose(-1, -2).contiguous()
        if plan.keep_meta is not None:
            initial = state[plan.keep_meta].transpose(-1, -2).contiguous()
        output, final, *_ = prefill(
            q_norm.transpose(1, 2).contiguous(),
            k_norm.transpose(1, 2).contiguous(),
            v.contiguous().transpose(1, 2).contiguous(),
            reference["g"].to(torch.float32).contiguous(),
            beta.to(v.dtype).contiguous(),
            initial_state=initial,
            output_final_state=True,
            chunk_size=64,
            cu_seqlens=(
                plan.cu_seqlens_host if plan.keep_meta is None else plan.cu_seqlens_kern
            ),
            chunk_indices=plan.chunk_indices_chunk64_host,
            scale=q.shape[-1] ** -0.5,
            layout="BNSD",
            use_exp2=False,
            use_qk_l2norm_in_kernel=False,
            allow_neg_eigval=False,
            disable_recompute=True,
            state_v_first=False,
        )
        if plan.keep_meta is not None:
            full = state.transpose(-1, -2).contiguous()
            full[plan.keep_meta] = final
            final = full
        state.copy_(final.transpose(-1, -2).contiguous())
        return output, state

    return _run, (state,)
