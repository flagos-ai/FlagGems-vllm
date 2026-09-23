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

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.flash_attn_varlen_func_w8a8_fp8 import (
    _flash_attn_varlen_func_w8a8_fp8,
    mha_varlan_fwd,
)
from flaggems_vllm.runtime import torch_device_fn


@triton.jit
def _fp8_pv_dot(
    P,
    V,
    acc,
    v_descale,
    fp8_p_max: tl.constexpr,
    fp8_dtype: tl.constexpr,
    precise_p: tl.constexpr = False,
):
    if fp8_p_max == 448.0:
        # The scaled row sum cancels the factor of 256 during normalization.
        P_scaled = P
        p_descale = 1.0
        p_dtype: tl.constexpr = fp8_dtype
    else:
        P_scaled = P * 256.0
        p_descale = 1.0 / 256.0
        # MUSA lowers same-format E5M2 products to native matrix instructions.
        # Compensate its shorter mantissa with the residual product below.
        p_dtype: tl.constexpr = tl.float8e5
    P_fp8 = P_scaled.to(p_dtype)
    pv = tl.dot(P_fp8, V, out_dtype=tl.float32)
    # Long-KV loops with non-unit block descales also need residual compensation.
    P_residual = ((P_scaled - P_fp8.to(tl.float32)) * 32.0).to(p_dtype)
    pv += tl.dot(P_residual, V, out_dtype=tl.float32) * (1.0 / 32.0)
    return acc + pv * (p_descale * v_descale)


def _normalize_varlen_window(
    max_seqlen_q,
    max_seqlen_k,
    alibi_slopes,
    is_causal,
    window_size_left,
    window_size_right,
):
    if max_seqlen_q == 1 and alibi_slopes is None:
        is_causal = False

    if is_causal:
        window_size_right = 0

    # check disable swa
    if window_size_left >= max_seqlen_k:
        window_size_left = -1
    unlimited_right = max(max_seqlen_q, max_seqlen_k)
    if window_size_right >= unlimited_right:
        window_size_right = -1

    is_local = window_size_left >= 0
    is_local = is_local or (window_size_right >= 0 and not is_causal)
    if is_local:
        # A negative side means unbounded, not an offset of minus one.
        window_size_left = max_seqlen_k if window_size_left < 0 else window_size_left
        window_size_right = (
            unlimited_right if window_size_right < 0 else window_size_right
        )
    return is_causal, is_local, window_size_left, window_size_right


def _get_varlen_fwd_config(args, head_size, use_varlen_split_d, is_paged, *_):
    cfg = runtime.get_heuristic_config("mha_block_32")
    cfg_params = {
        "BLOCK_M": cfg["BLOCK_M"](args),
        "BLOCK_N": cfg["BLOCK_N"](args),
        "BLOCK_K": max(32, triton.next_power_of_2(head_size)),
        "BLOCK_D": (
            64 if use_varlen_split_d else max(32, triton.next_power_of_2(head_size))
        ),
        "SPLIT_D": use_varlen_split_d,
        "num_warps": cfg["num_warps"](args),
        "num_stages": 1 if not is_paged else cfg["num_stages"](args),
    }
    if head_size not in (64, 128) and not is_paged:
        cfg_params.update(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=1)
    return cfg_params


def _mha_varlan_fwd(*args, **kwargs):
    return mha_varlan_fwd(
        *args,
        **kwargs,
        window_normalizer=_normalize_varlen_window,
        get_config=_get_varlen_fwd_config,
        pv_dot=_fp8_pv_dot,
        allow_group_swap=False,
    )


def flash_attn_varlen_func_w8a8_fp8(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,  # only used for non-paged prefill
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    # Compatibility arguments from the shared FlashAttention API.
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    s_aux=None,
    num_splits: int = 0,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k=None,
    fa_version: int = 2,
):
    """Run block-scaled FP8 FlashAttention-2 on MUSA, including MQA and GQA.

    Q/K/V use E4M3FN or E5M2 with descales per logical 128-token block.
    The output uses ``out.dtype`` when provided, or BF16 otherwise.
    """
    if torch_device_fn.get_device_capability()[0] < 3:
        raise NotImplementedError(
            "W8A8 FP8 attention requires MUSA capability 3 or newer"
        )
    return _flash_attn_varlen_func_w8a8_fp8(
        q=q,
        k=k,
        v=v,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        cu_seqlens_k=cu_seqlens_k,
        seqused_k=seqused_k,
        q_v=q_v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        block_table=block_table,
        return_softmax_lse=return_softmax_lse,
        out=out,
        scheduler_metadata=scheduler_metadata,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        s_aux=s_aux,
        num_splits=num_splits,
        cp_world_size=cp_world_size,
        cp_rank=cp_rank,
        cp_tot_seqused_k=cp_tot_seqused_k,
        fa_version=fa_version,
        fp8_dtypes=(torch.float8_e4m3fn, torch.float8_e5m2),
        allow_gqa=True,
        use_dense=False,
        varlen_fwd=_mha_varlan_fwd,
    )
