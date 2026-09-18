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

from numbers import Integral

import torch

from .attention_impl.tensor_utils import ensure_last_dim_contiguous
from .flash_api import mha_varlan_fwd


def _validate_num_splits(num_splits):
    if isinstance(num_splits, bool) or not isinstance(num_splits, int):
        raise TypeError(f"num_splits must be an integer in [0, 32], got {num_splits!r}")
    if num_splits < 0 or num_splits > 32:
        raise ValueError(f"num_splits must be in [0, 32], got {num_splits}")
    return num_splits


def flash_attn_varlen_func(
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
    # Dummy FA3 arguments
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
    """MetaX inference varlen attention, with optional FP32 [heads, total_q] LSE.

    Matches the public vLLM-compatible FA2 signature. Nonzero dropout and FA3
    extensions are unsupported. The forward-only deterministic flag requires
    no separate path. No original FlagGems C extension is used.
    """
    if fa_version != 2:
        raise NotImplementedError("MetaX varlen attention supports fa_version=2")
    if dropout_p != 0:
        raise NotImplementedError("MetaX varlen attention supports dropout_p=0")
    if return_attn_probs:
        raise NotImplementedError(
            "MetaX varlen attention does not return attention probabilities"
        )
    if (
        any(
            x is not None
            for x in (
                q_v,
                scheduler_metadata,
                q_descale,
                k_descale,
                v_descale,
                s_aux,
                cp_tot_seqused_k,
            )
        )
        or cp_world_size != 1
        or cp_rank != 0
    ):
        raise NotImplementedError(
            "MetaX varlen attention does not support FA3 or context-parallel extensions"
        )
    num_splits = _validate_num_splits(num_splits)
    if not isinstance(max_seqlen_q, Integral) or not isinstance(max_seqlen_k, Integral):
        raise TypeError("max_seqlen_q and max_seqlen_k must be host integers")
    max_seqlen_q, max_seqlen_k = int(max_seqlen_q), int(max_seqlen_k)
    if max_seqlen_q < 0 or max_seqlen_k < 0:
        raise ValueError("maximum sequence lengths must be nonnegative")
    assert cu_seqlens_k is not None or seqused_k is not None
    assert cu_seqlens_k is None or seqused_k is None
    assert block_table is None or seqused_k is not None
    assert q.device == k.device == v.device
    assert cu_seqlens_q.device == q.device
    if cu_seqlens_k is not None:
        assert cu_seqlens_k.device == q.device
    if seqused_k is not None:
        assert seqused_k.device == q.device and seqused_k.dtype == torch.int32
    if block_table is not None:
        assert block_table.device == q.device and block_table.dtype == torch.int32
        assert block_table.stride(-1) == 1
    if out is not None:
        assert out.device == q.device
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if window_size is None:
        real_window_size = (-1, -1)
    else:
        assert len(window_size) == 2
        real_window_size = tuple(window_size)
    q, k, v = [ensure_last_dim_contiguous(x) for x in (q, k, v)]
    # The seqused_k path never reads this placeholder.
    if cu_seqlens_k is None:
        cu_seqlens_k = torch.empty_like(cu_seqlens_q)
    output, lse = mha_varlan_fwd(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        None,
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        0.0,
        softmax_scale,
        False,
        causal,
        real_window_size[0],
        real_window_size[1],
        softcap,
        False,
        None,
        num_splits,
    )
    return (output, lse) if return_softmax_lse else output


__all__ = ["flash_attn_varlen_func"]
