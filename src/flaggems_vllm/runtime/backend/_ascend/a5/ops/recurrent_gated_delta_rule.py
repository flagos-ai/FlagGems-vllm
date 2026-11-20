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
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn


@triton.jit
def _gdn_one_token_fold_kernel(
    Q,
    K,
    V,
    STATE,
    BETA,
    G,
    SSM,
    CU,
    OUT,
    scale,
    HK_RATIO: tl.constexpr,
    NV: tl.constexpr,
    TILES: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SQ1: tl.constexpr,
    SQ2: tl.constexpr,
    SK1: tl.constexpr,
    SK2: tl.constexpr,
    SV1: tl.constexpr,
    SV2: tl.constexpr,
    POOL: tl.constexpr,
    ST0: tl.constexpr,
    ST1: tl.constexpr,
    ST2: tl.constexpr,
    ST3: tl.constexpr,
    SO1: tl.constexpr,
    SO2: tl.constexpr,
):
    """Same body as _gdn_one_token_kernel with the scalar decay folded into the
    two premise vectors: no materialised decayed tile, so S is swept three times
    instead of four. The reduction reassociation (d applied to k/q instead of to
    S) shifts the mutated state by about one fp32 ULP."""
    pid = tl.program_id(0)
    vt = pid % TILES
    rest = pid // TILES
    hv = rest % NV
    hk = hv // HK_RATIO

    sidx = tl.load(SSM).to(tl.int64)
    # A padding sequence must not overwrite the selected state slot.
    if sidx < 0 or sidx >= POOL or tl.load(CU) != 0 or tl.load(CU + 1) != 1:
        col = vt * BV + tl.arange(0, BV)
        tl.store(OUT + hv * SO1 + col * SO2, tl.full((BV,), 0, tl.float32))
        return
    ov = vt * BV + tl.arange(0, BV)
    ok = tl.arange(0, BK)

    sp = STATE + sidx * ST0 + hv * ST1 + ov[:, None] * ST2 + ok[None, :] * ST3
    S = tl.load(sp).to(tl.float32)

    qv = tl.load(Q + hk * SQ1 + ok * SQ2).to(tl.float32) * scale
    kv = tl.load(K + hk * SK1 + ok * SK2).to(tl.float32)
    vv = tl.load(V + hv * SV1 + ov * SV2).to(tl.float32)
    bb = tl.load(BETA + hv).to(tl.float32)
    gg = tl.load(G + hv)

    d = tl.exp(gg)
    u = tl.sum(S * (kv * d)[None, :], 1)
    delta = bb * (vv - u)
    # reassociated output off the update chain: q.S_new == q.S + (q.k)*delta,
    # so the q reduction reads the decayed pre-update tile while the hoisted
    # 32 KiB state DMA below runs
    w = tl.sum(S * (qv * d)[None, :], 1)
    S = S * d + delta[:, None] * kv[None, :]
    tl.store(sp, S.to(STATE.dtype.element_ty))
    o = w + tl.sum(qv * kv, 0) * delta

    tl.store(OUT + hv * SO1 + ov * SO2, o.to(OUT.dtype.element_ty))


def recurrent_gated_delta_rule(
    query,
    key,
    value,
    state,
    *,
    beta,
    g,
    actual_seq_lengths,
    ssm_state_indices,
    scale=None,
):
    """FLA one-token decode with prefix lengths [0,1] and one valid pool index.

    Update state in place using its actual strides, including padded state pools.
    The output is the BF16 value projection [1, value_heads, 128].
    """
    q, k, v = query, key, value
    if (
        q.device.type != "npu"
        or q.shape != (1, 8, 128)
        or k.shape != q.shape
        or v.ndim != 3
        or v.shape[0] != 1
        or v.shape[2] != 128
        or v.shape[1] not in (8, 16, 32)
    ):
        raise ValueError(
            "expected a single query token, eight query heads and 128-wide Q/K/V"
        )
    hv = v.shape[1]
    if (
        state.ndim != 4
        or state.shape[1:] != (hv, 128, 128)
        or state.shape[0] < 1
        or state.dtype != torch.float32
        or state.stride(3) != 1
        or state.stride(2) < 128
        or state.stride(1) < 128 * state.stride(2)
        or state.stride(0) < hv * state.stride(1)
    ):
        raise ValueError("expected a nonoverlapping channel-last FP32 state pool")
    if (
        actual_seq_lengths.shape != (2,)
        or ssm_state_indices.shape != (1,)
        or beta.shape != (1, hv)
        or g.shape != (1, hv)
    ):
        raise ValueError("expected one active sequence and one gate per value head")
    if (
        any(t.dtype != torch.bfloat16 for t in (q, k, v, beta))
        or g.dtype != torch.float32
        or actual_seq_lengths.dtype != torch.int32
        or ssm_state_indices.dtype != torch.int32
    ):
        raise ValueError("expected BF16 q/k/v/beta, FP32 g and int32 metadata")
    if any(
        t.device != q.device
        for t in (k, v, state, beta, g, actual_seq_lengths, ssm_state_indices)
    ) or not all(
        t.is_contiguous() for t in (beta, g, actual_seq_lengths, ssm_state_indices)
    ):
        raise ValueError("expected one device and contiguous gates")
    if scale is None:
        scale = 128**-0.5
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    out = torch.empty_like(v)
    bv = 64 if hv >= 16 else 32
    tiles = 128 // bv
    with torch_device_fn.device(q.device):
        _gdn_one_token_fold_kernel[(hv * tiles,)](
            q,
            k,
            v,
            state,
            beta,
            g,
            ssm_state_indices,
            actual_seq_lengths,
            out,
            scale,
            HK_RATIO=hv // 8,
            NV=hv,
            TILES=tiles,
            BK=128,
            BV=bv,
            SQ1=q.stride(1),
            SQ2=q.stride(2),
            SK1=k.stride(1),
            SK2=k.stride(2),
            SV1=v.stride(1),
            SV2=v.stride(2),
            POOL=state.shape[0],
            ST0=state.stride(0),
            ST1=state.stride(1),
            ST2=state.stride(2),
            ST3=state.stride(3),
            SO1=out.stride(1),
            SO2=out.stride(2),
        )
    return out
