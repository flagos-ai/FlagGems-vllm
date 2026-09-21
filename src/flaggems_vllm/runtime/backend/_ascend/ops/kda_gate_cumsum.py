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

"""Fused Triton replacement for the AscendC ``kda_gate_cumsum`` operator.

The AscendC kernel parallelises varlen work as only ``seqNum * hv`` blocks,
so each block serially walks a whole sequence token by token on a handful of
AIV cores (varlen prefill uses <2% of the vector cores).  This Triton version
keeps the exact AscendC semantics while launching one program per
(chunk, head) tile:

    gate[t, h, d] = lower_bound * sigmoid((g[t, h, d] + dt_bias[h, d]) * exp(A_log[h]))
    gk[t, h, d]  = RCP_LN2 * sum_{t' in [chunk_start, t]} gate[t', h, d]

where chunk boundaries are aligned to each sequence start and the cumsum
resets every chunk (see
csrc/attention/kda_gate_cumsum/op_kernel/kda_gate_cumsum.cpp, ProcessTask /
ProcessChunk).  Only the safe-gate + in-kernel-gate path used by Kimi/GLM is
implemented; everything else falls back to the AscendC operator.
"""

import torch
import triton
import triton.language as tl

RCP_LN2 = tl.constexpr(1.4426950408889634)


@triton.jit(do_not_specialize=["lower_bound", "H"])
def kda_gate_cumsum_kernel(
    g,
    A_log,
    dt_bias,
    cu_seqlens,
    chunk_indices,
    gk,
    lower_bound,
    H,
    D: tl.constexpr,
    BT: tl.constexpr,
    USE_CUMSUM_OP: tl.constexpr,
):
    i_c, i_h = tl.program_id(0), tl.program_id(1)
    i_s = tl.load(chunk_indices + i_c * 2).to(tl.int32)
    i_t = tl.load(chunk_indices + i_c * 2 + 1).to(tl.int32)
    bos = tl.load(cu_seqlens + i_s).to(tl.int32)
    eos = tl.load(cu_seqlens + i_s + 1).to(tl.int32)
    start = bos + i_t * BT
    end = tl.minimum(start + BT, eos)

    b_a = tl.load(A_log + i_h).to(tl.float32)
    o_d = tl.arange(0, D)
    b_bias = tl.load(dt_bias + i_h * D + o_d).to(tl.float32)

    # g/gk are [1, T, H, D]; offset of (t, h, d) is (t * H + h) * D + d.
    o_t = tl.arange(0, BT)
    m_t = start + o_t < end
    offs = (start + o_t)[:, None] * (H * D) + i_h * D + o_d[None, :]

    b_g = tl.load(g + offs, mask=m_t[:, None], other=0.0).to(tl.float32)
    b_g = (b_g + b_bias[None, :]) * tl.exp(b_a)
    b_gate = lower_bound * tl.sigmoid(b_g)

    if USE_CUMSUM_OP:
        b_o = tl.cumsum(b_gate, axis=0)
    else:
        # Mask-dot cumsum: same result, friendlier to vector engines when the
        # associative scan lowers poorly.
        o_i = tl.arange(0, BT)
        m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0).to(tl.float32)
        b_o = tl.dot(m_s, b_gate, allow_tf32=False)
    b_o = b_o * RCP_LN2

    tl.store(gk + offs, b_o, mask=m_t[:, None])


def kda_gate_cumsum_triton(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    lower_bound: float,
    chunk_size: int = 64,
    num_warps: int = 4,
    use_cumsum_op: bool = True,
) -> torch.Tensor:
    """Compute the fused safe-gate + chunk-local cumsum in one Triton kernel.

    Args:
        g: ``[1, T, H, D]`` contiguous BSND gate input (bf16/fp32/fp16).
        A_log: ``[H]`` fp32 log-decay per head.
        dt_bias: ``[H * D]`` fp32 gate bias.
        cu_seqlens: ``[num_seqs + 1]`` int64 device tensor.
        chunk_indices: ``[NT, 2]`` int64 device tensor from
            :func:`prepare_chunk_indices` (pairs of (seq_idx, chunk_idx),
            chunk boundaries aligned to each sequence start).
        lower_bound: negative gate bound (e.g. -5.0).

    Returns:
        ``[1, T, H, D]`` fp32 tensor with the chunk-local cumsum of the gate,
        pre-multiplied by ``1 / ln(2)`` for ``chunk_kda_fwd``.
    """
    assert g.dim() == 4 and g.shape[0] == 1, "g must be [1, T, H, D]"
    _, T, H, D = g.shape
    assert D <= 256, "KDA gate head dim must be <= 256"
    gk = torch.empty_like(g, dtype=torch.float32)

    NT = chunk_indices.shape[0]
    if NT == 0:
        return gk

    kda_gate_cumsum_kernel[(NT, H)](
        g,
        A_log,
        dt_bias,
        cu_seqlens,
        chunk_indices,
        gk,
        lower_bound,
        H,
        D=D,
        BT=chunk_size,
        USE_CUMSUM_OP=use_cumsum_op,
        num_warps=num_warps,
        num_stages=2,
    )
    return gk
