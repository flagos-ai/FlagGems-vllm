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

"""Huawei Ascend910B4-optimized mhc_post.

[KernelGen] Auto-generated and tuned for Huawei Ascend910B4.
Measured geo-mean speedup vs torch reference: 3.68x.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver


@triton.jit
def _mhc_post_kernel(
    x_ptr,
    residual_ptr,
    comb_ptr,
    post_ptr,
    out_ptr,
    T,
    D,
    ND,
    H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    """Fused mHC post block.

    out[t, i, d] = bf16( sum_j comb_res_mix[t, i, j] * residual[t, j, d]
                         + post_layer_mix[t, i] * x[t, d] )

    Pure-vector persistent kernel: grid = min(num_vectorcore, T*ND) programs
    (decremented by one when that count is a multiple of ND and programs
    iterate >= 2 units, so tail chunks are not pinned to a core subset), each
    striding over (t, d-chunk) units by tl.num_programs(0). Every unit
    computes all H output rows for one contiguous BLOCK_D column strip of one
    token, so each residual element is read exactly once from DRAM.

    BLOCK_D is the widest UB-affordable tile (<= 2048). When BLOCK_D does not
    divide D only the LAST chunk of each token (db == ND-1) is partial; that
    tail is loaded/stored masked (other=0 pads the out-of-range lanes, which
    never reach memory because the store is predicated), while every full chunk
    stays mask-free through a per-unit scalar branch.
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    total_units = T * ND

    cols = tl.arange(0, BLOCK_D)
    irows = tl.arange(0, H)
    hd = H * D

    # Grid-strided persistent traversal, striding by the actual launched program
    # count. The host picks a grid size coprime to ND whenever programs iterate
    # >= 2 units, so db = unit % ND cycles through all chunk residues per
    # program and masked-tail chunks (db == ND-1) are spread across every core
    # instead of being pinned to the pid % ND == ND-1 subset.
    for unit in range(pid, total_units, num_progs):
        t = unit // ND
        db = unit - t * ND
        offs = db * BLOCK_D + cols

        # x segment for this token / d-chunk: [BLOCK_D] bf16 -> fp32
        x_off = t * D + offs
        if EVEN_D:
            xseg = tl.load(x_ptr + x_off)
        else:
            col_mask = offs < D  # tail mask; full chunks are all-true
            if db == ND - 1:  # runtime scalar: only the per-token tail chunk
                xseg = tl.load(x_ptr + x_off, mask=col_mask, other=0.0)
            else:
                xseg = tl.load(x_ptr + x_off)

        # acc[i, d] fp32 accumulator; j-reduction ascending (mirrors einsum)
        acc = tl.zeros((H, BLOCK_D), dtype=tl.float32)
        comb_base = t * H * H
        res_row_base = t * hd
        for j in tl.static_range(H):
            # comb column j for all i: comb[t, i, j], i-stride H  (fp32 native)
            colj = tl.load(comb_ptr + comb_base + j + irows * H)
            # residual row j: [BLOCK_D] bf16, contiguous
            if EVEN_D:
                rj = tl.load(residual_ptr + res_row_base + j * D + offs)
            else:
                if db == ND - 1:
                    rj = tl.load(
                        residual_ptr + res_row_base + j * D + offs,
                        mask=col_mask,
                        other=0.0,
                    )
                else:
                    rj = tl.load(residual_ptr + res_row_base + j * D + offs)
            acc += colj[:, None] * rj.to(tl.float32)[None, :]

        # post term added after the mix sum, in fp32 (reference ordering)
        pl = tl.load(post_ptr + t * H + irows)  # fp32 native
        acc += pl[:, None] * xseg.to(tl.float32)[None, :]

        out_val = acc.to(tl.bfloat16)
        row_ptr = out_ptr + (t * H + irows)[:, None] * D + offs[None, :]
        if EVEN_D:
            tl.store(row_ptr, out_val)
        else:
            if db == ND - 1:
                tl.store(row_ptr, out_val, mask=(offs < D)[None, :])
            else:
                tl.store(row_ptr, out_val)


def _pick_block_d(D):
    # Widest UB-affordable tile (2048 cap: acc[H=4,2048] fp32 = 32KB leaves
    # multi-buffer headroom). Round-4 profile: per-core MTE2 bandwidth scales
    # with the contiguous extent of each load request, so prefer the maximum
    # tile even when it does not divide D (tail chunk is masked). For D < 2048
    # fall back to the largest power-of-two divisor.
    if D >= 2048:
        return 2048
    b = D
    while b >= 128:
        if D % b == 0:
            return b
        b //= 2
    return 128


def run(x, residual, post_layer_mix, comb_res_mix):
    T, D = x.shape
    H = residual.shape[1]
    if not x.is_contiguous():
        x = x.contiguous()
    if not residual.is_contiguous():
        residual = residual.contiguous()
    if not comb_res_mix.is_contiguous():
        comb_res_mix = comb_res_mix.contiguous()
    if not post_layer_mix.is_contiguous():
        post_layer_mix = post_layer_mix.contiguous()

    out = torch.empty((T, H, D), dtype=residual.dtype, device=x.device)

    BLOCK_D = _pick_block_d(D)
    EVEN_D = (D % BLOCK_D) == 0
    ND = triton.cdiv(D, BLOCK_D)
    total_units = T * ND

    props = driver.active.utils.get_device_properties(torch.npu.current_device())
    nvc = int(props["num_vectorcore"])
    G = min(total_units, nvc)
    # Tail-balance the grid-strided loop: with unit = pid + G*k, db = unit % ND
    # is constant per program whenever G is a multiple of ND, pinning all
    # masked-tail chunks (db == ND-1) onto the pid % ND == ND-1 cores. When the
    # tile does not divide D (masked tails exist), every program iterates >= 2
    # units, and G is a multiple of ND, decrement G by one so the stride is
    # coprime to ND and each program cycles through all chunk residues. This
    # depends only on divisibility of the launched grid by ND, never on an
    # absolute core count; EVEN_D tiles have no tails and keep full G.
    if G > 1 and total_units > G and not EVEN_D and G % ND == 0:
        G -= 1
    grid = (G,)

    _mhc_post_kernel[grid](
        x,
        residual,
        comb_res_mix,
        post_layer_mix,
        out,
        T,
        D,
        ND,
        H=H,
        BLOCK_D=BLOCK_D,
        EVEN_D=EVEN_D,
    )
    return out


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return run(x, residual, post_layer_mix, comb_res_mix)
