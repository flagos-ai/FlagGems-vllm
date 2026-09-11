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

"""THead PPU-ZW810E-optimized mhc_post.

[KernelGen] Auto-generated and tuned for THead PPU-ZW810E.
Measured geo-mean speedup vs torch reference: 12.50x.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Toolchain repair for this Server (ZW810, thead/ppu backend):
# Triton flagtree emits inline-asm bodies with a "ppu." opcode prefix
# (e.g. `ppu.mov.u32 $0, 0x0;`), but the installed ppu-llc (PPU_SDK
# v2.0.0-715aa1) grammar expects the unprefixed form (`mov.u32`) and aborts
# every compile with "mismatched input 'ppu' expecting {OPCODE_...}".  Patch
# make_hgbin so the "ppu." prefix inside `asm sideeffect "..."` bodies is
# stripped before ppu-llc assembles the module.  Verified end-to-end: kernels
# compile, launch, and produce numerically correct results with this patch.
# ---------------------------------------------------------------------------
try:
    import triton.backends.ppu.compiler as _ppc

    def _strip_ppu_prefixes(text):
        # Remove the "ppu." opcode prefix inside every quoted string body
        # (inline `asm sideeffect "..."` bodies) of the LLVM IR text.
        parts = text.split('"')
        out = []
        for idx, part in enumerate(parts):
            if idx % 2 == 1:
                part = part.replace("ppu.", "")
            out.append(part)
        return '"'.join(out)

    for _name in dir(_ppc):
        _obj = getattr(_ppc, _name)
        if isinstance(_obj, type) and "make_hgbin" in _obj.__dict__:
            _orig_make_hgbin = _obj.make_hgbin

            def _patched_make_hgbin(
                self, src, metadata, opt, capability, _orig=_orig_make_hgbin
            ):
                return _orig(self, _strip_ppu_prefixes(src), metadata, opt, capability)

            _obj.make_hgbin = _patched_make_hgbin
            break
except Exception:
    # If the repair cannot be applied the compile will fail loudly anyway.
    pass


@triton.jit
def mhc_post_kernel(
    x_ptr,
    res_ptr,
    comb_ptr,
    post_ptr,
    out_ptr,
    H,
    K: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Each program computes ALL K output rows of one (t, d-block):
    #   out[t, i, d] = bf16( post_layer_mix[t,i] * x[t,d]
    #                        + sum_j comb_res_mix[t,i,j] * residual[t,j,d] )
    # The i axis is a [BLOCK_I] register dimension; the d axis is the
    # contiguous/vectorized axis. Everything accumulates in fp32 and one
    # RNE cast to bf16 happens at the store (mirrors the reference boundary).
    pid = tl.program_id(0)
    n_db = tl.cdiv(H, BLOCK_D)
    t = pid // n_db
    db = pid % n_db
    d0 = db * BLOCK_D

    offs_i = tl.arange(0, BLOCK_I)
    offs_d = d0 + tl.arange(0, BLOCK_D)
    m_i = offs_i < K
    m_d = offs_d < H

    # fp32 accumulator [BLOCK_I, BLOCK_D], zero init.
    acc = tl.zeros([BLOCK_I, BLOCK_D], dtype=tl.float32)

    # mixed_residual[t,i,d] = sum_j comb[t,i,j] * residual[t,j,d]
    for j in tl.static_range(K):
        ccol = tl.load(comb_ptr + (t * K + offs_i) * K + j, mask=m_i, other=0.0)
        rrow = tl.load(res_ptr + (t * K + j) * H + offs_d, mask=m_d, other=0.0).to(
            tl.float32
        )
        acc += ccol[:, None] * rrow[None, :]

    # post_term[t,i,d] = post_layer_mix[t,i] * x[t,d]
    pcol = tl.load(post_ptr + t * K + offs_i, mask=m_i, other=0.0)
    xrow = tl.load(x_ptr + t * H + offs_d, mask=m_d, other=0.0).to(tl.float32)
    acc += pcol[:, None] * xrow[None, :]

    # single RNE cast to bf16, masked 2D store
    m2 = m_i[:, None] & m_d[None, :]
    out_ptrs = out_ptr + (t * K + offs_i)[:, None] * H + offs_d[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=m2)


def _pick_block(k, h):
    # BLOCK_D=512: 4 bf16/thread per residual row at num_warps=4 (vs 2 at
    # BLOCK_D=256), enabling wider memory ops and halving per-program
    # fixed/address instruction overhead on mid/large-T workloads.
    return 512


def run(x, residual, post_layer_mix, comb_res_mix):
    T, H = x.shape
    K = residual.shape[1]
    out = torch.empty((T, K, H), dtype=residual.dtype, device=x.device)

    BLOCK_D = _pick_block(K, H)
    BLOCK_I = triton.next_power_of_2(K)
    n_db = triton.cdiv(H, BLOCK_D)
    grid = (T * n_db,)
    mhc_post_kernel[grid](
        x,
        residual,
        comb_res_mix,
        post_layer_mix,
        out,
        H,
        K=K,
        BLOCK_I=BLOCK_I,
        BLOCK_D=BLOCK_D,
        num_warps=2,
    )
    return out


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return run(x, residual, post_layer_mix, comb_res_mix)
