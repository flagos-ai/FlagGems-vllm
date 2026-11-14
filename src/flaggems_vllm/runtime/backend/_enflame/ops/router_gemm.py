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

"""bf16 x bf16 -> fp32 router GEMM, GCU (enflame) adaptation.

Differences vs the generic implementation:

* **No split-K / ``tl.atomic_add`` path.** Atomics are forbidden on GCU
  (knowledge/coding_rules.md) and the baseline split-K kernel produced
  wrong values (NaN, 26% mismatch) under the old toolchain. Every shape
  now runs a single ``tl.dot`` tiled kernel (or a 1-D row kernel for
  tiny M/N), each output element written exactly once by ``tl.store``
  -- which also removes the ``out.zero_()`` memset launch.
* **MLIR dialect shim.** The installed FlagTree wheel emits
  ``memref_ext.*`` / ``gcu.matmul`` ops with a trailing zero operand
  segment that the external ``gcu-compiler-opt`` rejects, so any 2-D
  tile kernel (including ``tl.dot``) fails first-time compilation. The
  module-level shim below rewrites the generated MLIR text to the
  layout the external tool accepts (precedent: FLA/
  parallel_nsa_compression.py, see compiler_issues/vllm_moe_sum.md).
"""

import logging
import re

import torch
import triton
import triton.language as tl

from flaggems_vllm.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# ===========================================================================
# GCU compiler-toolchain dialect compatibility shim
# ===========================================================================
# The installed FlagTree wheel emits MLIR where several `memref_ext.*` /
# `gcu.matmul` ops carry a trailing zero operand segment, while the external
# `gcu-compiler-opt` tool shipped in /opt/triton_gcu (triton-gcu 1.0.20261020)
# still registers the older operand layouts. The mismatch makes every kernel
# with 2-D tensor accesses (block pointers, masked 2-D loads, tl.dot tiles)
# fail to compile with "operandSegmentSizes ... size mismatch". The shim below
# rewrites the generated MLIR text to the layout the external tool accepts:
#   * `memref_ext.memset_start` must carry no properties at all;
#   * other `memref_ext.*` / `gcu.matmul` ops drop their trailing zero
#     segments (pure metadata, no operand rebinding);
#   * `memref.reinterpret_cast` / `memref.subview` need explicit segment
#     sizes derived from their INT64_MIN sentinel counts when missing.
# It is idempotent and leaves already-compatible IR (e.g. pure 1-D kernels)
# untouched.

_GCU_SENTINEL = "-9223372036854775808"


def _fix_gcu_attr(attr_match, line, op_name):
    inner = attr_match.group(1)
    seg_match = re.search(r"operandSegmentSizes = array<i32: ([^>]*)>", inner)
    if op_name == "memref_ext.memset_start":
        return ""
    if op_name.startswith("memref_ext.") or op_name == "gcu.matmul":
        if seg_match:
            segs = [s.strip() for s in seg_match.group(1).split(",")]
            while len(segs) > 1 and segs[-1] == "0":
                segs.pop()
            new_seg = "array<i32: " + ", ".join(segs) + ">"
            rest = re.sub(r"operandSegmentSizes = array<i32:[^>]*>", "", inner)
            rest = rest.strip().strip(",").strip()
            if rest:
                return f" <{{operandSegmentSizes = {new_seg}, {rest}}}>"
            return f" <{{operandSegmentSizes = {new_seg}}}>"
        return attr_match.group(0)
    if op_name in ("memref.reinterpret_cast", "memref.subview") and not seg_match:
        opers = re.search(r"\(([^()]*)\)\s*<\{", line)
        n_ops = len([x for x in opers.group(1).split(",") if x.strip()]) if opers else 1
        offs = re.search(r"static_offsets = array<i64: ([^>]*)>", line)
        sizes = re.search(r"static_sizes = array<i64: ([^>]*)>", line)
        strides = re.search(r"static_strides = array<i64: ([^>]*)>", line)
        n_off = offs.group(1).count(_GCU_SENTINEL) if offs else 0
        n_size = sizes.group(1).count(_GCU_SENTINEL) if sizes else 0
        n_stride = strides.group(1).count(_GCU_SENTINEL) if strides else 0
        if 1 + n_off + n_size + n_stride == n_ops:
            seg = f"array<i32: 1, {n_off}, {n_size}, {n_stride}"
            rest = inner.strip()
            if rest:
                return f" <{{operandSegmentSizes = {seg}, {rest}}}>"
            return f" <{{operandSegmentSizes = {seg}}}>"
    return attr_match.group(0)


def _fix_gcu_dialect_line(line):
    op_match = re.search(r'"([a-z0-9_.]+)"\(', line)
    if not op_match:
        return line
    op_name = op_match.group(1)
    if not (
        op_name.startswith("memref_ext.")
        or op_name.startswith("gcu.")
        or op_name in ("memref.reinterpret_cast", "memref.subview")
    ):
        return line
    return re.sub(
        r" <\{([^}]*)\}>",
        lambda m: _fix_gcu_attr(m, line, op_name),
        line,
        count=1,
    )


def _patch_gcu_compiler_opt():
    import triton.backends.enflame.toolkit as _toolkit

    if getattr(_toolkit.gcu_compiler_opt, "_flaggems_dialect_shim", False):
        return
    _orig = _toolkit.gcu_compiler_opt

    def _shim(content, *args):
        if isinstance(content, str) and "operandSegmentSizes" in content:
            content = "\n".join(_fix_gcu_dialect_line(l) for l in content.split("\n"))
        return _orig(content, *args)

    _shim._flaggems_dialect_shim = True
    _toolkit.gcu_compiler_opt = _shim


# _patch_gcu_compiler_opt()  # DISABLED 2026-11-09: 私有 GCUIR shim 实测引入回归,详见 commit b333be6


@triton.jit
def _router_gemm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_idxs = k_start + offs_k
        a = tl.load(
            A + offs_m[:, None] * stride_am + k_idxs[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B + k_idxs[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k_idxs[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    tl.store(
        C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _router_gemm_rows_kernel(
    A,
    W,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wn,
    stride_wk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 1-D row-blocked form for tiny M/N: one program per (expert, row block),
    # pure 1-D affine loads/stores, [BLOCK_K] vector accumulator with a final
    # reduction and a [BLOCK_M] where-accumulator across rows (FLA
    # row-streaming / fp32_router_gemm precedent).
    expert = tl.program_id(0)
    row_base = tl.program_id(1) * BLOCK_M
    rows = row_base + tl.arange(0, BLOCK_M)
    offsets_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M,), tl.float32)
    for start in range(0, K, BLOCK_K):
        cols = start + offsets_k
        weight = tl.load(
            W + expert * stride_wn + cols * stride_wk,
            mask=cols < K,
            other=0.0,
        ).to(tl.float32)
        for r in tl.static_range(BLOCK_M):
            x = tl.load(
                A + (row_base + r) * stride_am + cols * stride_ak,
                mask=(cols < K) & (row_base + r < M),
                other=0.0,
            ).to(tl.float32)
            partial = tl.sum(x * weight)
            accumulator = tl.where(
                tl.arange(0, BLOCK_M) == r, accumulator + partial, accumulator
            )

    tl.store(C + rows * stride_cm + expert * stride_cn, accumulator, mask=rows < M)


def _tiled_router_gemm(x, weight_t, out, M, N, K):
    logger.debug(
        "FlagGems-vllm router_gemm, tiled, shape: [%s, %s, %s](M, N, K)",
        M,
        N,
        K,
    )
    # Dispatch anchored on bench shapes (GCU400 / Libra, 48 SPC, DSM 448KB):
    # * per-K-iteration cost is dominated by the tl.dot SIP call barrier
    #   (~2us/iter), so BLOCK_K is pushed to the largest 2048/1024/512 that
    #   divides K exactly (a masked partial-K tile at BLOCK_K=2048 produced
    #   NaN on [BM=32,BN=32] tiles - partial-tile mma lane corruption, same
    #   family as chunk_cumsum - so non-divisible K falls back to 256);
    # * num_warps=2 is the sweet spot for these dot tiles (nw4/nw8 ~+25-30%);
    # * grid is kept within one 48-program wave: BLOCK_N doubles to 128 when
    #   the linear grid would exceed 48 programs (M=4096,N=128 anchor).
    if M <= 64:
        block_m = 16
    elif M <= 256:
        block_m = 32
    elif M <= 1024:
        block_m = 64
    else:
        block_m = 128
    if N <= 16:
        block_n = 16
    elif N <= 64:
        block_n = 32
    else:
        block_n = 64

    grid_n = triton.cdiv(N, block_n)
    if triton.cdiv(M, block_m) * grid_n > 48 and block_n < 128:
        block_n *= 2
        grid_n = triton.cdiv(N, block_n)
    # Keep grid.x within the GCU 65535 limit for extreme token counts.
    while triton.cdiv(M, block_m) * grid_n > 65535:
        block_m *= 2

    for block_k in (2048, 1024, 512):
        if K % block_k == 0:
            break
    else:
        block_k = 256
    # DSM guard: (A tile + B tile) bf16 + fp32 accumulator must stay below
    # the 448KB dual-program envelope.
    while (
        block_m * block_k * 2 + block_k * block_n * 2 + block_m * block_n * 4
    ) > 448 * 1024 and block_k > 64:
        block_k //= 2

    grid = (triton.cdiv(M, block_m) * grid_n,)
    with torch_device_fn.device(x.device):
        _router_gemm_kernel[grid](
            x,
            weight_t,
            out,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            weight_t.stride(0),
            weight_t.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            num_warps=2,
            num_stages=4,
        )
    return out


def _rows_router_gemm(x, weight, out, M, N, K):
    logger.debug(
        "FlagGems-vllm router_gemm, rows, shape: [%s, %s, %s](M, N, K)",
        M,
        N,
        K,
    )
    block_m = 4 if M <= 4 else 8
    with torch_device_fn.device(x.device):
        _router_gemm_rows_kernel[(N, triton.cdiv(M, block_m))](
            x,
            weight,
            out,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            weight.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M=block_m,
            BLOCK_K=2048,
            num_warps=4,
            num_stages=1,
        )
    return out


def router_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """bf16 x bf16 -> fp32 GEMM for MoE router gate.

    Args:
        x: Router input with shape (M, K).
        weight: Router weight with shape (N, K).
    """
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("router_gemm expects 2D x and weight tensors")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("router_gemm expects x.shape[1] == weight.shape[1]")
    if x.device != weight.device:
        raise ValueError("router_gemm expects x and weight to be on the same device")

    if x.stride(0) > 1 and x.stride(1) > 1:
        x = x.contiguous()

    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    if M == 0 or N == 0:
        return out

    if M <= 16 and N <= 16:
        return _rows_router_gemm(x, weight, out, M, N, K)
    return _tiled_router_gemm(x, weight.t(), out, M, N, K)
