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

indexer_fwd_configs = [
    triton.Config({"num_stages": 2, "num_warps": 4}),
    triton.Config({"num_stages": 4, "num_warps": 8}),
]


@triton.autotune(  # Decorate the kernel
    configs=indexer_fwd_configs,
    key=["Q", "K", "H", "D"],
)
@triton.jit
def triton_lighting_indexer_k_tiled(
    q_index,
    k_index,
    cu_bg_seqlens,
    cu_ed_seqlens,
    weights,
    logits,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kd,
    stride_wh,
    stride_lm,
    stride_ln,
    Q: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    TK: tl.constexpr,
    D: tl.constexpr,
    CU: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
):
    i_sh, i_k = tl.program_id(0), tl.program_id(1)

    offs_cu = tl.arange(0, BQ) + i_sh * BQ
    mask_cu = offs_cu < CU
    bos_vec, eos_vec = tl.load(
        cu_bg_seqlens + offs_cu, mask_cu, 1000000000
    ) + i_k * TK, tl.load(
        cu_ed_seqlens + offs_cu, mask_cu, -1000000000
    )  # [BQ]
    eos_vec = tl.minimum(eos_vec, bos_vec + (i_k + 1) * TK)
    bos, eos = max(bos_vec.min(0), 0), min(eos_vec.max(0), K)
    CK = eos - bos
    if CK > 0:
        q_base = q_index
        k_base = k_index + bos * stride_kn
        w_base = weights
        o_base = logits + bos * stride_ln
        offs_bq = tl.arange(0, BQ * H) + i_sh * (BQ * H)
        offs_boq = tl.arange(0, BQ) + i_sh * BQ
        offs_d = tl.arange(0, D)
        offs_w = offs_bq
        mask_bq = offs_bq < Q * H
        mask_d = offs_d < D
        mask_boq = offs_boq < Q

        q_ptr = q_base + offs_bq[:, None] * stride_qh + offs_d[None, :] * stride_qd
        q_msk = mask_bq[:, None] & mask_d[None, :]
        q_blk = tl.load(q_ptr, q_msk, 0.0).to(tl.float16)  # [BQ*H, D]

        w_ptr = w_base + offs_w * stride_wh
        w_msk = mask_bq
        w_blk = tl.load(w_ptr, w_msk, 0.0).to(tl.float16)  # [BQ*H]

        CK = tl.cdiv(CK, BK)
        for ck in range(CK, warp_specialize=True):
            offs_bk = ck * BK + tl.arange(0, BK)
            mask_bk = bos + offs_bk < eos
            k_ptr = k_base + offs_d[:, None] * stride_kd + offs_bk[None, :] * stride_kn
            k_msk = mask_d[:, None] & mask_bk[None, :]
            k_blk = tl.load(k_ptr, k_msk, 0.0).to(tl.float16)
            acc = tl.dot(q_blk, k_blk, out_dtype=tl.float16)  # [BQ*H, BK]
            acc = tl.maximum(acc, 0.0) * w_blk[:, None]
            out_blk = acc.trans().reshape(BK, BQ, H).sum(-1).trans()  # [BQ, BK]
            out_ptr = (
                o_base + offs_boq[:, None] * stride_lm + offs_bk[None, :] * stride_ln
            )
            out_msk = (
                mask_boq[:, None]
                & mask_bk[None, :]
                & (bos_vec[:, None] <= offs_bk[None, :] + bos)
                & (eos_vec[:, None] > offs_bk[None, :] + bos)
            )
            tl.store(out_ptr, out_blk.to(tl.float16), out_msk)


def triton_lighting_indexer_k_tiled_interface(
    q, kv, weights, cu_seqlen_ks, cu_seqlen_ke
):
    Q, H, D = q.shape[0], q.shape[1], q.shape[2]
    K = kv.shape[0]
    CU = cu_seqlen_ks.shape[0]
    logits = torch.full([Q, K], float("-inf"), device="cuda", dtype=torch.float32)
    BQ = 1
    BK = 64
    TK = 2048
    NQ = triton.cdiv(Q, BQ)
    NK = triton.cdiv(K, TK)
    grid = (NQ, NK)
    triton_lighting_indexer_k_tiled[grid](
        q,
        kv,
        cu_seqlen_ks,
        cu_seqlen_ke,
        weights,
        logits,
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        weights.stride(1),
        logits.stride(0),
        logits.stride(1),
        Q,
        H,
        K,
        TK,
        D,
        CU,
        BQ,
        BK,
    )
    return logits


# ===========================================================================
# GCU compiler-toolchain compatibility shim
# ===========================================================================
# Same family as compiler_issues/vllm_parallel_nsa_gcu_matmul.md: the
# 2026-11-07 20:22 in-process pass library emits GCUIR where this kernel's
# `gcu.matmul` carries operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 0, 0>
# plus an inputPrecision property. The old external gcu-compiler-opt tool
# rejects every encoding derived from it:
#   * 8 entries  -> "size mismatch" (extra trailing empty variadic segments);
#   * 4 entries  (what the parallel_nsa_compression shim leaves after stripping
#     all trailing zeros) -> "size mismatch in attribute conversion: 4 vs 5".
# The tool's legacy gcu.matmul layout needs exactly FIVE segment entries:
# array<i32: 1, 1, 1, 1, 0>. Verified offline against gcu-compiler-opt with the
# captured IR of this kernel (both with and without the inputPrecision
# property). The rewrite below is fallback-only and scoped to this kernel's
# name, so already-compilable kernels keep their exact code path. Remove the
# shim once the external tool re-syncs with the in-process passes.


def _downgrade_gcuir_for_external_tool(text):
    """Rewrite new-format GCUIR op encodings back to the old external-tool form."""
    import re

    def _strip_trailing_zero_segments(match):
        entries = [entry.strip() for entry in match.group(2).split(",")]
        while len(entries) > 1 and entries[-1] == "0":
            entries.pop()
        return match.group(1) + ", ".join(entries) + match.group(3)

    out_lines = []
    for line in text.splitlines():
        if '"memref_ext.slice_' in line or '"gcu.matmul"' in line:
            if "operandSegmentSizes = array<i32:" in line:
                line = re.sub(
                    r"(operandSegmentSizes = array<i32: )"
                    r"((?:\s*\d+\s*,\s*)*\s*\d+\s*)(>)",
                    _strip_trailing_zero_segments,
                    line,
                )
            if '"gcu.matmul"' in line:
                # Legacy gcu.matmul keeps one empty variadic segment entry.
                line = re.sub(
                    r"operandSegmentSizes = array<i32: (\d+(?:, \d+){3})>",
                    r"operandSegmentSizes = array<i32: \1, 0>",
                    line,
                )
                line = re.sub(r"inputPrecision = \d+ : i32, ", "", line)
                line = re.sub(r", inputPrecision = \d+ : i32", "", line)
        if '"memref_ext.memset_start"' in line:
            line = re.sub(
                r" <\{operandSegmentSizes = array<i32: [0-9, ]+>\}>", "", line
            )
        out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def _install_gcu_matmul_compat():
    """Wrap enflame toolkit.gcu_compiler_opt with a scoped downgrade retry."""
    try:
        from triton.backends.enflame import toolkit
    except ImportError:
        return
    if getattr(toolkit.gcu_compiler_opt, "_indexer_k_tiled_compat", False):
        return
    _prev = toolkit.gcu_compiler_opt

    def gcu_compiler_opt_compat(content, *args):
        try:
            return _prev(content, *args)
        except Exception as first_error:
            text = content if isinstance(content, str) else str(content)
            if "triton_lighting_indexer_k_tiled" not in text:
                raise first_error
            try:
                # Bypass the wrapper chain: inner shims would strip the
                # trailing zero segment that the legacy tool requires.
                return toolkit._run_command(
                    "gcu-compiler-opt",
                    _downgrade_gcuir_for_external_tool(text),
                    "-mlir-print-op-generic",
                    *args,
                )
            except Exception:
                raise first_error

    gcu_compiler_opt_compat._indexer_k_tiled_compat = True
    toolkit.gcu_compiler_opt = gcu_compiler_opt_compat


# _install_gcu_matmul_compat()  # DISABLED 2026-11-09: 私有 GCUIR shim 实测引入回归(降级重试路径
# 搞挂合法 kernel,同 conftest shim 事故);详见 commit b333be6/e1acdf6 与
# compiler_issues/vllm_memref_slice_pad_start.md。编译器修复后统一处置。


# ===========================================================================
# Wiring
# ===========================================================================


def _install():
    """Wire the GCU implementation into the direct-import entrypoint.

    tests/test_DSA/test_indexer_k_tiled.py imports
    ``from flaggems_vllm.ops.DSA.indexer_k_tiled import
    triton_lighting_indexer_k_tiled_interface``, a path the SpecOpRegistrar
    namespace swap cannot reach. The generic module is not exported from
    flaggems_vllm.ops, so import it here and replace the attribute in place,
    mirroring the _kunlunxin mhc_pre / _enflame bin_topk patterns.
    """
    import sys

    generic_name = "flaggems_vllm.ops.DSA.indexer_k_tiled"
    mod = sys.modules.get(generic_name)
    if mod is None:
        import flaggems_vllm.ops.DSA.indexer_k_tiled as generic_mod

        mod = generic_mod
    current = getattr(mod, "triton_lighting_indexer_k_tiled_interface", None)
    if current is not None and current.__module__ == generic_name:
        mod.triton_lighting_indexer_k_tiled_interface = (
            triton_lighting_indexer_k_tiled_interface
        )


try:
    _install()
except Exception:
    # pytest assertion-rewrite 上下文中 @triton.jit 源定位可能失败
    # (collect 期动态触碰 generic 模块触发);跳过接线不影响 op 路由
    # (SpecOpRegistrar 已保证),直连 import 入口的接线由运行期重试兜底。
    pass
