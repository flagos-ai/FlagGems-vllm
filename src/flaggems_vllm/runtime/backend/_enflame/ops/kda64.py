# SPDX-License-Identifier: Apache-2.0
"""GCU adapters for the installed vLLM 0.20.x 64-token KDA kernels.

The launch sequence and materialization dtypes follow vLLM's FLA KDA core,
with three GCU-side deviations from the generic (vLLM >= 0.23, framework
PR 454) adapters, all forced by the vLLM build pinned in this environment:

1. Gate units: the 0.20.x FLA kernels decay with ``tl.exp`` on natural-log
   cumulative gates. The public ``safe_kda_gate_chunk_cumsum`` output stays
   in log2 units (exact-match API), so the log2 gate is converted here with
   a tiny Triton scale kernel. Both the operator path and the vLLM-module
   reference shim convert the same log2 bits with the same kernel, keeping
   the two paths bit-identical.
2. State layout: 0.20.x ``chunk_gated_delta_rule_fwd_kernel_h_blockdim64``
   stores the chunk state as [V, K] while ``chunk_gla_fwd_kernel_o`` reads
   it as [K, V] (upstream fixed the reader in 0.21 with a transpose). The
   history is transposed between the two launches so the output kernel sees
   the intended values.
3. The state kernel has no ``USE_EXP2`` constexpr; the resolver rejects
   newer cores explicitly instead of silently miscomputing.

Request indices and offsets are supplied by the framework. The adapters
never invoke the core hosts that initialize workspaces with Torch compute
kernels.

4. Decode reroute: the 0.20.x ``fused_recurrent_gated_delta_rule_fwd_kernel``
   builds its [BV, BK] state tile with a 2-D bool mask whose row width equals
   BK. At BK >= 128 that mask lowers to the ``tops_cvt_v0_si8_to_v1_si8__4``
   packing intrinsic, whose ADDR16 ICALL relocation ld.lld rejects (see
   compiler_issues/ld_lld_relocation.md). The vLLM host pins BK to the next
   power of two of K, so every K > 64 decode is unlaunchable on this
   toolchain. ``recurrent_kda_vk`` reroutes those launches to a local kernel
   that mirrors the vLLM 0.20 math one-to-one but only emits 1-D bool masks
   (the state tile is accessed unmasked, which the exact-tile precondition
   ``K == BK and V % BV == 0`` makes safe).
"""

import os
import re
from functools import lru_cache
from importlib import import_module

import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.data_movement import fill
from flaggems_vllm.ops.kda64 import recurrent_kda_vk as _generic_recurrent_kda_vk
from flaggems_vllm.utils import libentry

LN2 = 0.6931471805599453


def _metadata_as_int32_payload(tensor):
    """Zero-copy int32 view for framework metadata consumed by scalar loads.

    torch_gcu (without TORCH_GCU_ENABLE_INT64_AND_UINT64=1) stores int64
    tensors as a compact int32 payload ("GCU not support Long use Int
    replace"): the value of element i lives in int32 word i. The FLA kernels
    read this metadata with runtime-offset scalar i64 loads, which lower to
    8-byte accesses at byte offset 8*i and fetch value[2i] |
    (value[2i+1] << 32) instead of value[i]; the chunked-state kernel then
    derives wild history pointers and aborts the device (verified: varlen
    prefill crashes with vLLM's int64 prepare_chunk_offsets output and passes
    with an int32 copy; see compiler_issues/int64_scalar_load_addressing.md
    and the rotary_embedding/indexer precedents). Re-exposing the payload as
    int32 words restores word-i addressing without any copy. Stride-taking
    callers must compute strides from the original tensor first: the view's
    element strides double, but element i of the original still lives at
    word i, so original element strides stay correct. Non-contiguous int64
    metadata is left untouched (the op-level validation already rejects it).
    """
    if tensor is None or tensor.dtype != torch.int64:
        return tensor
    if os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64", "") in ("1", "true"):
        return tensor
    if not tensor.is_contiguous():
        return tensor
    return tensor.view(torch.int32)


# The FlagTree wheel (0.7.0+enflame.git72408c41) emits MLIR that the external
# `gcu-compiler-opt` tool in /opt/triton_gcu (triton-gcu 1.0.20261020) does
# not accept: `memref_ext.*` / `gcu.matmul` ops carry a trailing zero operand
# segment the older tool did not register, so every kernel with 2-D tensor
# access (all vLLM FLA kernels included) fails first-compile with
# "operandSegmentSizes ... size mismatch". The shim below rewrites the MLIR
# text to the layout the external tool accepts, mirroring the proven
# flashmla_sparse precedent:
#   * `memref_ext.memset_start` must carry no properties at all;
#   * other `memref_ext.*` / `gcu.matmul` ops drop their trailing zero
#     segments (pure metadata, no operand rebinding);
#   * `memref.reinterpret_cast` / `memref.subview` need explicit segment
#     sizes derived from their INT64_MIN sentinel counts when missing.
# It is process-global (the vLLM site-package kernels compile through the
# same backend entry), idempotent, and leaves already-compatible IR
# untouched. Remove once the wheel and the external tool are aligned (see
# compiler_issues/vllm_moe_sum.md and vllm_kda_bounded.md).

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
            content = "\n".join(
                _fix_gcu_dialect_line(line) for line in content.split("\n")
            )
        return _orig(content, *args)

    _shim._flaggems_dialect_shim = True
    _toolkit.gcu_compiler_opt = _shim


def _downgrade_gcuir_for_external_tool(text):
    """Rewrite GCUIR op encodings back to the legacy external-tool form.

    Applies the shared dialect fixes above (idempotent on already-rewritten
    text) and then restores the legacy 4-operand ``gcu.matmul`` layout: the
    tool's attribute table wants exactly five segment entries
    (``array<i32: 1, 1, 1, 1, 0>``) and no ``inputPrecision`` property,
    while the wheel's in-process passes emit four entries. Verified against
    gcu-compiler-opt with the captured IR of the vLLM FLA kernels; same
    family as compiler_issues/vllm_parallel_nsa_gcu_matmul.md.
    """
    out_lines = []
    for line in text.splitlines():
        line = _fix_gcu_dialect_line(line)
        if '"gcu.matmul"' in line:
            # Legacy gcu.matmul keeps one empty variadic segment entry.
            line = re.sub(
                r"operandSegmentSizes = array<i32: (\d+(?:, \d+){3})>",
                r"operandSegmentSizes = array<i32: \1, 0>",
                line,
            )
            line = re.sub(r"inputPrecision = \d+ : i32, ", "", line)
            line = re.sub(r", inputPrecision = \d+ : i32", "", line)
        out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def _patch_gcu_matmul_compat():
    import triton.backends.enflame.toolkit as _toolkit

    if getattr(_toolkit.gcu_compiler_opt, "_flaggems_kda_matmul_compat", False):
        return
    _prev = _toolkit.gcu_compiler_opt

    def _compat(content, *args):
        try:
            return _prev(content, *args)
        except Exception as first_error:
            text = content if isinstance(content, str) else str(content)
            if '"gcu.matmul"' not in text or "gcu.matmul" not in str(first_error):
                raise first_error
            try:
                # Bypass the wrapper chain: the dialect shims above would
                # strip the trailing zero segment the legacy tool requires.
                return _toolkit._run_command(
                    "gcu-compiler-opt",
                    _downgrade_gcuir_for_external_tool(text),
                    *args,
                )
            except Exception:
                raise first_error

    _compat._flaggems_kda_matmul_compat = True
    _toolkit.gcu_compiler_opt = _compat


# _patch_gcu_compiler_opt()  # DISABLED 2026-11-09: 私有 GCUIR shim 实测引入回归,详见 commit b333be6
# _patch_gcu_matmul_compat()  # DISABLED 2026-11-09: 私有 GCUIR shim 实测引入回归,详见 commit b333be6


def _kernel(module, name, required):
    kernel = getattr(module, name, None)
    original = kernel
    while kernel is not None and not hasattr(kernel, "arg_names"):
        kernel = getattr(kernel, "fn", None)
    if (
        kernel is None
        or not hasattr(original, "__getitem__")
        or not set(required).issubset(kernel.arg_names)
    ):
        raise NotImplementedError(f"The installed KDA core lacks {name} ABI")
    return original


def _arg_names(kernel):
    while kernel is not None and not hasattr(kernel, "arg_names"):
        kernel = getattr(kernel, "fn", None)
    return set(kernel.arg_names) if kernel is not None else set()


@lru_cache(maxsize=1)
def _prefill_core():
    """Resolve the vLLM 0.20.x kernel capabilities before any computation."""
    kda = import_module("vllm.model_executor.layers.fla.ops.kda")
    utils = import_module("vllm.model_executor.layers.fla.ops.solve_tril")
    delta_h = import_module("vllm.model_executor.layers.fla.ops.chunk_delta_h")
    state = _kernel(
        delta_h,
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
        ("k", "v", "w", "v_new", "gk", "h", "h0", "ht", "BT"),
    )
    if "USE_EXP2" in _arg_names(state):
        raise NotImplementedError(
            "The GCU KDA prefill adapter targets the vLLM 0.20.x FLA ABI "
            "(natural-exp kernels); the installed core exports USE_EXP2"
        )
    kernels = (
        _kernel(
            kda,
            "chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter",
            ("q", "k", "g", "beta", "A", "Aqk", "BT", "BC", "NC"),
        ),
        _kernel(
            kda,
            "chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra",
            ("q", "k", "g", "beta", "A", "Aqk", "BT", "BC", "BK"),
        ),
        _kernel(
            utils,
            "merge_16x16_to_64x64_inverse_kernel",
            ("A", "Ai", "BT", "USE_TMA", "DOT_PRECISION"),
        ),
        _kernel(
            kda,
            "recompute_w_u_fwd_kernel",
            ("q", "k", "kg", "v", "beta", "w", "u", "A", "gk", "BT"),
        ),
        state,
        _kernel(
            kda,
            "chunk_gla_fwd_kernel_o",
            ("q", "v", "g", "h", "o", "A", "BT", "scale"),
        ),
    )
    if not callable(getattr(kda, "l2norm_fwd", None)):
        raise NotImplementedError("The installed KDA core lacks l2norm_fwd")
    return kernels, kda.l2norm_fwd, utils.is_tma_supported, utils.FLA_TRIL_PRECISION


@lru_cache(maxsize=1)
def _recurrent_core():
    kda = import_module("vllm.model_executor.layers.fla.ops.kda")
    return _kernel(
        kda,
        "fused_recurrent_gated_delta_rule_fwd_kernel",
        (
            "q",
            "k",
            "v",
            "g",
            "beta",
            "o",
            "h0",
            "ht",
            "cu_seqlens",
            "ssm_state_indices",
            "IS_KDA",
            "INPLACE_FINAL_STATE",
        ),
    )


@libentry()
@triton.jit
def _gate_log2_to_nat_kernel(
    G,
    Y,
    SCALE,
    N,
    GRID_X: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Purely affine 1D access pattern: the current toolchain rejects 2D tile
    # lowering, and the op/reference paths must share one deterministic
    # conversion. No perf parameters are exposed, so no libtuner entry.
    pid = tl.program_id(0) + tl.program_id(1) * GRID_X
    i = pid * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    tl.store(Y + i, tl.load(G + i, mask, other=0) * SCALE, mask)


def _gate_log2_to_nat(gate):
    out = torch.empty_like(gate)
    numel = out.numel()
    if numel:
        block = 1024
        blocks = triton.cdiv(numel, block)
        grid_x = min(blocks, 65535)
        _gate_log2_to_nat_kernel[(grid_x, triton.cdiv(blocks, grid_x))](
            gate,
            out,
            LN2,
            numel,
            grid_x,
            block,
            num_warps=4,
        )
    return out


def prefill_kda64(
    q,
    k,
    v,
    gate,
    beta,
    scale,
    initial_state,
    output_final_state,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    core,
):
    """Run the 0.20.x chunk pipeline on an already-cumulative log2 gate."""
    kernels, _, use_tma, precision = core
    inter, intra, inverse, recompute, state, output = kernels
    b, t, h, dk = q.shape
    dv, bt = v.shape[-1], 64
    n = b if cu_seqlens is None else cu_seqlens.numel() - 1
    nt = triton.cdiv(t, bt) if cu_seqlens is None else chunk_indices.shape[0]
    # The FLA kernels load these with runtime-offset scalar i64 loads, which
    # misaddress GCU's compact int64 payload (see _metadata_as_int32_payload).
    cu_seqlens = _metadata_as_int32_payload(cu_seqlens)
    chunk_indices = _metadata_as_int32_payload(chunk_indices)
    chunk_offsets = _metadata_as_int32_payload(chunk_offsets)
    gate = _gate_log2_to_nat(gate)
    a = fill(torch.empty((b, t, h, bt), dtype=torch.float32, device=q.device), 0)
    aqk = fill(torch.empty_like(a), 0)
    common = dict(
        q=q,
        k=k,
        g=gate,
        beta=beta,
        A=a,
        Aqk=aqk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=t,
        H=h,
        K=dk,
        BT=bt,
        BC=16,
    )
    inter[(nt, 16, b * h)](**common, NC=4)
    intra[(nt, 4, b * h)](**common, BK=max(triton.next_power_of_2(dk), 16))
    ai = fill(torch.empty(a.shape, dtype=k.dtype, device=k.device), 0)
    inverse[(nt, b * h)](
        A=a,
        Ai=ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=t,
        H=h,
        BT=bt,
        USE_TMA=use_tma,
        DOT_PRECISION=precision,
    )
    w, u, kg = torch.empty_like(k), torch.empty_like(v), torch.empty_like(k)
    recompute[(nt, b * h)](
        q=None,
        k=k,
        qg=None,
        kg=kg,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=ai,
        gk=gate,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=t,
        H=h,
        K=dk,
        V=dv,
        BT=bt,
        BK=64,
        BV=64,
        DOT_PRECISION="ieee",
    )
    history_vk = torch.empty((b, nt, h, dv, dk), dtype=k.dtype, device=k.device)
    final = (
        torch.empty((n, h, dv, dk), dtype=torch.float32, device=k.device)
        if output_final_state
        else None
    )
    v_new = torch.empty_like(u)
    state[lambda meta: (triton.cdiv(dv, meta["BV"]), n * h)](
        k=kg,
        v=u,
        w=w,
        v_new=v_new,
        g=None,
        gk=gate,
        h=history_vk,
        h0=initial_state,
        ht=final,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=t,
        H=h,
        Hg=h,
        K=dk,
        V=dv,
        BT=bt,
    )
    # 0.20.x chunk_gla_fwd_kernel_o reads the chunk state as [K, V]; the
    # state kernel wrote [V, K]. Transposing the materialized history makes
    # the reader see the values the 0.21+ kernels (tl.trans read) would.
    history = history_vk.transpose(-2, -1).contiguous()
    output[lambda meta: (triton.cdiv(dv, meta["BV"]), nt, b * h)](
        q=q,
        v=v_new,
        g=gate,
        h=history,
        o=v,
        A=aqk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=t,
        H=h,
        K=dk,
        V=dv,
        BT=bt,
    )
    return v, final


@libentry()
@triton.jit(do_not_specialize=["N", "T"])
def _gcu_recurrent_kda_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    scale,
    N: tl.int64,
    T: tl.int64,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_FINAL_STATE: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
):
    # Line-by-line mirror of vLLM 0.20's fused_recurrent_gated_delta_rule_
    # fwd_kernel with IS_KDA=True, IS_SPEC_DECODING=False and NK=1, minus the
    # 2-D bool state mask: the exact-tile precondition checked by the host
    # guarantees o_k < K and o_v < V on the state tile, so h0/ht are accessed
    # unmasked and only 1-D bool masks remain (the 128-lane 2-D bool mask is
    # what makes the stock kernel un-linkable at BK >= 128).
    i_nh, i_v = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    if T == 0:
        return

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv
    p_gk = g + (bos * HV + i_hv) * K + o_k
    p_o = o + (bos * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(
                tl.int64
            )
            if state_idx <= 0:
                return
            p_h0 = h0 + state_idx * stride_init_state_token
        else:
            p_h0 = h0 + bos * HV * V * K
        p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0).to(tl.float32)

    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        # [BV, BK]
        b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
        b_h *= tl.exp(b_gk)[None, :]
        # [BV]
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        # [BV, BK]
        b_h += b_v[:, None] * b_k[None, :]
        # [BV]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        if INPLACE_FINAL_STATE:
            final_state_idx = tl.load(
                ssm_state_indices + i_n * stride_indices_seq + i_t * stride_indices_tok
            ).to(tl.int64)
            if final_state_idx > 0:
                p_ht = ht + final_state_idx * stride_final_state_token
                p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty))
        else:
            p_ht = ht + (bos + i_t) * stride_final_state_token
            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty))

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_gk += HV * K
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)


def _gcu_recurrent_kda_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state,
    inplace_final_state,
    use_qk_l2norm_in_kernel,
    cu_seqlens,
    ssm_state_indices,
):
    """vLLM 0.20 ``fused_recurrent_kda_fwd`` on the GCU-safe decode kernel."""
    b, t, h, dk = k.shape
    hv, dv = v.shape[2], v.shape[-1]
    n = b if cu_seqlens is None else len(cu_seqlens) - 1
    bk, bv = triton.next_power_of_2(dk), min(triton.next_power_of_2(dv), 8)
    if dk != bk or dv % bv:
        raise NotImplementedError(
            "The GCU KDA decode reroute covers exact K tiles and V blocks "
            f"(got K={dk}, BK={bk}, V={dv}, BV={bv}); the stock vLLM kernel "
            "cannot link at BK >= 128 on this toolchain"
        )
    o = torch.empty_like(v)
    final = (
        initial_state
        if inplace_final_state
        else q.new_empty(t, hv, dv, dk, dtype=initial_state.dtype)
    )
    stride_seq = 1 if ssm_state_indices is None else ssm_state_indices.stride(0)
    stride_tok = (
        1
        if ssm_state_indices is None or ssm_state_indices.ndim == 1
        else ssm_state_indices.stride(1)
    )
    # Strides are taken from the original tensors on purpose: after the
    # int32 payload view the element strides double, but element i of the
    # original int64 tensor still lives at word i, so the original element
    # strides keep addressing the right words.
    cu_seqlens = _metadata_as_int32_payload(cu_seqlens)
    ssm_state_indices = _metadata_as_int32_payload(ssm_state_indices)
    # Mutation is launched once at vLLM's fixed 1-warp/3-stage configuration;
    # autotuning this launch would repeatedly update live recurrent state.
    _gcu_recurrent_kda_fwd_kernel[(n * hv, triton.cdiv(dv, bv))](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        scale=scale,
        N=n,
        T=t,
        B=b,
        H=h,
        HV=hv,
        K=dk,
        V=dv,
        BK=bk,
        BV=bv,
        stride_init_state_token=initial_state.stride(0),
        stride_final_state_token=final.stride(0),
        stride_indices_seq=stride_seq,
        stride_indices_tok=stride_tok,
        USE_INITIAL_STATE=initial_state is not None,
        INPLACE_FINAL_STATE=inplace_final_state,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        IS_CONTINUOUS_BATCHING=ssm_state_indices is not None,
        num_warps=1,
        num_stages=3,
    )
    return o, final


def recurrent_kda_vk(
    q,
    k,
    v,
    gate,
    beta,
    scale,
    initial_state,
    inplace_final_state,
    use_qk_l2norm_in_kernel,
    cu_seqlens,
    ssm_state_indices,
    kernel,
):
    """Preserve vLLM's [slots,H,V,K] state and per-token final-state ABI."""
    # 1-D metadata is safe to re-expose for both launch paths (stride 1);
    # 2-D int64 slot maps keep their original strides, which only the GCU
    # reroute below accounts for when it takes the int32 payload view.
    cu_seqlens = _metadata_as_int32_payload(cu_seqlens)
    if ssm_state_indices is not None and ssm_state_indices.ndim == 1:
        ssm_state_indices = _metadata_as_int32_payload(ssm_state_indices)
    if triton.next_power_of_2(q.shape[-1]) < 128:
        return _generic_recurrent_kda_vk(
            q,
            k,
            v,
            gate,
            beta,
            scale,
            initial_state,
            inplace_final_state,
            use_qk_l2norm_in_kernel,
            cu_seqlens,
            ssm_state_indices,
            kernel,
        )
    return _gcu_recurrent_kda_fwd(
        q,
        k,
        v,
        gate,
        beta,
        scale,
        initial_state,
        inplace_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
        ssm_state_indices,
    )
