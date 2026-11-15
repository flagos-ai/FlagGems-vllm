# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""TP2 continuous-KV Q-resident prefill; private register-resident implementation."""
import hashlib as _pv_hashlib
import re as _pv_re

import triton.language as tl
from triton.runtime.cache import get_cache_manager as _pv_cache_manager
from triton.runtime.jit import JITFunction as _PVJITFunction

from flaggems_vllm.runtime.backend._metax.ops.flash_attention.single_prefill import (
    _private_nolicm_codegen,
    _scaled_online_softmax_t,
)
from flaggems_vllm.utils import libentry

PV_N = 128
PV_LAYOUT_COMPILES = []


def _rewrite_pv_layout(ir):
    """PV K8 for N32/N64; remove only allocations read by PV's operand1."""
    n = PV_N
    pattern = (
        "(?m)^#([A-Za-z0-9_]+) = #ttg\\.ma"
        "ca_mma<\\{versionMajor = 2, versi"
        "onMinor = 0, warpsPerCTA = \\[4, "
        "1\\], elementsMNK = \\[1, 1, 4\\](["
        "^\\n]*)$"
    )
    matches = list(_pv_re.finditer(pattern, ir))
    if len(matches) != 1:
        return None
    alias = matches[0][1]
    if f"tensor<128x64xf32, #{alias}>" not in ir:
        return None
    ir = _pv_re.sub(
        pattern,
        lambda m: "#"
        + f"{m[1]}"
        + " = #ttg.maca_mma<{versionMajor ="
        + " 2, versionMinor = 0, warpsPerCT"
        + "A = [4, 1], elementsMNK = [1, 1,"
        + " 8]"
        + m[2],
        ir,
    )
    pv = "#" + alias
    v_pattern = _pv_re.compile(
        "(?m)^([ \\t]*)(%[A-Za-z0-9_]+) = ttg\\.convert_layout (%[A-Za-z0-9_]+) : tensor<"
        + str(n)
        + "x128xbf16, #([A-Za-z0-9_]+)> -> tensor<"
        + str(n)
        + (
            "x128xbf16, #([A-Za-z0-9_]+)>([^\\"
            "n]*)\\n[ \\t]*(%[A-Za-z0-9_]+) = t"
            "t\\.trans \\2 \\{order = array<i32:"
            " 1, 0>\\} : tensor<"
        )
        + str(n)
        + "x128xbf16, #\\5> -> tensor<128x"
        + str(n)
        + "xbf16, #ttg\\.dot_op<\\{opIdx = 0, parent = "
        + _pv_re.escape(pv)
        + "\\}>>([^\\n]*)$"
    )

    def v_rewrite(m):
        (indent, temp, src, blocked, linear, loc1, dest, loc2) = m.groups()
        return (
            f"{indent}"
            + f"{temp}"
            + " = tt.trans "
            + f"{src}"
            + " {order = array<i32: 1, 0>} : te"
            + "nsor<"
            + f"{n}"
            + "x128xbf16, #"
            + f"{blocked}"
            + "> -> tensor<128x"
            + f"{n}"
            + "xbf16, #pv_local_transposed>"
            + f"{loc1}"
            + "\n"
            + f"{indent}"
            + f"{dest}"
            + " = ttg.convert_layout "
            + f"{temp}"
            + " : tensor<128x"
            + f"{n}"
            + "xbf16, #pv_local_transposed> -> "
            + "tensor<128x"
            + f"{n}"
            + "xbf16, #ttg.dot_op<{opIdx = 0, p"
            + "arent = "
            + f"{pv}"
            + "}>>"
            + f"{loc2}"
        )

    (ir, count) = v_pattern.subn(v_rewrite, ir)
    if count != 6:
        return None
    output_pattern = _pv_re.compile(
        (
            "(?m)^([ \\t]*)(%acc[01]_f32) = tt"
            "\\.trans (%[A-Za-z0-9_]+#[01]) \\{"
            "order = array<i32: 1, 0>\\} : ten"
            "sor<128x64xf32, "
        )
        + _pv_re.escape(pv)
        + "> -> tensor<64x128xf32, #([A-Za-z0-9_]+)>([^\\n]*)$"
    )

    def output_rewrite(m):
        (indent, dest, src, linear, loc) = m.groups()
        tmp = dest + "_pv_layout"
        return (
            f"{indent}"
            + f"{tmp}"
            + " = ttg.convert_layout "
            + f"{src}"
            + " : tensor<128x64xf32, "
            + f"{pv}"
            + "> -> tensor<128x64xf32, #pv_loca"
            + "l_transposed>"
            + f"{loc}"
            + "\n"
            + f"{indent}"
            + f"{dest}"
            + " = tt.trans "
            + f"{tmp}"
            + " {order = array<i32: 1, 0>} : te"
            + "nsor<128x64xf32, #pv_local_trans"
            + "posed> -> tensor<64x128xf32, #pv"
            + "_local_output>"
            + f"{loc}"
        )

    aliases = {m[4] for m in output_pattern.finditer(ir)}
    (ir, count) = output_pattern.subn(output_rewrite, ir)
    if count != 2:
        return None
    for name in aliases:
        ir = _pv_re.sub("(?m)^#" + _pv_re.escape(name) + " =[^\\n]*\\n", "", ir)
        ir = _pv_re.sub("#" + _pv_re.escape(name) + "\\b", "#pv_local_output", ir)
    loads = _pv_re.compile(
        "(?m)^([ \\t]*)(%[A-Za-z0-9_]+) = ttg.local_load (%[A-Za-z0-9_]+) : !ttg.memdesc<"
        + str(n)
        + "x64xbf16, #[A-Za-z0-9_]+, #smem> -> (tensor<"
        + str(n)
        + "x64xbf16, #ttg.dot_op<\\{opIdx = 1, parent = "
        + _pv_re.escape(pv)
        + "\\}>>)([^\\n]*)$"
    )
    edits = []
    declarations = set()
    for m in loads.finditer(ir):
        mem = m[3]
        declaration_pattern = _pv_re.compile(
            "(?m)^[ \\t]*"
            + _pv_re.escape(mem)
            + " = ttg.local_alloc (%[A-Za-z0-9_]+) : \\((tensor<"
            + str(n)
            + "x64xbf16, #[A-Za-z0-9_]+>)\\) -> !ttg.memdesc<"
            + str(n)
            + "x64xbf16, #[A-Za-z0-9_]+, #smem>[^\\n]*$"
        )
        preceding = list(declaration_pattern.finditer(ir, 0, m.start()))
        if not preceding:
            return None
        declaration = preceding[-1]
        replacement = (
            f"{m[1]}"
            + f"{m[2]}"
            + " = ttg.convert_layout "
            + f"{declaration[1]}"
            + " : "
            + f"{declaration[2]}"
            + " -> "
            + f"{m[4]}"
            + f"{m[5]}"
        )
        edits.append((m.start(), m.end(), replacement))
        declarations.add((declaration.start(), declaration.end()))
    if len(edits) != 3 or len(declarations) != 3:
        return None
    edits.extend(((begin, end, "") for (begin, end) in declarations))
    for begin, end, replacement in sorted(edits, reverse=True):
        ir = ir[:begin] + replacement + ir[end:]
    prefix = (
        "#pv_local_transposed = #ttg.bloc"
        "ked<{sizePerThread = [8, 1], thr"
        "eadsPerWarp = [16, 4], warpsPerC"
        "TA = [1, 4], order = [0, 1]}>\n#p"
        "v_local_output = #ttg.blocked<{s"
        "izePerThread = [1, 8], threadsPe"
        "rWarp = [4, 16], warpsPerCTA = ["
        "4, 1], order = [1, 0]}>\n"
    )
    lines = []
    count = 0
    for line in ir.splitlines():
        lines.append(line)
        if (
            "ttg.convert_layout" in line
            and "-> tensor<" in line
            and ("#ttg.dot_op<" in line)
            and ("parent = " + pv + "}" in line)
        ):
            lines.append(line[: len(line) - len(line.lstrip())] + "ttg.barrier_shared")
            count += 1
    if count != 9:
        return None
    return _hoist_q_register_loads(
        _qresident_warp_layout(prefix + "\n".join(lines) + "\n")
    )


class _PrivatePVLayoutJIT(_PVJITFunction):

    def _do_compile(self, key, signature, device, constexprs, options, attrs, warmup):
        base = super()._do_compile(
            key, signature, device, constexprs, options, attrs, warmup
        )
        if base is None or not hasattr(base, "asm") or options.ir_override:
            return base
        (cache, _, target, backend, _) = self.device_caches[device]
        if target.backend != "maca" or options.num_warps != 4:
            return base
        ir = _rewrite_pv_layout(base.asm["ttgir"])
        if ir is None:
            PV_LAYOUT_COMPILES.append(
                {
                    "optimized": False,
                    "reason": "unsupported IR layout",
                    "base_hash": base.hash,
                }
            )
            return base
        digest = _pv_hashlib.sha256(ir.encode()).hexdigest()
        manager = _pv_cache_manager(digest)
        path = manager.get_file("attention_pv_layout.ttgir") or manager.put(
            ir, "attention_pv_layout.ttgir"
        )
        modified = backend.parse_options(dict(options.__dict__, ir_override=path))
        kernel = super()._do_compile(
            key, signature, device, constexprs, modified, attrs, warmup
        )
        (kernel, codegen) = _private_nolicm_codegen(kernel, backend, target)
        cache[key] = kernel
        PV_LAYOUT_COMPILES.append(
            {
                "optimized": True,
                "codegen": codegen,
                "base_hash": base.hash,
                "kernel_hash": kernel.hash,
                "ir_sha256": digest,
                "ir_file": path,
                "base_ir_sha256": _pv_hashlib.sha256(
                    base.asm["ttgir"].encode()
                ).hexdigest(),
            }
        )
        return kernel


def _pv_layout_jit(**options):
    return lambda fn: _PrivatePVLayoutJIT(fn, **options)


@libentry()
@_pv_layout_jit(do_not_specialize=["total_q", "host_q_len", "host_k_len"])
def flash_varlen_fwd_d256_tn_single_prefill_kernel_rf(
    DESC,
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    softmax_lse_ptr,
    q_row_stride,
    k_row_stride,
    v_row_stride,
    q_head_stride,
    k_head_stride,
    v_head_stride,
    o_row_stride,
    o_head_stride,
    scale_softmax: tl.constexpr,
    scale_softmax_log2: tl.constexpr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    seqused_k_ptr,
    IS_CU_SEQLENS_K: tl.constexpr,
    IS_SEQUSED_K: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_RATIO: tl.constexpr,
    total_q,
    host_q_len,
    host_k_len,
    page_table_ptr,
    page_table_batch_stride: tl.constexpr,
    k_page_stride,
    v_page_stride,
    worklist_ptr,
    COMPACT_WORKLIST: tl.constexpr,
    GRID_ORDER: tl.constexpr,
    REVERSE_Q_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QK_BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    PACK_GROUP: tl.constexpr,
    TN_SCENARIO: tl.constexpr,
    TP2_FIRST_TILE: tl.constexpr = False,
    PAGE_SIZE: tl.constexpr = 32,
    MASK_K_TAIL: tl.constexpr = True,
    FULL_Q: tl.constexpr = False,
    PAD_K: tl.constexpr = 0,
    PAD_Q: tl.constexpr = 0,
):
    tl.static_assert(
        BLOCK_K == 128
        and QK_BLOCK_K == 64
        and (BLOCK_N == 32 or BLOCK_N == 64 or BLOCK_N == 128),
        "D256 TN requires QK BK64, PV D128, BN128",
    )
    tl.static_assert(not COMPACT_WORKLIST, "Fast attention requires a rectangular grid")
    if GRID_ORDER == 1:
        head_pid = tl.program_id(0)
        task_pid = tl.program_id(2)
    else:
        task_pid = tl.program_id(0)
        head_pid = tl.program_id(2)
    if REVERSE_Q_TILES:
        if GRID_ORDER == 1:
            task_pid = tl.num_programs(2) - 1 - task_pid
        else:
            task_pid = tl.num_programs(0) - 1 - task_pid
    hid = head_pid
    if COMPACT_WORKLIST:
        descriptor_offset = task_pid * 2
        bid = tl.load(DESC)
        if bid < 0:
            return
        m_block = tl.load(worklist_ptr + descriptor_offset + 1).to(tl.int32)
    else:
        bid = tl.load(DESC)
        if bid < 0:
            return
        m_block = task_pid
    q_bos = tl.load(cu_seqlens_q_ptr + bid)
    q_len = tl.load(cu_seqlens_q_ptr + bid + 1) - q_bos
    effective_q_len = q_len * PACK_GROUP
    if m_block * BLOCK_M >= effective_q_len:
        return
    k_len = tl.load(seqused_k_ptr + bid)
    packed_row = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    packed_row = tl.max_contiguous(tl.multiple_of(packed_row, BLOCK_M), BLOCK_M)
    row_idx = packed_row // PACK_GROUP
    query_head = hid * PACK_GROUP + packed_row % PACK_GROUP
    kv_head = hid // (HEAD_RATIO // PACK_GROUP)
    q_block_end = tl.cdiv((m_block + 1) * BLOCK_M, PACK_GROUP)
    n_block_min = 0
    n_block_max = min(
        tl.cdiv(k_len, BLOCK_N), tl.cdiv(q_block_end + k_len - q_len, BLOCK_N)
    )
    n_masking_steps = min(n_block_max - n_block_min, 2)
    q_base = q_ptr
    o_base = o_ptr + q_bos * o_row_stride
    k_base = k_ptr + kv_head * k_head_stride
    v_base = v_ptr + kv_head * v_head_stride
    page_table_ptr += bid * page_table_batch_stride
    d_idx = tl.arange(0, BLOCK_K)
    output_d_offset = 0
    acc0 = tl.zeros((BLOCK_K, BLOCK_M), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_K, BLOCK_M), dtype=tl.float32)
    row_max = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    qk_d_idx = tl.arange(0, QK_BLOCK_K)
    cached_q_ptr = (
        q_base
        + (row_idx[:, None] * NUM_HEADS + query_head[:, None]) * QK_BLOCK_K
        + qk_d_idx[None, :]
    )
    q0 = tl.load(cached_q_ptr + 0 * PAD_Q * NUM_HEADS * QK_BLOCK_K)
    q1 = tl.load(cached_q_ptr + 1 * PAD_Q * NUM_HEADS * QK_BLOCK_K)
    q2 = tl.load(cached_q_ptr + 2 * PAD_Q * NUM_HEADS * QK_BLOCK_K)
    q3 = tl.load(cached_q_ptr + 3 * PAD_Q * NUM_HEADS * QK_BLOCK_K)
    n_block = n_block_max - 1
    for mask_step in tl.static_range(0, 2):
        if mask_step < n_masking_steps:
            start_n = n_block * BLOCK_N
            col_idx = tl.max_contiguous(
                tl.multiple_of(start_n + tl.arange(0, BLOCK_N), BLOCK_N), BLOCK_N
            )
            k_cache_offset = col_idx * 64
            scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            k_tile_ptr = k_base + qk_d_idx[:, None] + k_cache_offset[None, :]
            kp0 = tl.load(k_tile_ptr)
            kp1 = tl.load(k_tile_ptr + PAD_K * QK_BLOCK_K)
            scores = tl.dot(q0, kp0, scores)
            kp2 = tl.load(k_tile_ptr + 2 * PAD_K * QK_BLOCK_K)
            scores = tl.dot(q1, kp1, scores)
            kp3 = tl.load(k_tile_ptr + 3 * PAD_K * QK_BLOCK_K)
            scores = tl.dot(q2, kp2, scores)
            scores = tl.dot(q3, kp3, scores)
            scores_t = tl.trans(scores)
            causal_end = tl.minimum(k_len - 1, row_idx + k_len - q_len)
            scores_t = tl.where(
                col_idx[:, None] > causal_end[None, :], float("-inf"), scores_t
            )
            v_cache_offset = col_idx * 128
            if PAGE_SIZE == 16 or MASK_K_TAIL:
                v0 = tl.load(
                    v_base + v_cache_offset[:, None] + d_idx[None, :],
                    mask=col_idx[:, None] < k_len,
                    other=0.0,
                )
            else:
                v0 = tl.load(v_base + v_cache_offset[:, None] + d_idx[None, :])
            (acc_scale, probabilities, row_max, row_sum) = _scaled_online_softmax_t(
                scores_t,
                row_max,
                row_sum,
                scale_softmax_log2=scale_softmax_log2,
                IS_BORDER=True,
            )
            probabilities = probabilities.to(v_ptr.type.element_ty)
            if not mask_step == 0:
                acc0 = acc0 * acc_scale[None, :]
            if mask_step == 0:
                acc0 = tl.dot(tl.trans(v0), probabilities)
            else:
                acc0 = tl.dot(tl.trans(v0), probabilities, acc0)
            if not mask_step == 0:
                acc1 = acc1 * acc_scale[None, :]
            if PAGE_SIZE == 16 or MASK_K_TAIL:
                v1 = tl.load(
                    v_base
                    + v_cache_offset[:, None]
                    + (PAD_K * BLOCK_K + d_idx)[None, :],
                    mask=col_idx[:, None] < k_len,
                    other=0.0,
                )
            else:
                v1 = tl.load(
                    v_base
                    + v_cache_offset[:, None]
                    + (PAD_K * BLOCK_K + d_idx)[None, :]
                )
            if mask_step == 0:
                acc1 = tl.dot(tl.trans(v1), probabilities)
            else:
                acc1 = tl.dot(tl.trans(v1), probabilities, acc1)
            pass
            n_block -= 1
    main_start = n_block_max - n_masking_steps - 1
    for n_block in tl.range(main_start, n_block_min - 1, step=-1):
        start_n = n_block * BLOCK_N
        col_idx = tl.max_contiguous(
            tl.multiple_of(start_n + tl.arange(0, BLOCK_N), BLOCK_N), BLOCK_N
        )
        k_cache_offset = col_idx * 64
        scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        k_tile_ptr = k_base + qk_d_idx[:, None] + k_cache_offset[None, :]
        kp0 = tl.load(k_tile_ptr)
        kp1 = tl.load(k_tile_ptr + PAD_K * QK_BLOCK_K)
        scores = tl.dot(q0, kp0, scores)
        kp2 = tl.load(k_tile_ptr + 2 * PAD_K * QK_BLOCK_K)
        scores = tl.dot(q1, kp1, scores)
        kp3 = tl.load(k_tile_ptr + 3 * PAD_K * QK_BLOCK_K)
        scores = tl.dot(q2, kp2, scores)
        scores = tl.dot(q3, kp3, scores)
        scores_t = tl.trans(scores)
        v_cache_offset = col_idx * 128
        v0 = tl.load(v_base + v_cache_offset[:, None] + d_idx[None, :])
        (acc_scale, probabilities, row_max, row_sum) = _scaled_online_softmax_t(
            scores_t,
            row_max,
            row_sum,
            scale_softmax_log2=scale_softmax_log2,
            IS_BORDER=False,
        )
        probabilities = probabilities.to(v_ptr.type.element_ty)
        acc0 = acc0 * acc_scale[None, :]
        acc0 = tl.dot(tl.trans(v0), probabilities, acc0)
        acc1 = acc1 * acc_scale[None, :]
        v1 = tl.load(
            v_base + v_cache_offset[:, None] + (PAD_K * BLOCK_K + d_idx)[None, :]
        )
        acc1 = tl.dot(tl.trans(v1), probabilities, acc1)
        pass
    o_mask = True if FULL_Q else row_idx[:, None] < q_len
    o0_ptr = (
        o_base
        + row_idx[:, None] * o_row_stride
        + query_head[:, None] * o_head_stride
        + output_d_offset
        + d_idx[None, :]
    )
    o1_ptr = (
        o_base
        + row_idx[:, None] * o_row_stride
        + query_head[:, None] * o_head_stride
        + BLOCK_K
        + d_idx[None, :]
    )
    lse = tl.where(
        (row_sum == 0) | (row_sum != row_sum),
        float("-inf"),
        row_max * (scale_softmax / scale_softmax_log2) + tl.log(row_sum),
    )
    inv_sum = tl.where((row_sum == 0) | (row_sum != row_sum), 1.0, 1.0 / row_sum)
    acc0_f32 = tl.trans(acc0) * inv_sum[:, None]
    acc1_f32 = tl.trans(acc1) * inv_sum[:, None]
    tl.store(o0_ptr, acc0_f32.to(o_ptr.type.element_ty), mask=o_mask)
    tl.store(o1_ptr, acc1_f32.to(o_ptr.type.element_ty), mask=o_mask)
    lse_mask = True if FULL_Q else row_idx < q_len
    tl.store(
        softmax_lse_ptr + query_head * total_q + q_bos + row_idx, lse, mask=lse_mask
    )


def _hoist_q_register_loads(ir):
    """Convert each cached Q slab to MMA registers once, before all N loops."""
    assert "%q_rf_" not in ir
    declarations = _pv_re.compile(
        (
            "(?m)^    (%q([0-3])[A-Za-z0-9_]*"
            ") = ttg.local_alloc (%q[A-Za-z0-"
            "9_]*) : \\(tensor<64x64xbf16, #[A"
            "-Za-z0-9_]+>\\) -> (!ttg.memdesc<"
            "64x64xbf16, #[A-Za-z0-9_]+, #sme"
            "m>)([^\\n]*)$"
        )
    )
    roots = list(declarations.finditer(ir))
    assert len(roots) == 4 and {m[2] for m in roots} == set("0123"), [
        (m[1], m[2]) for m in roots
    ]
    edits = []
    replacements = {}
    for root in roots:
        (mem, index, source, memory_type, loc) = root.groups()
        loads = _pv_re.compile(
            "(?m)^[ \\t]*(%q[0-3][A-Za-z0-9_]*) = ttg.local_load "
            + _pv_re.escape(mem)
            + " : "
            + _pv_re.escape(memory_type)
            + " -> (tensor<64x64xbf16, #ttg.dot_op<\\{opIdx = 0, parent = #[A-Za-z0-9_]+\\}>>)([^\\n]*)$"
        )
        uses = list(loads.finditer(ir))
        assert len(uses) == 3, (mem, len(uses))
        assert len({m[2] for m in uses}) == 1
        name = "%q_rf_" + index
        addition = (
            "\n    ttg.barrier_shared\n"
            + f"    {name} = ttg.local_load {mem} : {memory_type} -> {uses[0][2]}{loc}\n"
            + "    ttg.barrier_shared"
        )
        edits.append((root.end(), root.end(), addition))
        for m in uses:
            edits.append((m.start(), m.end(), ""))
            replacements[m[1]] = name
    for begin, end, text in sorted(edits, reverse=True):
        ir = ir[:begin] + text + ir[end:]
    for source, dest in replacements.items():
        ir = _pv_re.sub(_pv_re.escape(source) + "\\b", dest, ir)
    return ir


def _qresident_warp_layout(ir):
    """Put warps on Q rows so the cached Q operand is not replicated in N."""
    pattern = _pv_re.compile(
        (
            "tt.trans %[A-Za-z0-9_]+ \\{order "
            "= array<i32: 1, 0>\\} : tensor<64"
            "x(32|64|128)xf32, #([A-Za-z0-9_]"
            "+)> -> tensor<(?:32|64|128)x64xf"
            "32, #([A-Za-z0-9_]+)>"
        )
    )
    transposes = list(pattern.finditer(ir))
    assert len(transposes) == 3, len(transposes)
    ns = {int(m[1]) for m in transposes}
    parents = {m[2] for m in transposes}
    linear = {m[3] for m in transposes}
    assert len(ns) == len(parents) == len(linear) == 1
    n = ns.pop()
    parent = parents.pop()
    alias = linear.pop()
    definition = _pv_re.compile(
        "(?m)^#" + _pv_re.escape(parent) + " = #ttg.maca_mma<[^\\n]*$"
    )
    m = definition.search(ir)
    assert m is not None
    if "warpsPerCTA = [4, 1]" in m[0]:
        return ir
    assert "warpsPerCTA = [2, 2]" in m[0], m[0]
    ir = (
        ir[: m.start()]
        + m[0].replace("warpsPerCTA = [2, 2]", "warpsPerCTA = [4, 1]")
        + ir[m.end() :]
    )
    assert n in (64, 128), n
    registers = (
        [[1, 0], [2, 0], [0, 1], [0, 2], [64, 0]]
        if n == 128
        else [[1, 0], [0, 1], [0, 2], [32, 0]]
    )
    lanes = (
        [[4, 0], [8, 0], [16, 0], [32, 0], [0, 4], [0, 8]]
        if n == 128
        else [[2, 0], [4, 0], [8, 0], [16, 0], [0, 4], [0, 8]]
    )
    layout = (
        f"#{alias} = #ttg.linear<{{register = "
        + str(registers)
        + ", lane = "
        + str(lanes)
        + ", warp = [[0, 16], [0, 32]], block = []}>"
    )
    (ir, count) = _pv_re.subn(
        "(?m)^#" + _pv_re.escape(alias) + " = #ttg.linear<[^\\n]*$", layout, ir
    )
    assert count == 1, count
    return ir
