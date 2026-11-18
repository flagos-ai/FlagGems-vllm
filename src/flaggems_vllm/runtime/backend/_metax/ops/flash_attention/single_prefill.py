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

"""Page32/GQA8 and page16/GQA4 single-prefill over fragment-major K/V.

All input classification, collection and output writes run on the GPU. A valid
group has one long prefill with single-token queries, or one short query in a two-request batch. Rejected groups
retain the original attention route on every graph replay. KV copying and the
decode path belong to the operator; no vendor attention call is used.


The private JIT wrapper is specific to this kernel. It preserves the AST
launcher signature and shared-memory contract while choosing PV layout and
backend codegen options. Explicit barriers protect layout scratch lifetimes.
Unsupported IR layouts retain the original compiled kernel.
"""
import copy
import copy as _cg_copy
import hashlib as _pv_hashlib
import json as _cg_json
import re as _pv_re
import types as _cg_types
from pathlib import Path as _CGPath

import torch
import triton
import triton.backends.metax.compiler as _cg_backend_module
import triton.language as tl
from triton.runtime.cache import get_cache_manager as _pv_cache_manager
from triton.runtime.jit import JITFunction as _PVJITFunction

from flaggems_vllm.runtime.backend._metax.ops.flash_attention.common import (
    tn_compile_scenario,
)
from flaggems_vllm.utils import libentry


@triton.jit
def _select_single_prefill(
    CU,
    USED,
    DESC,
    DCU,
    DU,
    FCU,
    B: tl.constexpr,
    R: tl.constexpr,
    MIN_Q: tl.constexpr = 4108,
):
    b = tl.arange(0, R)
    start = tl.load(CU + b, b <= B, other=0)
    end = tl.load(CU + b + 1, b < B, other=0)
    q = end - start
    large = (b < B) & (q > MIN_Q)
    count = tl.sum(large.to(tl.int32), 0)
    other = tl.sum(((b < B) & (q != 1) & ~large).to(tl.int32), 0)
    valid = (count == 1) & (other == 0)
    bid = tl.max(tl.where(large, b, -1), 0)
    tl.store(DESC, tl.where(valid, bid, -1))
    used = tl.load(USED + b, b < B, other=0)
    tl.store(DU + b, tl.where((q == 1) & valid, used, 0), b < B)
    tl.store(DCU + b, b, b <= B)
    tl.store(FCU + b, tl.where(valid, 0, start), b <= B)


@triton.jit
def _single_prefill_gather(
    K,
    V,
    KC,
    VC,
    T,
    USED,
    DESC,
    KP: tl.constexpr,
    VP: tl.constexpr,
    KR: tl.constexpr,
    VR: tl.constexpr,
    TS: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK: tl.constexpr,
    PAGE_SIZE: tl.constexpr = 32,
):
    bid = tl.load(DESC)
    if bid < 0:
        return
    length = tl.load(USED + bid)
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n = x // 256
    d = x % 256
    valid = (n < length) & (n < PAD)
    p = tl.load(T + bid * TS + n // PAGE_SIZE, valid, other=0).to(tl.int64)
    k = tl.load(K + p * KP + n % PAGE_SIZE * KR + d, valid, other=0)
    v = tl.load(V + p * VP + n % PAGE_SIZE * VR + d, valid, other=0)
    tl.store(KC + d // 64 * PAD * 64 + n * 64 + d % 64, k, n < PAD)
    tl.store(VC + d // 128 * PAD * 128 + n * 128 + d % 128, v, n < PAD)


@triton.jit
def _single_prefill_gather_q(
    Q,
    QC,
    CU,
    DESC,
    QR: tl.constexpr,
    QH: tl.constexpr,
    PAD_Q: tl.constexpr,
    BLOCK: tl.constexpr,
    HEADS: tl.constexpr = 8,
):
    bid = tl.load(DESC)
    if bid < 0:
        return
    qbos = tl.load(CU + bid).to(tl.int64)
    qlen = tl.load(CU + bid + 1).to(tl.int64) - qbos
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    fragment = x // (PAD_Q * HEADS * 64)
    row_head = x // 64 % (PAD_Q * HEADS)
    row = row_head // HEADS
    head = row_head % HEADS
    d = fragment * 64 + x % 64
    value = tl.load(
        Q + (qbos + row) * QR + head * QH + d,
        (x < PAD_Q * HEADS * 256) & (row < qlen),
        other=0.0,
    )
    tl.store(QC + x, value, x < PAD_Q * HEADS * 256)


@triton.jit
def _copy_decode_q(
    Q,
    DQ,
    CU,
    QR: tl.constexpr,
    QH: tl.constexpr,
    BLOCK: tl.constexpr,
    HEADS: tl.constexpr = 8,
):
    b = tl.program_id(0)
    start = tl.load(CU + b)
    end = tl.load(CU + b + 1)
    x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    q = tl.load(
        Q + start * QR + x // 256 * QH + x % 256,
        (x < (HEADS * 256)) & (end - start == 1),
        other=0,
    )
    tl.store(DQ + b * (HEADS * 256) + x, q, x < (HEADS * 256))


@triton.jit
def _scatter_decode(
    DESC,
    O,
    L,
    DO,
    DL,
    CU,
    OR: tl.constexpr,
    OH: tl.constexpr,
    TOTAL: tl.constexpr,
    B: tl.constexpr,
    DH: tl.constexpr,
    DR: tl.constexpr,
    HEADS: tl.constexpr = 8,
):
    if tl.load(DESC) < 0:
        return
    b = tl.program_id(0)
    start = tl.load(CU + b)
    end = tl.load(CU + b + 1)
    if end - start != 1:
        return
    x = tl.arange(0, (HEADS * 256))
    values = tl.load(DO + b * (HEADS * 256) + x)
    tl.store(O + start * OR + x // 256 * OH + x % 256, values)
    h = tl.arange(0, HEADS)
    decode_lse = tl.load(DL + h * DH + b * DR)
    tl.store(L + h * TOTAL + start, decode_lse)


@triton.jit
def _scaled_online_softmax_t(
    scores_t, row_max, row_sum, scale_softmax_log2, IS_BORDER: tl.constexpr
):
    scores_t *= scale_softmax_log2
    previous_max = row_max
    row_max = tl.maximum(row_max, tl.max(scores_t, 0))
    if IS_BORDER:
        current_max = tl.where(row_max == float("-inf"), 0.0, row_max)
    else:
        current_max = row_max
    accumulator_scale = tl.math.exp2(previous_max - current_max)
    row_sum *= accumulator_scale
    max_scaled = tl.where(row_max == float("-inf"), 0.0, row_max)
    probabilities_t = tl.math.exp2(scores_t - max_scaled[None, :])
    row_sum += tl.sum(probabilities_t, 0)
    return (accumulator_scale, probabilities_t, row_max, row_sum)


PV_LAYOUT_COMPILES = []


def _rewrite_pv_layout(ir):
    pattern = (
        "(?m)^#([\\w]+) = #ttg\\.maca_mma<\\{versionMajor = 2, versionMinor = 0, wa"
        "rpsPerCTA = \\[4, 1\\], elementsMNK = \\[1, 1, 4\\]([^\\n]*)$"
    )
    matches = list(_pv_re.finditer(pattern, ir))
    if len(matches) != 1:
        return None
    alias = matches[0].group(1)
    if f"tensor<128x64xf32, #{alias}>" not in ir:
        return None
    ir = _pv_re.sub(
        pattern,
        lambda m: (
            "#"
            f"{m[1]}"
            " = #ttg.maca_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1]"
            ", elementsMNK = [1, 1, 8]"
        )
        + m[2],
        ir,
    )
    pv = "#" + alias
    v_pattern = _pv_re.compile(
        (
            "(?m)^(\\s*)(%[\\w]+) = ttg\\.convert_layout (%[\\w]+) : tensor<128x128xbf1"
            "6, #([\\w]+)> -> tensor<128x128xbf16, #([\\w]+)>([^\\n]*)\\n\\s*(%[\\w]+) "
            "= tt\\.trans \\2 \\{order = array<i32: 1, 0>\\} : tensor<128x128xbf16, #\\"
            "5> -> tensor<128x128xbf16, #ttg\\.dot_op<\\{opIdx = 0, parent = "
        )
        + _pv_re.escape(pv)
        + ("\\}>>([^\\n]*)$")
    )

    def v_rewrite(m):
        (indent, temp, src, blocked, linear, loc1, dest, loc2) = m.groups()
        return (
            f"{indent}"
            f"{temp}"
            " = tt.trans "
            f"{src}"
            " {order = array<i32: 1, 0>} : tensor<128x128xbf16, #"
            f"{blocked}"
            "> -> tensor<128x128xbf16, #pv_local_transposed>"
            f"{loc1}"
            "\n"
            f"{indent}"
            f"{dest}"
            " = ttg.convert_layout "
            f"{temp}"
            " : tensor<128x128xbf16, #pv_local_transposed> -> tensor<128x128xbf16, #ttg"
            ".dot_op<{opIdx = 0, parent = "
            f"{pv}"
            "}>>"
            f"{loc2}"
        )

    (ir, v_count) = v_pattern.subn(v_rewrite, ir)
    if v_count != 6:
        return None
    o_pattern = _pv_re.compile(
        (
            "(?m)^(\\s*)(%acc[01]_f32) = tt\\.trans (%[\\w]+#[01]) \\{order = array<i32"
            ": 1, 0>\\} : tensor<128x64xf32, "
        )
        + _pv_re.escape(pv)
        + ("> -> tensor<64x128xf32, #([\\w]+)>([^\\n]*)$")
    )

    def o_rewrite(m):
        (indent, dest, src, linear, loc) = m.groups()
        temp = dest + "_pv_layout"
        return (
            f"{indent}"
            f"{temp}"
            " = ttg.convert_layout "
            f"{src}"
            " : tensor<128x64xf32, "
            f"{pv}"
            "> -> tensor<128x64xf32, #pv_local_transposed>"
            f"{loc}"
            "\n"
            f"{indent}"
            f"{dest}"
            " = tt.trans "
            f"{temp}"
            " {order = array<i32: 1, 0>} : tensor<128x64xf32, #pv_local_transposed> -> "
            "tensor<64x128xf32, #pv_local_output>"
            f"{loc}"
        )

    output_aliases = [m.group(4) for m in o_pattern.finditer(ir)]
    (ir, o_count) = o_pattern.subn(o_rewrite, ir)
    if o_count != 2:
        return None
    for alias in set(output_aliases):
        ir = _pv_re.sub("(?m)^#" + _pv_re.escape(alias) + " =[^\\n]*\\n", "", ir)
        ir = ir.replace("#" + alias, "#pv_local_output")
    alloc = {}
    count = 0
    lines = []
    for line in ir.splitlines():
        m = _pv_re.search(
            (
                "(%[\\w]+) = ttg.local_alloc (%[\\w]+) : \\(tensor<128x64xbf16, #([\\w]+)>"
                "\\) -> !ttg.memdesc<128x64xbf16, #([\\w]+), #smem>"
            ),
            line,
        )
        if m:
            alloc[m[1]] = (m[2], m[3])
            continue
        m = _pv_re.search(
            (
                "(%[\\w]+) = ttg.local_load (%[\\w]+) : !ttg.memdesc<128x64xbf16, #[\\w]+, "
                "#smem> -> tensor<128x64xbf16, #ttg.dot_op<\\{opIdx = 1, parent = "
            )
            + _pv_re.escape(pv)
            + ("\\}>>"),
            line,
        )
        if m:
            if m[2] not in alloc:
                return None
            (src, src_layout) = alloc[m[2]]
            line = line[: len(line) - len(line.lstrip())] + (
                f"{m[1]}"
                " = ttg.convert_layout "
                f"{src}"
                " : tensor<128x64xbf16, #"
                f"{src_layout}"
                "> -> tensor<128x64xbf16, #ttg.dot_op<{opIdx = 1, parent = "
                f"{pv}"
                "}>>"
            )
            count += 1
        lines.append(line)
    if count != 3:
        return None
    prefix = (
        "#pv_local_transposed = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWar"
        "p = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>\n#pv_local_output = #t"
        "tg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA "
        "= [4, 1], order = [1, 0]}>\n"
    )
    result = prefix + "\n".join(lines) + "\n"
    lines = []
    count = 0
    for line in result.splitlines():
        lines.append(line)
        if (
            ("ttg.convert_layout") in line
            and ("-> tensor<") in line
            and (("#ttg.dot_op<") in line)
            and (("parent = ") + pv + ("}") in line)
        ):
            indent = line[: len(line) - len(line.lstrip())]
            lines.append(indent + "ttg.barrier_shared")
            count += 1
    if count != 9:
        return None
    return "\n".join(lines) + "\n"


def _private_nolicm_codegen(program, backend, target):
    flag = " -mllvm -metaxgpu-disable-licm=true "
    original_hash = program.hash
    recipe = _pv_hashlib.sha256(
        ("attention-nolicm-v1:" + original_hash + flag).encode()
    ).hexdigest()
    manager = _pv_cache_manager(recipe)
    binary_path = manager.get_file("attention_nolicm.mcfatbin")
    effective = []
    if binary_path:
        binary = _CGPath(binary_path).read_bytes()
    else:
        method = backend.make_mcfatbin
        original_translate = _cg_backend_module.metax.translate_llvmir_to_mcfatbin

        def local_translate(src, mxcc, maca, options):
            effective.append(options + flag)
            return original_translate(src, mxcc, maca, options + flag)

        namespace = method.__globals__.copy()
        namespace["metax"] = _cg_types.SimpleNamespace(
            translate_llvmir_to_mcfatbin=local_translate
        )
        private_make = _cg_types.FunctionType(
            method.__code__,
            namespace,
            name=("private_attention_nolicm"),
            argdefs=method.__defaults__,
            closure=method.__closure__,
        )
        binary = private_make(program.asm["llir"], {}, program.metadata, target.arch)
        binary_path = manager.put(binary, "attention_nolicm.mcfatbin", binary=True)
    new_hash = _pv_hashlib.sha256(
        (recipe + _pv_hashlib.sha256(binary).hexdigest()).encode()
    ).hexdigest()
    changed = _cg_copy.copy(program)
    changed.kernel = binary
    changed.hash = new_hash
    changed.metadata = program.metadata._replace(hash=new_hash)
    changed.packed_metadata = backend.pack_metadata(changed.metadata)
    changed.asm = dict(program.asm)
    changed.asm[backend.binary_ext] = binary
    changed.metadata_group = dict(program.metadata_group)
    for name, path in list(changed.metadata_group.items()):
        if name.endswith("." + backend.binary_ext):
            changed.metadata_group[name] = binary_path
        if name.endswith(".json"):
            record = _cg_json.loads(_CGPath(path).read_text())
            record["hash"] = new_hash
            changed.metadata_group[name] = manager.put(
                _cg_json.dumps(record), name, binary=False
            )
    changed.module = None
    changed.function = None
    changed._run = None
    record = dict(
        recipe=recipe,
        extra_codegen_flag=flag,
        pre_codegen_hash=original_hash,
        kernel_hash=new_hash,
        binary_sha256=_pv_hashlib.sha256(binary).hexdigest(),
        binary_file=binary_path,
        original_binary_sha256=_pv_hashlib.sha256(program.kernel).hexdigest(),
        effective_options=effective,
        global_compiler_mutation=False,
        ast_launcher_signature_preserved=changed.src is program.src,
        shared_bytes_preserved=changed.metadata.shared == program.metadata.shared,
    )
    manager.put(_cg_json.dumps(record), "attention_codegen_record.json", binary=False)
    return (changed, record)


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
                ("optimized"): True,
                ("codegen"): codegen,
                ("base_hash"): base.hash,
                ("kernel_hash"): kernel.hash,
                ("ir_sha256"): digest,
                ("ir_file"): path,
                ("base_ir_sha256"): _pv_hashlib.sha256(
                    base.asm[("ttgir")].encode()
                ).hexdigest(),
            }
        )
        return kernel


def _pv_layout_jit(**options):
    return lambda fn: _PrivatePVLayoutJIT(fn, **options)


@libentry()
@_pv_layout_jit(do_not_specialize=["total_q", "host_q_len", "host_k_len"])
def flash_varlen_fwd_d256_tn_single_prefill_kernel(
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
        BLOCK_K == 128 and QK_BLOCK_K == 64 and (BLOCK_N == 128),
        ("D256 TN requires QK BK64, PV D128, BN128"),
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
    n_block = n_block_max - 1
    for mask_step in tl.static_range(0, 2):
        if mask_step < n_masking_steps:
            start_n = n_block * BLOCK_N
            col_idx = tl.max_contiguous(
                tl.multiple_of(start_n + tl.arange(0, BLOCK_N), BLOCK_N), BLOCK_N
            )
            k_cache_offset = col_idx * 64
            scores_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            q_tile_ptr = (
                q_base
                + (row_idx[None, :] * NUM_HEADS + query_head[None, :]) * QK_BLOCK_K
                + qk_d_idx[:, None]
            )
            k_tile_ptr = k_base + qk_d_idx[None, :] + k_cache_offset[:, None]
            for _d_block in tl.range(0, 256 // QK_BLOCK_K):
                k_tile = tl.load(k_tile_ptr)
                q_tile = tl.load(q_tile_ptr)
                scores_t = tl.dot(k_tile, q_tile, scores_t)
                q_tile_ptr += PAD_Q * NUM_HEADS * QK_BLOCK_K
                k_tile_ptr += PAD_K * QK_BLOCK_K
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
        scores_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        q_tile_ptr = (
            q_base
            + (row_idx[None, :] * NUM_HEADS + query_head[None, :]) * QK_BLOCK_K
            + qk_d_idx[:, None]
        )
        k_tile_ptr = k_base + qk_d_idx[None, :] + k_cache_offset[:, None]
        for _d_block in tl.range(0, 256 // QK_BLOCK_K):
            k_tile = tl.load(k_tile_ptr)
            q_tile = tl.load(q_tile_ptr)
            scores_t = tl.dot(k_tile, q_tile, scores_t)
            q_tile_ptr += PAD_Q * NUM_HEADS * QK_BLOCK_K
            k_tile_ptr += PAD_K * QK_BLOCK_K
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
        float(("-inf")),
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


def can_use_tp2_single_prefill(params, num_splits):
    return (
        num_splits == 0
        and 1 <= params.b <= 64
        and (
            (
                params.h == 8
                and 4108 < params.seqlen_q <= 16384
                and 16384 <= params.seqlen_k <= 67584
            )
            or (
                params.h == 4
                and 4096 < params.seqlen_q <= 16384
                and 4096 <= params.seqlen_k <= 67584
            )
        )
        and (
            params.total_q == params.seqlen_q + params.b - 1
            or (params.b == 2 and 2 <= params.total_q - params.seqlen_q <= 16)
            or (
                params.h == 8
                and params.block_size == 32
                and 2 <= params.b <= 64
                and params.seqlen_k >= 32768
                and params.b - 1 < params.total_q - params.seqlen_q <= 4096
                and (params.b > 2 or params.total_q - params.seqlen_q > 16)
            )
        )
        and (params.h in (4, 8))
        and (params.hk == 1)
        and (params.d == 256)
        and params.is_paged
        and (
            (params.h == 8 and params.block_size == 32)
            or (params.h == 4 and params.block_size == 16)
        )
        and params.is_cu_seqlens_q
        and params.is_seqused_k
        and (not params.seqlenq_ngroups_swapped)
        and params.is_causal
        and (not params.is_local)
        and (not params.is_dropout)
        and (not params.is_alibi)
        and (not params.is_softcap)
        and (params.q_ptr.dtype == torch.bfloat16)
        and all(
            (
                x.stride(-1) == 1
                for x in (params.q_ptr, params.k_ptr, params.v_ptr, params.o_ptr)
            )
        )
    )


def _launch_tp2_single_token_group(params, *, num_splits, fallback):
    from flaggems_vllm.ops.attention import flash_attn_varlen_func

    heads = params.h
    (q, k, v, out) = (params.q_ptr, params.k_ptr, params.v_ptr, params.o_ptr)
    (cu, used, table) = (
        params.cu_seqlens_q_ptr,
        params.seqused_k_ptr,
        params.page_table_ptr,
    )
    (b, total, max_q, max_k) = (
        params.b,
        params.total_q,
        params.seqlen_q,
        params.seqlen_k,
    )
    pad = triton.cdiv(max_k, 128) * 128
    desc = torch.empty((1,), dtype=torch.int32, device=q.device)
    dcu = torch.empty((b + 1,), dtype=torch.int32, device=q.device)
    fcu = torch.empty_like(dcu)
    du = torch.empty((b,), dtype=torch.int32, device=q.device)
    dq = torch.empty((b, heads, 256), dtype=q.dtype, device=q.device)
    do = torch.empty_like(dq)
    kc = torch.empty((pad, 1, 256), dtype=k.dtype, device=k.device)
    vc = torch.empty_like(kc)
    pad_q = triton.cdiv(max_q, 128) * 128
    qc = torch.empty((4, pad_q * heads, 64), dtype=q.dtype, device=q.device)
    lse = params.softmax_lse_ptr
    _select_single_prefill[1,](
        cu,
        used,
        desc,
        dcu,
        du,
        fcu,
        b,
        triton.next_power_of_2(b + 1),
        MIN_Q=4096 if heads == 4 else 4108,
        num_warps=1,
    )
    _single_prefill_gather[triton.cdiv(pad * 256, 1024),](
        k,
        v,
        kc,
        vc,
        table,
        used,
        desc,
        k.stride(0),
        v.stride(0),
        k.stride(1),
        v.stride(1),
        table.stride(0),
        pad,
        1024,
        num_warps=4,
        num_stages=1,
        PAGE_SIZE=params.block_size,
    )
    _single_prefill_gather_q[triton.cdiv(pad_q * heads * 256, 1024),](
        q,
        qc,
        cu,
        desc,
        q.stride(0),
        q.stride(1),
        pad_q,
        1024,
        num_warps=4,
        num_stages=1,
        HEADS=heads,
    )
    _copy_decode_q[b, triton.cdiv(heads * 256, 1024)](
        q, dq, cu, q.stride(0), q.stride(1), 1024, num_warps=4, HEADS=heads
    )
    (_, dl) = flash_attn_varlen_func(
        q=dq,
        k=k,
        v=v,
        cu_seqlens_q=dcu,
        seqused_k=du,
        max_seqlen_q=1,
        max_seqlen_k=max_k,
        block_table=table,
        out=do,
        return_softmax_lse=True,
        causal=True,
        num_splits=0,
        softmax_scale=params.scale_softmax,
    )
    _scatter_decode[b,](
        desc,
        out,
        lse,
        do,
        dl,
        cu,
        out.stride(0),
        out.stride(1),
        total,
        b,
        dl.stride(0),
        dl.stride(1),
        num_warps=4,
        HEADS=heads,
    )
    scenario = tn_compile_scenario(q, kc, vc, out, lse, cu, used, table, desc)
    kernel = flash_varlen_fwd_d256_tn_single_prefill_kernel
    if heads == 8:
        from flaggems_vllm.runtime.backend._metax.ops.flash_attention.single_prefill_rf import (
            flash_varlen_fwd_d256_tn_single_prefill_kernel_rf,
        )

        kernel = flash_varlen_fwd_d256_tn_single_prefill_kernel_rf
    result = kernel[triton.cdiv(max_q * heads, 64), 1, 1](
        desc,
        qc,
        kc,
        vc,
        out,
        lse,
        q.stride(0),
        256,
        256,
        q.stride(1),
        256,
        256,
        out.stride(0),
        out.stride(1),
        params.scale_softmax,
        params.scale_softmax_log2,
        cu,
        None,
        used,
        False,
        True,
        heads,
        heads,
        total,
        max_q,
        max_k,
        table,
        table.stride(0),
        0,
        0,
        table,
        COMPACT_WORKLIST=False,
        GRID_ORDER=0,
        REVERSE_Q_TILES=True,
        BLOCK_M=64,
        BLOCK_N=128,
        BLOCK_K=128,
        QK_BLOCK_K=64,
        num_warps=4,
        num_stages=4,
        PACK_GROUP=heads,
        TN_SCENARIO=scenario,
        TP2_FIRST_TILE=False,
        PAGE_SIZE=params.block_size,
        MASK_K_TAIL=False,
        FULL_Q=False,
        PAD_K=pad,
        PAD_Q=pad_q,
        pipeline=("cpasync"),
        scenario=scenario,
        inner_stages=(0, 0),
        pipeline_load_num=-1,
    )
    old = copy.copy(params)
    old.cu_seqlens_q_ptr = fcu
    fallback(old, num_splits=num_splits, _skip_single_prefill=True)
    return result


@triton.jit
def _select_short_prefill(
    CU,
    USED,
    DESC,
    DCU,
    DU,
    FCU,
    B: tl.constexpr,
    R: tl.constexpr,
    HOST_SHORT: tl.constexpr,
    HOST_Q: tl.constexpr,
    MIN_Q: tl.constexpr = 4108,
):
    b = tl.arange(0, R)
    start = tl.load(CU + b, b <= B, other=0)
    end = tl.load(CU + b + 1, b < B, other=0)
    q = end - start
    large = (b < B) & (q > MIN_Q)
    short = (b < B) & (q > 0) & (q <= HOST_SHORT)
    count = tl.sum(large.to(tl.int32), 0)
    other = tl.sum(((b < B) & (q != 0) & ~large & ~short).to(tl.int32), 0)
    short_total = tl.sum(tl.where(short, q, 0), 0)
    big_q = tl.max(tl.where(large, q, 0), 0)
    valid = (
        (count == 1) & (other == 0) & (short_total == HOST_SHORT) & (big_q == HOST_Q)
    )
    bid = tl.max(tl.where(large, b, -1), 0)
    tl.store(DESC, tl.where(valid, bid, -1))
    used = tl.load(USED + b, b < B, other=0)
    tl.store(DU + b, tl.where(short & valid, used, 0), b < B)
    length = tl.where(valid, tl.where(short, q, 0), tl.where(b == 0, HOST_SHORT, 0))
    prefix = tl.cumsum(length, 0) - length
    tl.store(DCU + b, prefix, b <= B)
    tl.store(FCU + b, tl.where(valid, 0, start), b <= B)


@triton.jit
def _copy_short_decode_q(
    Q,
    DQ,
    CU,
    DCU,
    DESC,
    QR: tl.constexpr,
    QH: tl.constexpr,
    BLOCK: tl.constexpr,
    HOST_SHORT: tl.constexpr,
    HEADS: tl.constexpr = 8,
):
    if tl.load(DESC) < 0:
        return
    b = tl.program_id(0)
    start = tl.load(CU + b).to(tl.int64)
    length = tl.load(CU + b + 1).to(tl.int64) - start
    if length <= 0 or length > HOST_SHORT:
        return
    dst = tl.load(DCU + b)
    x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    row = x // (HEADS * 256)
    hd = x % (HEADS * 256)
    value = tl.load(
        Q + (start + row) * QR + hd // 256 * QH + hd % 256, row < length, other=0.0
    )
    tl.store(DQ + (dst + row) * (HEADS * 256) + hd, value, row < length)


@triton.jit
def _scatter_short_decode(
    DESC,
    O,
    L,
    DO,
    DL,
    CU,
    DCU,
    OR: tl.constexpr,
    OH: tl.constexpr,
    TOTAL: tl.constexpr,
    B: tl.constexpr,
    DH: tl.constexpr,
    DR: tl.constexpr,
    HOST_SHORT: tl.constexpr,
    HEADS: tl.constexpr = 8,
):
    if tl.load(DESC) < 0:
        return
    b = tl.program_id(0)
    row = tl.program_id(1)
    start = tl.load(CU + b).to(tl.int64)
    length = tl.load(CU + b + 1).to(tl.int64) - start
    if length <= 0 or length > HOST_SHORT or row >= length:
        return
    src = tl.load(DCU + b) + row
    x = tl.arange(0, (HEADS * 256))
    value = tl.load(DO + src * (HEADS * 256) + x)
    tl.store(O + (start + row) * OR + x // 256 * OH + x % 256, value)
    h = tl.arange(0, HEADS)
    lse = tl.load(DL + h * DH + src * DR)
    tl.store(L + h * TOTAL + start + row, lse)


def _launch_tp2_short_query_group(params, *, num_splits, fallback):
    from flaggems_vllm.ops.attention import flash_attn_varlen_func

    heads = params.h
    (q, k, v, out) = (params.q_ptr, params.k_ptr, params.v_ptr, params.o_ptr)
    (cu, used, table) = (
        params.cu_seqlens_q_ptr,
        params.seqused_k_ptr,
        params.page_table_ptr,
    )
    (b, total, max_q, max_k) = (
        params.b,
        params.total_q,
        params.seqlen_q,
        params.seqlen_k,
    )
    short_total = total - max_q
    pad = triton.cdiv(max_k, 128) * 128
    desc = torch.empty((1,), dtype=torch.int32, device=q.device)
    dcu = torch.empty((b + 1,), dtype=torch.int32, device=q.device)
    fcu = torch.empty_like(dcu)
    du = torch.empty((b,), dtype=torch.int32, device=q.device)
    dq = torch.empty((short_total, heads, 256), dtype=q.dtype, device=q.device)
    do = torch.empty_like(dq)
    kc = torch.empty((pad, 1, 256), dtype=k.dtype, device=k.device)
    vc = torch.empty_like(kc)
    pad_q = triton.cdiv(max_q, 128) * 128
    qc = torch.empty((4, pad_q * heads, 64), dtype=q.dtype, device=q.device)
    lse = params.softmax_lse_ptr
    _select_short_prefill[1,](
        cu,
        used,
        desc,
        dcu,
        du,
        fcu,
        b,
        triton.next_power_of_2(b + 1),
        short_total,
        max_q,
        MIN_Q=4096 if heads == 4 else 4108,
        num_warps=1,
    )
    _single_prefill_gather[triton.cdiv(pad * 256, 1024),](
        k,
        v,
        kc,
        vc,
        table,
        used,
        desc,
        k.stride(0),
        v.stride(0),
        k.stride(1),
        v.stride(1),
        table.stride(0),
        pad,
        1024,
        num_warps=4,
        num_stages=1,
        PAGE_SIZE=params.block_size,
    )
    _single_prefill_gather_q[triton.cdiv(pad_q * heads * 256, 1024),](
        q,
        qc,
        cu,
        desc,
        q.stride(0),
        q.stride(1),
        pad_q,
        1024,
        num_warps=4,
        num_stages=1,
        HEADS=heads,
    )
    _copy_short_decode_q[b, triton.cdiv(heads * 256, 1024) * short_total](
        q,
        dq,
        cu,
        dcu,
        desc,
        q.stride(0),
        q.stride(1),
        1024,
        short_total,
        num_warps=4,
        HEADS=heads,
    )
    (_, dl) = flash_attn_varlen_func(
        q=dq,
        k=k,
        v=v,
        cu_seqlens_q=dcu,
        seqused_k=du,
        max_seqlen_q=short_total,
        max_seqlen_k=max_k,
        block_table=table,
        out=do,
        return_softmax_lse=True,
        causal=True,
        num_splits=0,
        softmax_scale=params.scale_softmax,
    )
    _scatter_short_decode[b, short_total](
        desc,
        out,
        lse,
        do,
        dl,
        cu,
        dcu,
        out.stride(0),
        out.stride(1),
        total,
        b,
        dl.stride(0),
        dl.stride(1),
        short_total,
        num_warps=4,
        HEADS=heads,
    )
    scenario = tn_compile_scenario(q, kc, vc, out, lse, cu, used, table, desc)
    kernel = flash_varlen_fwd_d256_tn_single_prefill_kernel
    if heads == 8:
        from flaggems_vllm.runtime.backend._metax.ops.flash_attention.single_prefill_rf import (
            flash_varlen_fwd_d256_tn_single_prefill_kernel_rf,
        )

        kernel = flash_varlen_fwd_d256_tn_single_prefill_kernel_rf
    result = kernel[triton.cdiv(max_q * heads, 64), 1, 1](
        desc,
        qc,
        kc,
        vc,
        out,
        lse,
        q.stride(0),
        256,
        256,
        q.stride(1),
        256,
        256,
        out.stride(0),
        out.stride(1),
        params.scale_softmax,
        params.scale_softmax_log2,
        cu,
        None,
        used,
        False,
        True,
        heads,
        heads,
        total,
        max_q,
        max_k,
        table,
        table.stride(0),
        0,
        0,
        table,
        COMPACT_WORKLIST=False,
        GRID_ORDER=0,
        REVERSE_Q_TILES=True,
        BLOCK_M=64,
        BLOCK_N=128,
        BLOCK_K=128,
        QK_BLOCK_K=64,
        num_warps=4,
        num_stages=4,
        PACK_GROUP=heads,
        TN_SCENARIO=scenario,
        TP2_FIRST_TILE=False,
        PAGE_SIZE=params.block_size,
        MASK_K_TAIL=False,
        FULL_Q=False,
        PAD_K=pad,
        PAD_Q=pad_q,
        pipeline=("cpasync"),
        scenario=scenario,
        inner_stages=(0, 0),
        pipeline_load_num=-1,
    )
    old = copy.copy(params)
    old.cu_seqlens_q_ptr = fcu
    fallback(old, num_splits=num_splits, _skip_single_prefill=True)
    return result


def launch_tp2_single_prefill(params, *, num_splits, fallback):
    if (
        params.h == 8
        and params.block_size == 32
        and 2 <= params.b <= 64
        and params.seqlen_k >= 32768
        and params.b - 1 < params.total_q - params.seqlen_q <= 4096
        and (params.b > 2 or params.total_q - params.seqlen_q > 16)
    ):
        return _launch_tp2_mixed_query_group(
            params, num_splits=num_splits, fallback=fallback
        )
    if params.b == 2 and 2 <= params.total_q - params.seqlen_q <= 16:
        return _launch_tp2_short_query_group(
            params, num_splits=num_splits, fallback=fallback
        )
    return _launch_tp2_single_token_group(
        params, num_splits=num_splits, fallback=fallback
    )


def _launch_tp2_mixed_query_group(params, *, num_splits, fallback):
    heads = params.h
    (q, k, v, out) = (params.q_ptr, params.k_ptr, params.v_ptr, params.o_ptr)
    (cu, used, table) = (
        params.cu_seqlens_q_ptr,
        params.seqused_k_ptr,
        params.page_table_ptr,
    )
    (b, total, max_q, max_k) = (
        params.b,
        params.total_q,
        params.seqlen_q,
        params.seqlen_k,
    )
    short_total = total - max_q
    # Admission caps Q/K. Fixed pitches avoid specializing on each length.
    pad = 67584
    desc = torch.empty((1,), dtype=torch.int32, device=q.device)
    dcu = torch.empty((b + 1,), dtype=torch.int32, device=q.device)
    fcu = torch.empty_like(dcu)
    du = torch.empty((b,), dtype=torch.int32, device=q.device)
    dq = torch.empty((short_total, heads, 256), dtype=q.dtype, device=q.device)
    do = torch.empty_like(dq)
    row_map = torch.empty((short_total,), dtype=torch.int32, device=q.device)
    kc = torch.empty((pad, 1, 256), dtype=k.dtype, device=k.device)
    vc = torch.empty_like(kc)
    pad_q = 16384
    qc = torch.empty((4, pad_q * heads, 64), dtype=q.dtype, device=q.device)
    lse = params.softmax_lse_ptr
    _select_mixed_prefill[1,](
        cu,
        used,
        desc,
        dcu,
        du,
        fcu,
        b,
        triton.next_power_of_2(b + 1),
        short_total,
        max_q,
        MIN_Q=4096 if heads == 4 else 4108,
        num_warps=1,
    )
    _single_prefill_gather[triton.cdiv(pad * 256, 1024),](
        k,
        v,
        kc,
        vc,
        table,
        used,
        desc,
        k.stride(0),
        v.stride(0),
        k.stride(1),
        v.stride(1),
        table.stride(0),
        pad,
        1024,
        num_warps=4,
        num_stages=1,
        PAGE_SIZE=params.block_size,
    )
    _single_prefill_gather_q[triton.cdiv(pad_q * heads * 256, 1024),](
        q,
        qc,
        cu,
        desc,
        q.stride(0),
        q.stride(1),
        pad_q,
        1024,
        num_warps=4,
        num_stages=1,
        HEADS=heads,
    )
    _build_short_row_map[b,](cu, dcu, desc, row_map, short_total, num_warps=1)
    _mapped_short_q[triton.cdiv(short_total * heads * 256, 1024),](
        q,
        dq,
        desc,
        row_map,
        q.stride(0),
        q.stride(1),
        short_total,
        heads,
        1024,
        num_warps=4,
    )
    from flaggems_vllm.runtime.backend._metax.ops.flash_attention.splitkv import (
        launch_d256_tn_compact_splitkv as launch_d256_tn_splitkv,
    )

    dl = torch.empty((heads, short_total), dtype=torch.float32, device=q.device)
    residual = copy.copy(params)
    residual.q_ptr = dq
    residual.o_ptr = do
    residual.softmax_lse_ptr = dl
    residual.cu_seqlens_q_ptr = dcu
    residual.seqused_k_ptr = du
    residual.cu_seqlens_k_ptr = None
    residual.is_cu_seqlens_k = False
    residual.total_q = short_total
    residual.seqlen_q = short_total
    residual.q_row_stride = dq.stride(0)
    residual.q_head_stride = dq.stride(1)
    residual.o_row_stride = do.stride(0)
    residual.o_head_stride = do.stride(1)
    launch_d256_tn_splitkv(
        residual,
        max_seqlen_q=short_total,
        batch_size=b,
        num_heads=heads,
        total_q=short_total,
        num_splits=8,
    )
    _mapped_short_scatter[triton.cdiv(short_total * heads * 256, 1024),](
        out,
        lse,
        do,
        dl,
        desc,
        row_map,
        out.stride(0),
        out.stride(1),
        total,
        dl.stride(0),
        dl.stride(1),
        short_total,
        heads,
        1024,
        num_warps=4,
    )
    scenario = tn_compile_scenario(q, kc, vc, out, lse, cu, used, table, desc)
    kernel = flash_varlen_fwd_d256_tn_single_prefill_kernel
    if heads == 8:
        from flaggems_vllm.runtime.backend._metax.ops.flash_attention.single_prefill_rf import (
            flash_varlen_fwd_d256_tn_single_prefill_kernel_rf,
        )

        kernel = flash_varlen_fwd_d256_tn_single_prefill_kernel_rf
    result = kernel[triton.cdiv(max_q * heads, 64), 1, 1](
        desc,
        qc,
        kc,
        vc,
        out,
        lse,
        q.stride(0),
        256,
        256,
        q.stride(1),
        256,
        256,
        out.stride(0),
        out.stride(1),
        params.scale_softmax,
        params.scale_softmax_log2,
        cu,
        None,
        used,
        False,
        True,
        heads,
        heads,
        total,
        max_q,
        max_k,
        table,
        table.stride(0),
        0,
        0,
        table,
        COMPACT_WORKLIST=False,
        GRID_ORDER=0,
        REVERSE_Q_TILES=True,
        BLOCK_M=64,
        BLOCK_N=128,
        BLOCK_K=128,
        QK_BLOCK_K=64,
        num_warps=4,
        num_stages=4,
        PACK_GROUP=heads,
        TN_SCENARIO=scenario,
        TP2_FIRST_TILE=False,
        PAGE_SIZE=params.block_size,
        MASK_K_TAIL=False,
        FULL_Q=False,
        PAD_K=pad,
        PAD_Q=pad_q,
        pipeline="cpasync",
        scenario=scenario,
        inner_stages=(0, 0),
        pipeline_load_num=-1,
    )
    old = copy.copy(params)
    old.cu_seqlens_q_ptr = fcu
    fallback(old, num_splits=num_splits, _skip_single_prefill=True)
    return result


@triton.jit(do_not_specialize=["HOST_SHORT"])
def _build_short_row_map(CU, DCU, DESC, MAP, HOST_SHORT):
    if tl.load(DESC) < 0:
        return
    b = tl.program_id(0)
    start = tl.load(CU + b)
    length = tl.load(CU + b + 1) - start
    if length <= 0 or length > HOST_SHORT:
        return
    dst = tl.load(DCU + b)
    if length == 1:
        tl.store(MAP + dst, start)
    else:
        r = tl.arange(0, 128)
        for offset in tl.range(0, length, 128):
            tl.store(MAP + dst + offset + r, start + offset + r, offset + r < length)


@triton.jit(do_not_specialize=["ROWS"])
def _mapped_short_q(
    Q,
    DQ,
    DESC,
    MAP,
    QR: tl.constexpr,
    QH: tl.constexpr,
    ROWS,
    HEADS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if tl.load(DESC) < 0:
        return
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = x // (HEADS * 256)
    hd = x % (HEADS * 256)
    original = tl.load(MAP + row, row < ROWS, other=0).to(tl.int64)
    value = tl.load(
        Q + original * QR + hd // 256 * QH + hd % 256, row < ROWS, other=0.0
    )
    tl.store(DQ + x, value, row < ROWS)


@triton.jit(do_not_specialize=["ROWS", "DH", "TOTAL"])
def _mapped_short_scatter(
    O,
    L,
    DO,
    DL,
    DESC,
    MAP,
    OR: tl.constexpr,
    OH: tl.constexpr,
    TOTAL,
    DH,
    DR: tl.constexpr,
    ROWS,
    HEADS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if tl.load(DESC) < 0:
        return
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = x // (HEADS * 256)
    hd = x % (HEADS * 256)
    head = hd // 256
    original = tl.load(MAP + row, row < ROWS, other=0).to(tl.int64)
    value = tl.load(DO + x, row < ROWS, other=0.0)
    tl.store(O + original * OR + head * OH + hd % 256, value, row < ROWS)
    mask = (row < ROWS) & (hd % 256 == 0)
    value_l = tl.load(DL + head * DH + row * DR, mask, other=0.0)
    tl.store(L + head * TOTAL + original, value_l, mask)


@triton.jit(do_not_specialize=["B", "HOST_SHORT", "HOST_Q"])
def _select_mixed_prefill(
    CU,
    USED,
    DESC,
    DCU,
    DU,
    FCU,
    B,
    R: tl.constexpr,
    HOST_SHORT,
    HOST_Q,
    MIN_Q: tl.constexpr = 4108,
):
    b = tl.arange(0, R)
    start = tl.load(CU + b, b <= B, other=0)
    end = tl.load(CU + b + 1, b < B, other=0)
    q = end - start
    large = (b < B) & (q > MIN_Q)
    short = (b < B) & (q > 0) & (q <= HOST_SHORT)
    count = tl.sum(large.to(tl.int32), 0)
    other = tl.sum(((b < B) & (q != 0) & ~large & ~short).to(tl.int32), 0)
    short_total = tl.sum(tl.where(short, q, 0), 0)
    big_q = tl.max(tl.where(large, q, 0), 0)
    valid = (
        (count == 1) & (other == 0) & (short_total == HOST_SHORT) & (big_q == HOST_Q)
    )
    bid = tl.max(tl.where(large, b, -1), 0)
    tl.store(DESC, tl.where(valid, bid, -1))
    used = tl.load(USED + b, b < B, other=0)
    tl.store(DU + b, tl.where(short & valid, used, 0), b < B)
    length = tl.where(valid, tl.where(short, q, 0), tl.where(b == 0, HOST_SHORT, 0))
    prefix = tl.cumsum(length, 0) - length
    tl.store(DCU + b, prefix, b <= B)
    tl.store(FCU + b, tl.where(valid, 0, start), b <= B)
