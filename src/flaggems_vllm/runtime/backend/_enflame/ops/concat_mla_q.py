# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""In-place MLA query concatenation without a Torch compute fallback."""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

# GCU400 launch constraints (measurements in the moe_sum adaptation):
#   - grid.x is hardware-limited to 65535, so the NVIDIA-style one-CTA-per-
#     element-block launch (2.36M CTAs for (8192, 64, 512, 64)) fails to launch.
#   - the dispatch floor is ~52 ns/CTA and one CTA per SP wins: >48 programs
#     only time-slice (131072 programs -> 6.6 ms vs 48 -> 52 us on a 16M-elem
#     workload). Cap the launch and let each program stride over the flattened
#     segment space; each (load+store) iteration serializes at ~0.4-0.6 us, so
#     the fast path also widens the per-iteration ops (multi-row flat tiles)
#     to amortize that latency.
_GCU_FLAT_GRID_TARGET = 48
_GCU_GRID_TARGET = 192

# Row-block fast path (2026-11-11 perf round, probe-verified on GCU400):
# the flat path's div/mod-derived pitched stores cap the big shapes at
# ~0.5 TB/s (fp16/bf16 t=8192: 2236 us = 0.61x torch). The row kernel moves
# 8-byte packed lanes instead: flat contiguous loads from both segments plus
# shift-derived computed stores (lane + (lane >> log2(AL)) * BL), which the
# backend compiles correctly at any group count (unlike div/mod addressing,
# see compiler_issues/vllm_concat_mla_q.md). Measured (framework bench core
# shapes, do_bench median w1000/r100): bf16 t=8192 2236 -> 444 us (2.58 TB/s
# = 92% of the flat-copy floor), fp32 2410 -> 839 us (2.87 TB/s, at floor),
# bf16 t=128 37.0 -> 12.1 us, fp32 t=128 39.6 -> 18.4 us. Broadcast rope is
# served by a scalar token base when RB divides the head count and by a
# shift-derived vector token ((r0 + row) >> log2(H), pow2 heads only)
# otherwise. The tier ladder (RB/num_warps by lane count) and the
# 16384-lane tile cap come from the probe sweep; they are derived at launch
# instead of autotuned because the tile cap is a compile-domain limit, not a
# perf preference.
_GCU_ROW_TILE_LANES = 16384
_GCU_ROW_SMALL_LANES = 1 << 15
_GCU_ROW_MEDIUM_LANES = 1 << 18
_GCU_ROW_LARGE_LANES = 1 << 24


# flagtree (triton 3.6 GCU backend, 2026-11-07) miscompiles vector accesses
# whose piecewise-contiguous address pattern spans >= 8 segment groups per
# vector op: from the 8th group on, lanes silently lose the constant part of
# the address (verified: rope rows land in the nope region with correct
# values). 4 groups are proven correct, so both flat-loop block sizes are
# capped at prev_pow2(4 * segment_width); see
# compiler_issues/vllm_concat_mla_q.md. The cap also keeps masked tail lanes
# within ~4 rows of the tensor, the overshoot class the generic kernel
# already proved safe on GCU (the hardware evaluates masked addresses).
_GCU_MAX_GROUPS = 4

# Local configs for the strided path: the enflame tune_configs.yaml has no
# concat_mla_q entry and the NVIDIA fallback (single BLOCK <= 1024,
# num_stages=1) undersizes the copy loop on GCU.
_GCU_TUNED_CONFIGS = [
    triton.Config({"BLOCK_N": 128, "BLOCK_R": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_N": 256, "BLOCK_R": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 512, "BLOCK_R": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 512, "BLOCK_R": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 1024, "BLOCK_R": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_N": 1024, "BLOCK_R": 256}, num_warps=8, num_stages=2),
]


@triton.jit
def _concat_mla_q_flat_kernel(
    nope,
    rope,
    out,
    SIZE,
    HEADS: tl.constexpr,
    NOPE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BROADCAST: tl.constexpr,
    WIDE: tl.constexpr,
):
    # Contiguous fast path. Both segments are walked in their own flattened
    # element space so a single wide load/store op covers several rows per
    # iteration (the GCU serializes whole load+store iterations, so widening
    # the op - not the program count - is the only lever). The nope source
    # is the flat index itself; the rope source either is the flat index
    # (per-head rope) or repeats one rope row per token (broadcast). All
    # piecewise addresses stay 1-D and within the _GCU_MAX_GROUPS budget.
    ROPE: tl.constexpr = WIDTH - NOPE
    rows = SIZE // WIDTH
    nope_elems = rows * NOPE
    for off0 in range(
        tl.program_id(0) * BLOCK_N, nope_elems, tl.num_programs(0) * BLOCK_N
    ):
        off = off0 + tl.arange(0, BLOCK_N)
        if WIDE:
            off = off.to(tl.int64)
        row = off // NOPE
        col = off - row * NOPE
        mask = off < nope_elems
        val = tl.load(nope + off, mask=mask, other=0)
        tl.store(out + row * WIDTH + col, val, mask=mask)
    rope_elems = rows * ROPE
    for off0 in range(
        tl.program_id(0) * BLOCK_R, rope_elems, tl.num_programs(0) * BLOCK_R
    ):
        off = off0 + tl.arange(0, BLOCK_R)
        if WIDE:
            off = off.to(tl.int64)
        row = off // ROPE
        col = off - row * ROPE
        mask = off < rope_elems
        if BROADCAST:
            src = (row // HEADS) * ROPE + col
        else:
            src = off
        val = tl.load(rope + src, mask=mask, other=0)
        tl.store(out + row * WIDTH + (NOPE + col), val, mask=mask)


@triton.jit
def _concat_mla_q_row_kernel(
    nope,
    rope,
    out,
    num_rows,
    HEADS: tl.constexpr,
    AL: tl.constexpr,
    BL: tl.constexpr,
    RB: tl.constexpr,
    BROADCAST: tl.constexpr,
    TOKEN_ALIGNED: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    WIDE: tl.constexpr,
):
    # Row-block fast path over packed 8-byte lanes. Each iteration moves RB
    # whole output rows: one flat contiguous load per source segment and one
    # shift-derived computed store per segment (no div/mod, no piecewise
    # segment decomposition - the two miscompile classes of the flat kernel
    # are absent by construction). The broadcast rope row index is a scalar
    # when RB divides HEADS and a shift-derived vector otherwise (pow2 heads
    # only; the host guarantees one of the two arms).
    pa = nope.to(tl.pointer_type(tl.int64))
    po = out.to(tl.pointer_type(tl.int64))
    TOT: tl.constexpr = AL + BL
    LAL: tl.constexpr = (AL - 1).bit_length()
    LBL: tl.constexpr = (BL - 1).bit_length() if BL > 0 else 0
    LHEADS: tl.constexpr = (HEADS - 1).bit_length()
    pid = tl.program_id(0)
    step = tl.num_programs(0) * RB
    lane_a = tl.arange(0, RB * AL)
    if HAS_ROPE:
        pb = rope.to(tl.pointer_type(tl.int64))
        lane_b = tl.arange(0, RB * BL)
    for r0 in range(pid * RB, num_rows, step):
        if WIDE:
            r0l = r0.to(tl.int64)
            base_a = r0l * AL
            base_o = r0l * TOT
            base_b = r0l * BL
        else:
            r0l = r0
            base_a = r0 * AL
            base_o = r0 * TOT
            base_b = r0 * BL
        val_a = tl.load(pa + base_a + lane_a)
        tl.store(po + base_o + lane_a + (lane_a >> LAL) * BL, val_a)
        if HAS_ROPE:
            if BROADCAST:
                if TOKEN_ALIGNED:
                    tok = r0l // HEADS
                    val_b = tl.load(pb + tok * BL + (lane_b & (BL - 1)))
                else:
                    tok = (r0l + (lane_b >> LBL)) >> LHEADS
                    val_b = tl.load(pb + tok * BL + (lane_b & (BL - 1)))
            else:
                val_b = tl.load(pb + base_b + lane_b)
            tl.store(
                po + base_o + AL + (lane_b >> LBL) * TOT + (lane_b & (BL - 1)),
                val_b,
            )


@libentry()
@libtuner(
    configs=_GCU_TUNED_CONFIGS,
    key=["SIZE", "WIDTH"],
    strategy=["log", "log"],
)
@triton.jit
def _concat_mla_q_kernel(
    nope,
    rope,
    out,
    SIZE,
    HEADS: tl.constexpr,
    NOPE: tl.constexpr,
    WIDTH: tl.constexpr,
    ns0,
    ns1,
    ns2,
    rs0,
    rs1,
    rs2,
    os0,
    os1,
    os2,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
    WIDE: tl.constexpr,
):
    # Strided fallback: one tile covers one row segment, so every address is
    # a scalar base plus one affine vector (immune to the piecewise-group
    # compiler bug above and fully stride-aware). Only reached for
    # non-contiguous tensors; the grid-stride loops keep the launch small.
    NUM_N_BLOCKS: tl.constexpr = (NOPE + BLOCK_N - 1) // BLOCK_N
    NUM_R_BLOCKS: tl.constexpr = (WIDTH - NOPE + BLOCK_R - 1) // BLOCK_R
    rows = SIZE // WIDTH

    nope_tiles = rows * NUM_N_BLOCKS
    for tile in range(tl.program_id(0), nope_tiles, tl.num_programs(0)):
        row = tile // NUM_N_BLOCKS
        token = row // HEADS
        head = row % HEADS
        col = tl.max_contiguous(
            tl.multiple_of(
                (tile % NUM_N_BLOCKS) * BLOCK_N + tl.arange(0, BLOCK_N), BLOCK_N
            ),
            BLOCK_N,
        )
        if WIDE:
            token = token.to(tl.int64)
            head = head.to(tl.int64)
            col = col.to(tl.int64)
        mask = col < NOPE
        val = tl.load(
            nope + token * ns0 + head * ns1 + col * ns2,
            mask=mask,
            other=0,
        )
        tl.store(out + token * os0 + head * os1 + col * os2, val, mask=mask)

    rope_tiles = rows * NUM_R_BLOCKS
    for tile in range(tl.program_id(0), rope_tiles, tl.num_programs(0)):
        row = tile // NUM_R_BLOCKS
        token = row // HEADS
        head = row % HEADS
        rcol = tl.max_contiguous(
            tl.multiple_of(
                (tile % NUM_R_BLOCKS) * BLOCK_R + tl.arange(0, BLOCK_R), BLOCK_R
            ),
            BLOCK_R,
        )
        scol = tl.max_contiguous(rcol + NOPE, BLOCK_R)
        if WIDE:
            token = token.to(tl.int64)
            head = head.to(tl.int64)
            rcol = rcol.to(tl.int64)
            scol = scol.to(tl.int64)
        mask = rcol < (WIDTH - NOPE)
        val = tl.load(
            rope + token * rs0 + head * rs1 + rcol * rs2,
            mask=mask,
            other=0,
        )
        tl.store(out + token * os0 + head * os1 + scol * os2, val, mask=mask)


def _validate_inputs(ql_nope, q_pe, q_out):
    if any(t.ndim != 3 for t in (ql_nope, q_pe, q_out)):
        raise ValueError("concat_mla_q requires rank-3 tensors")
    tokens, heads, nope = ql_nope.shape
    rope = q_pe.shape[-1]
    if heads <= 0 or nope <= 0:
        raise ValueError("concat_mla_q requires positive head count and nope width")
    if q_pe.shape[:2] not in ((tokens, heads), (tokens, 1)):
        raise ValueError("concat_mla_q rope token/head dimensions do not match")
    if q_out.shape != (tokens, heads, nope + rope):
        raise ValueError("concat_mla_q output shape does not match")
    if not (ql_nope.dtype == q_pe.dtype == q_out.dtype):
        raise ValueError("concat_mla_q tensors must have the same dtype")
    if not (ql_nope.device == q_pe.device == q_out.device):
        raise ValueError("concat_mla_q tensors must be on the same device")
    if q_out.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError("concat_mla_q supports FP16/BF16/FP32")
    if any(stride < 0 for t in (ql_nope, q_pe, q_out) for stride in t.stride()):
        raise NotImplementedError("concat_mla_q does not support negative strides")
    # Conservative metadata-only proof of disjoint output elements. Gaps in
    # storage are allowed, including padded token/head/feature strides.
    span = 1
    for stride, size in sorted(zip(q_out.stride(), q_out.shape)):
        if size <= 1:
            continue
        if stride < span:
            raise ValueError("concat_mla_q output elements overlap")
        span += (size - 1) * stride
    if q_out.numel():
        storage = q_out.untyped_storage().data_ptr()
        if any(storage == t.untyped_storage().data_ptr() for t in (ql_nope, q_pe)):
            raise ValueError("concat_mla_q output must not alias its inputs")
    if q_out.device.type != runtime.device.name:
        raise NotImplementedError("concat_mla_q requires CUDA tensors")


def _max_offset(tensor):
    return sum(
        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
    )


def _prev_pow2(value):
    return 1 << (max(int(value), 1).bit_length() - 1)


def _is_pow2(value):
    return value > 0 and value & (value - 1) == 0


def _rowblock_params(tokens, heads, nope, rope, esz, broadcast):
    """Pick the row-block lane widths and tile size, or None when unusable."""
    epl = 8 // esz
    al = nope // epl
    bl = rope // epl
    if not _is_pow2(al) or (bl and not _is_pow2(bl)):
        return None
    cap = min(128, _prev_pow2(_GCU_ROW_TILE_LANES // max(al, bl)))
    n_lanes = tokens * heads * (al + bl)
    if n_lanes > _GCU_ROW_LARGE_LANES:
        want = 128
    elif n_lanes > _GCU_ROW_MEDIUM_LANES:
        want = 64
    else:
        want = 32
    want = min(want, cap)
    num_rows = tokens * heads
    rb = want
    while rb >= 8:
        # Blocks must cover whole rows (num_rows % rb == 0); broadcast rope
        # additionally needs a block inside one token (heads % rb == 0) or
        # pow2 heads for the shift-derived vector-token arm.
        if num_rows % rb == 0 and (not broadcast or heads % rb == 0 or _is_pow2(heads)):
            return al, bl, rb
        rb //= 2
    return None


def concat_mla_q(
    ql_nope: torch.Tensor, q_pe: torch.Tensor, q_out: torch.Tensor
) -> None:
    """Write ``[ql_nope, q_pe]`` into q_out, broadcasting one rope head.

    Inputs and output may be strided; zero-width rope is supported. Output
    aliasing is rejected before any launch. This is an inference-only data copy,
    with no dtype conversion.
    """
    _validate_inputs(ql_nope, q_pe, q_out)
    if not q_out.numel():
        return
    tokens, heads, nope = ql_nope.shape
    rope = q_pe.shape[-1]
    width = q_out.shape[-1]
    # int32 addressing covers every realistic MLA shape; promote the whole
    # index math only when a byte-offset could exceed the signed 32-bit range.
    wide = max(_max_offset(t) for t in (ql_nope, q_pe, q_out)) >= 2**31

    # The flat path divides the flat index by the segment widths (and by the
    # head count for broadcast rope) inside the vector domain. Non-power-of-
    # two divisors miscompile on the current flagtree build like the >= 8
    # piecewise groups do, so it is restricted to power-of-two segments and
    # head counts; the strided kernel below handles every other layout.
    flat_ok = (
        ql_nope.is_contiguous()
        and q_pe.is_contiguous()
        and q_out.is_contiguous()
        and _is_pow2(nope)
        and (rope == 0 or _is_pow2(rope))
        and _is_pow2(heads)
    )

    with torch.cuda.device(ql_nope.device):
        esz = q_out.element_size()
        epl = 8 // esz
        broadcast = q_pe.shape[1] == 1
        # Launch-floor tier: below the row-block threshold the flat kernel
        # wins, so skip the row-parameter work entirely (its host cost is
        # visible on the smallest shapes).
        row = None
        if not (flat_ok and tokens * heads * width // epl <= _GCU_ROW_SMALL_LANES):
            if (
                esz in (2, 4)
                and nope % epl == 0
                and rope % epl == 0
                and ql_nope.is_contiguous()
                and q_pe.is_contiguous()
                and q_out.is_contiguous()
                and all(t.data_ptr() % 8 == 0 for t in (ql_nope, q_pe, q_out))
            ):
                row = _rowblock_params(tokens, heads, nope, rope, esz, broadcast)
        if row is not None:
            al, bl, rb = row
            num_rows = tokens * heads
            n_lanes = num_rows * (al + bl)
            # Row-block fast path. Block sizes are derived from the
            # measured size ladder, not autotuned: the 16384-lane tile
            # cap is a compile-domain limit, not a perf preference.
            blocks = triton.cdiv(num_rows, rb)
            _concat_mla_q_row_kernel[(min(blocks, _GCU_FLAT_GRID_TARGET),)](
                ql_nope,
                q_pe,
                q_out,
                num_rows,
                HEADS=heads,
                AL=al,
                BL=bl,
                RB=rb,
                BROADCAST=broadcast,
                TOKEN_ALIGNED=(not broadcast) or heads % rb == 0,
                HAS_ROPE=bl > 0,
                WIDE=n_lanes >= (1 << 28),
                num_warps=2 if n_lanes > _GCU_ROW_MEDIUM_LANES else 4,
            )
            return

        if flat_ok:
            # Flat fast path. Block sizes are derived from the segment widths,
            # not autotuned: the piecewise-group compiler bug makes any block
            # beyond prev_pow2(4 * segment_width) silently wrong, so the safe
            # search space cannot be expressed as a static config list.
            block_n = _prev_pow2(_GCU_MAX_GROUPS * nope)
            block_r = _prev_pow2(_GCU_MAX_GROUPS * rope)
            nope_tiles = triton.cdiv(tokens * heads * nope, block_n)
            rope_tiles = triton.cdiv(tokens * heads * rope, block_r)
            grid = (min(nope_tiles + rope_tiles, _GCU_FLAT_GRID_TARGET),)
            num_warps = 8 if max(block_n, block_r) >= 4096 else 4
            _concat_mla_q_flat_kernel[grid](
                ql_nope,
                q_pe,
                q_out,
                tokens * heads * width,
                heads,
                nope,
                width,
                BLOCK_N=block_n,
                BLOCK_R=block_r,
                BROADCAST=q_pe.shape[1] == 1,
                WIDE=wide,
                num_warps=num_warps,
                num_stages=2,
            )
            return

        def grid(meta):
            tiles_per_row = triton.cdiv(nope, meta["BLOCK_N"]) + triton.cdiv(
                rope, meta["BLOCK_R"]
            )
            tiles = tokens * heads * tiles_per_row
            return (min(tiles, _GCU_GRID_TARGET),)

        _concat_mla_q_kernel[grid](
            ql_nope,
            q_pe,
            q_out,
            tokens * heads * width,
            heads,
            nope,
            width,
            *ql_nope.stride(),
            q_pe.stride(0),
            0 if q_pe.shape[1] == 1 else q_pe.stride(1),
            q_pe.stride(2),
            *q_out.stride(),
            WIDE=wide,
        )


# ---------------------------------------------------------------------------
# torch_gcu aten::cat miscompile workaround
# (compiler_issues/torch_gcu_cat_strided_expand.md): a last-dim torch.cat
# silently reads strided inputs as if they were contiguous whenever the
# tensor list also contains an expanded (0-stride) input - verified wrong on
# the (7, 4, 31, 17) MLA layout (1339/1344 elements with a strided+expanded
# right input, 867/1344 with an expanded-but-contiguous right input), in the
# positional, keyword-dim, out= and tuple forms; three-input lists, other
# concat dims, all-contiguous lists and CPU tensors are unaffected. The
# concat_mla_q tests build exactly that reference (strided q-nope view plus a
# broadcast q-pe expand), and so can real vLLM callers, so torch.cat gets a
# narrow wrapper on the enflame backend: only the verified-broken input
# class (last-dim concat with at least one last-dim-strided tensor and at
# least one genuinely expanded tensor) is routed through contiguous copies
# before delegating to the original implementation. Every other call is
# passed through untouched.
def _torch_cat_miscompile_scope(tensors, dim):
    if not isinstance(tensors, (list, tuple)) or len(tensors) < 2:
        return False
    first = tensors[0]
    if not isinstance(first, torch.Tensor) or first.ndim == 0:
        return False
    dim = dim + first.ndim if dim < 0 else dim
    if dim != first.ndim - 1:
        return False
    last_strided = False
    expanded = False
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != first.ndim:
            return False
        if tensor.device.type != runtime.device.name:
            return False
        strides = tensor.stride()
        if tensor.shape[-1] > 1 and strides[-1] != 1:
            last_strided = True
        if any(size > 1 and stride == 0 for size, stride in zip(tensor.shape, strides)):
            expanded = True
    return last_strided and expanded


def _install_torch_cat_workaround():
    original = torch.cat

    def gcu_cat(tensors, dim=0, *, out=None):
        try:
            if _torch_cat_miscompile_scope(tensors, dim):
                tensors = [t if t.is_contiguous() else t.contiguous() for t in tensors]
        except Exception:  # noqa: BLE001 - any surprise: original behavior
            pass
        if out is None:
            return original(tensors, dim)
        return original(tensors, dim, out=out)

    gcu_cat._flaggems_vllm_gcu_cat = original
    torch.cat = gcu_cat


if runtime.device.vendor_name == "enflame" and not getattr(
    torch.cat, "_flaggems_vllm_gcu_cat", False
):
    try:
        _install_torch_cat_workaround()
    except Exception:  # noqa: BLE001 - never block import on the workaround
        pass


__all__ = ["concat_mla_q"]
