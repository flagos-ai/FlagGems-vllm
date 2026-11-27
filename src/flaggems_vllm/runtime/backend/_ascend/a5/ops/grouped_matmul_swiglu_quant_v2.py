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

"""GroupedMatmulSwigluQuantV2 (MX-FP8) — Triton on Ascend 950DT (A5).

Replaces ``torch_npu.npu_grouped_matmul_swiglu_quant_v2`` for the MXFP8 path
(``dequant_mode=2, quant_mode=2``), as called by vllm-ascend
``w8a8_mxfp8.py:apply_gmm1_act_quant``.

One kernel fuses three steps (keeping the [M, N] fp32 intermediate off GM):

    1. grouped matmul + MX dequant     -> fp32 [M, N]
    2. SwiGLU: split N in half          -> fp32 [M, N/2]
         S = silu(C[:, :N/2]) * C[:, N/2:]
    3. MX dynamic quant (block 32)     -> fp8_e4m3 [M, N/2] + e8m0 scale

Numerical contract (all verified byte-exact against vendor):

  * SwiGLU: alpha=1, bias=0, no clamp (V2 does not expose these parameters).
  * Quant uses scale_alg=0 (OCP), not cuBLAS.
  * Before quantising, fp32 SwiGLU output is rounded to BF16 (round-to-even).
  * Output scale dtype is ``float8_e8m0fnu``, shape [M, nblk/2, 2], physically
    a contiguous [M, nblk].  Compare with vendor via ``.view(torch.uint8)``
    (direct ``.int()`` raises aclnnInplaceCopy error 561103).
  * Rows outside grouplist are not written (matches vendor behaviour).

Constraints that must not be changed without re-profiling:

  * ``//`` and ``%`` must only appear on bare ``tl.arange`` vectors, never on
    ``tile_n * BLOCK_N + arange``; the compiler cannot recognise strided access
    otherwise and falls back to element-wise gather (17-37x slower).
  * Scale loads must not carry a mask; A-side is padded to M+BLOCK_M rows so
    the inner loop is mask-free, and out-of-bounds accumulator rows are dropped
    by the store mask.
  * B-side scale physical layout is ``[E, K/64, N, 2]`` truly interleaved; the
    index formula must be ``(kb0//2)*(N*2) + n*2 + (j//2)*(N*2) + j%2``.
  * ``tl.dot_scaled`` rhs_scale shape is ``[BLOCK_N, BLOCK_K/32]`` (N first).
  * ``block_k`` must be 64; larger values degrade access patterns on SB.
  * Both fp32 accumulators for gate and up halves live in L0C simultaneously,
    so BLOCK_N is capped at half the groupmatmul limit; see ``_pick_block_n``.
"""

import torch
import triton
import triton.language as tl

MX_BLOCK = 32  # elements per e8m0 scale, hardware-fixed
E4M3_EMAX = 8  # float8_e4m3fn max normal exponent
L0C_BYTES = 256 * 1024

__all__ = [
    "grouped_matmul_swiglu_quant_v2",
    "grouped_matmul_swiglu_quant_v2_ref",
    "MX_BLOCK",
]


@triton.jit
def _tile_order(idx, num_m_tiles, num_n_tiles, THRESHOLD: tl.constexpr):
    """Map linear block index to (tile_m, tile_n).

    When num_m_tiles < THRESHOLD, use N-major order so consecutive m-tiles
    share a B panel in L2; otherwise fall back to row-major.
    """
    if num_m_tiles < THRESHOLD:
        return idx % num_m_tiles, idx // num_m_tiles
    return idx // num_n_tiles, idx % num_n_tiles


@triton.jit
def _gmm_swiglu_mxquant_kernel(
    A,  # fp8_e4m3fn [M+BLOCK_M, K]      padded, so loads are mask-free
    SA,  # uint8      [M+BLOCK_M, K/32]   A-side e8m0, also padded
    B,  # fp8_e4m3fn [E, K, N]           ND or FRACTAL_NZ
    SB,  # uint8      [E, K/64, N, 2]     truly interleaved
    Y,  # fp8_e4m3fn [M, N/2]
    YS,  # uint8      [M, NSP]            flat e8m0
    group_list,  # int32/64   [E]
    GROUPS: tl.constexpr,
    NH: tl.constexpr,  # N/2, output width
    K: tl.constexpr,
    N: tl.constexpr,  # 2*NH, weight width
    NSP: tl.constexpr,  # YS row stride = NH/32
    GROUP_LIST_TYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,  # output tile width (not weight tile width)
    BLOCK_K: tl.constexpr,
    TILE_THRESHOLD: tl.constexpr,
    MX: tl.constexpr,
    WEIGHT_NZ: tl.constexpr,
    CEIL: tl.constexpr,  # quant scale algorithm: 0=OCP (vendor), 1=cuBLAS
    B_CACHE: tl.constexpr,
    SB_CACHE: tl.constexpr,
    EPI: tl.constexpr,  # 2=full; 1=SwiGLU only; 0=neither (timing only)
    DUMP: tl.constexpr,  # diagnostic: dump both accumulators as bf16
    DG,  # DUMP gate half bf16 [M, NH]
    DU,  # DUMP up   half bf16 [M, NH]
):
    total_cores = tl.num_programs(0)
    core_idx = tl.program_id(0)
    num_n_tiles: tl.constexpr = NH // BLOCK_N  # wrapper guarantees divisible
    n_kblk: tl.constexpr = K // MX
    sb_stride_e: tl.constexpr = n_kblk * N
    NB: tl.constexpr = BLOCK_N // MX  # quant groups per tile

    # Bit-field constants for bf16-domain quantisation.
    # Written as literals because jit cannot read non-constexpr module globals.
    SGN: tl.constexpr = -32768  # sign bit, int16
    MAG: tl.constexpr = 32767  # magnitude bits
    MANT: tl.constexpr = 127  # bf16 mantissa mask (7 bits)
    FM: tl.constexpr = 96  # fmax(1.75) mantissa = 127-31

    last_count = 0
    group_start = 0
    group_end = 0

    for group_idx in range(GROUPS):
        gv = tl.load(group_list + group_idx).to(tl.int32)
        group_m = gv - group_end if GROUP_LIST_TYPE == 0 else gv
        group_end = group_start + group_m
        num_m_tiles = tl.cdiv(group_m, BLOCK_M)

        current_count = last_count + num_m_tiles * num_n_tiles
        current_block = core_idx if core_idx >= last_count else core_idx + total_cores

        for block_idx in range(current_block, current_count, total_cores):
            tile_m, tile_n = _tile_order(
                block_idx - last_count, num_m_tiles, num_n_tiles, TILE_THRESHOLD
            )
            # int64: element count of B (E*K*N) can exceed 2^31
            offs_m = (group_start + tile_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(
                tl.int64
            )
            offs_n0 = tl.arange(0, BLOCK_N).to(tl.int64)  # bare arange
            offs_n = (tile_n * BLOCK_N + offs_n0).to(tl.int64)
            offs_kk = tl.arange(0, BLOCK_K).to(tl.int64)
            offs_sk = tl.arange(0, BLOCK_K // MX).to(tl.int64)

            mask_m = offs_m < group_end
            g64 = group_idx.to(tl.int64)

            a_base = A + offs_m[:, None] * K
            sa_base = SA + offs_m[:, None] * n_kblk + offs_sk[None, :]

            # B base pointers for gate half [0, NH) and up half [NH, N).
            # NH is a multiple of BLOCK_N (wrapper guarantee), so the up half
            # is a plain column offset; this holds for NZ too (NH%32==0).
            if WEIGHT_NZ:
                # FRACTAL_NZ = [E, N/32, K/16, 16, 32], flattened:
                #   offs = e*K*N + (n//32)*(K*32) + k*32 + (n%32)
                # //32 and %32 act only on the bare arange offs_n0 -- the tile
                # offset is a scalar hoisted out.  This is the 37x difference.
                nz_g: tl.constexpr = BLOCK_N // 32
                nvec = (offs_n0 // 32) * (K * 32) + (offs_n0 % 32)
                bg_base = (
                    B
                    + g64 * (K * N)
                    + tile_n.to(tl.int64) * (nz_g * K * 32)
                    + nvec[None, :]
                )
                bu_base = bg_base + (NH // 32) * (K * 32)
            else:
                bg_base = B + g64 * (K * N) + offs_n[None, :]
                bu_base = bg_base + NH
            sb_base = SB + g64 * sb_stride_e

            acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for k0 in range(0, n_kblk, BLOCK_K // MX):
                kb0 = k0.to(tl.int64)
                koff = kb0 * MX
                # A loads are mask-free: wrapper pads A to M+BLOCK_M rows;
                # out-of-bounds accumulator rows are discarded by the store mask.
                a = tl.load(a_base + koff + offs_kk[None, :])
                sa = tl.load(sa_base + kb0)

                # B-side scale: physical flat = (b//2)*(N*2) + n*2 + (b%2),
                # b = kb0 + j.  Since BLOCK_K is a multiple of 64, kb0 is even:
                #   (kb0+j)//2 = kb0//2 + j//2,  (kb0+j)%2 = j%2
                # so // and % act only on the bare arange j.
                sb_off = (
                    (kb0 // 2) * (N * 2)
                    + (offs_sk[None, :] // 2) * (N * 2)
                    + (offs_sk[None, :] % 2)
                )
                sbg = tl.load(
                    sb_base + sb_off + offs_n[:, None] * 2, cache_modifier=SB_CACHE
                )
                sbu = tl.load(
                    sb_base + sb_off + (offs_n[:, None] + NH) * 2,
                    cache_modifier=SB_CACHE,
                )

                if WEIGHT_NZ:
                    kstep = (koff + offs_kk)[:, None] * 32
                else:
                    kstep = (koff + offs_kk)[:, None] * N
                bg = tl.load(bg_base + kstep, cache_modifier=B_CACHE)
                bu = tl.load(bu_base + kstep, cache_modifier=B_CACHE)

                # rhs_scale shape is [BLOCK_N, BLOCK_K/32] (N first)
                acc_g = tl.dot_scaled(a, sa, "e4m3", bg, sbg, "e4m3", acc_g)
                acc_u = tl.dot_scaled(a, sa, "e4m3", bu, sbu, "e4m3", acc_u)

            if DUMP:
                # Diagnostic: store both gmm halves as bf16 to separate gate/up
                # errors from epilogue errors.  fp32 store would trigger
                # bishengir-compile SIGABRT (LLVM ERROR: Cannot calculate static
                # size from mmadL1 shape); bf16 is sufficient for structure checks.
                d_off = offs_m[:, None] * NH + offs_n[None, :]
                tl.store(DG + d_off, acc_g.to(tl.bfloat16), mask=mask_m[:, None])
                tl.store(DU + d_off, acc_u.to(tl.bfloat16), mask=mask_m[:, None])

            if EPI == 0:
                # Timing probe: skip SwiGLU and quant; store gate+up as fp8.
                # Same data movement as the real path, so the diff is epilogue cost.
                y_off0 = offs_m[:, None] * NH + offs_n[None, :]
                tl.store(
                    Y + y_off0,
                    (acc_g + acc_u).to(tl.float8e4nv),
                    mask=mask_m[:, None],
                )
                s_off0 = (
                    offs_m[:, None] * NSP
                    + tile_n.to(tl.int64) * NB
                    + tl.arange(0, NB)[None, :]
                )
                tl.store(
                    YS + s_off0,
                    tl.zeros((BLOCK_M, NB), dtype=tl.uint8),
                    mask=mask_m[:, None],
                )
            elif EPI == 3:
                # Split path (default): SwiGLU only; store bf16; quant is done
                # by _mxquant_kernel on the dense [M, NH] buffer.
                # bf16 store here is exactly the rounding-to-bf16 step that the
                # spec requires before quantisation -- not an extra precision loss.
                s3 = (acc_g * tl.sigmoid(acc_g)) * acc_u
                tl.store(
                    Y + offs_m[:, None] * NH + offs_n[None, :],
                    s3.to(tl.bfloat16),
                    mask=mask_m[:, None],
                )
            elif EPI == 1:
                # Timing probe: SwiGLU only, no quant reduce chain.
                s1 = (acc_g * tl.sigmoid(acc_g)) * acc_u
                y_off1 = offs_m[:, None] * NH + offs_n[None, :]
                tl.store(Y + y_off1, s1.to(tl.float8e4nv), mask=mask_m[:, None])
                s_off1 = (
                    offs_m[:, None] * NSP
                    + tile_n.to(tl.int64) * NB
                    + tl.arange(0, NB)[None, :]
                )
                tl.store(
                    YS + s_off1,
                    tl.zeros((BLOCK_M, NB), dtype=tl.uint8),
                    mask=mask_m[:, None],
                )
            else:
                # ---------------- SwiGLU ----------------
                # V2 MX path: alpha=1, bias=0, no clamp. silu(gate) * up.
                s = (acc_g * tl.sigmoid(acc_g)) * acc_u

                # ---------------- MX dynamic quant ----------------
                # Round to bf16 first (spec requirement), then operate in bf16
                # bit-domain.
                xt = tl.reshape(s.to(tl.bfloat16), (BLOCK_M * NB, MX))
                xi = xt.to(tl.int16, bitcast=True)
                # Clear sign bit: int magnitude is monotone with |x|, so the
                # bit-domain max is the max-magnitude element's bit pattern.
                mg = (xi & MAG).to(tl.int16)
                mm = tl.max(mg, 1)
                # Convert through fp32 to read exponent: denormals have no
                # implicit leading 1, so the exponent field is not the exponent.
                af = mm.to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)
                ab = af.to(tl.int32, bitcast=True)
                ef = (ab >> 23) & 0xFF
                eb = ef - 8  # float8_e4m3fn emax; literal matches E4M3_EMAX
                if CEIL:
                    eb = eb + ((ab & 0x7FFFFF) > 0x600000).to(tl.int32)
                nan = ef == 0xFF
                if CEIL:
                    flush = nan
                else:
                    flush = (eb <= 0) | nan
                ebs = tl.where(nan, 255, tl.maximum(eb, 0))
                # rc = 2^(127-ebs), assembled in bit-domain; minimum keeps
                # inf/nan paths finite so 0*rc stays 0 not nan.
                rc = ((254 - tl.minimum(ebs, 254)) << 23).to(tl.float32, bitcast=True)
                if CEIL:
                    cap = tl.where(nan, 0, MAG)
                else:
                    # cap = 1.75 * 2^e_amax: use amax's exponent bits with
                    # fmax mantissa; ebs clamped to 0 is still valid here.
                    cap = tl.where(flush, 0, (mm.to(tl.int32) & ~MANT) | FM)
                sgn = (xi & SGN).to(tl.int16)
                # Saturate in int magnitude domain to avoid fp32 min/max
                # pressure on UB.
                src = (tl.minimum(mg, cap[:, None].to(tl.int16)) | sgn).to(tl.int16)
                q = src.to(tl.bfloat16, bitcast=True).to(tl.float32) * rc[:, None]
                q8 = q.to(tl.float8e4nv)

                y_off = offs_m[:, None] * NH + offs_n[None, :]
                tl.store(
                    Y + y_off,
                    tl.reshape(q8, (BLOCK_M, BLOCK_N)),
                    mask=mask_m[:, None],
                )
                s_off = (
                    offs_m[:, None] * NSP
                    + tile_n.to(tl.int64) * NB
                    + tl.arange(0, NB)[None, :]
                )
                tl.store(
                    YS + s_off,
                    tl.reshape(ebs.to(tl.uint8), (BLOCK_M, NB)),
                    mask=mask_m[:, None],
                )

        # last_count must stay in [0, total_cores): the `core_idx >= last_count`
        # guard and the `core_idx + total_cores` wrap-around both depend on it.
        last_count = current_count % total_cores
        group_start = group_end


@triton.jit
def _mxquant_kernel(
    X,
    Y,
    S,
    M,
    NH: tl.constexpr,
    NSP: tl.constexpr,
    ROWS: tl.constexpr,
    NB: tl.constexpr,
    PROGS: tl.constexpr,
):
    """Second kernel for split path: bf16 [M, NH] -> fp8_e4m3 + e8m0 scale.

    Operates on a dense [M, NH] buffer where M is the true row count, so it
    does not pay the BLOCK_M/group_m padding overhead of the fused epilogue.
    One program consumes ROWS full rows (W = NB*32).  Columns are not tiled:
    partial-row tiles would produce strided DMA, which is significantly slower.
    """
    pid = tl.program_id(0)
    MXC: tl.constexpr = 32  # literal: jit cannot read module-level globals
    W: tl.constexpr = NB * MXC
    SGN: tl.constexpr = -32768
    MAG: tl.constexpr = 32767
    MANT: tl.constexpr = 127
    FM: tl.constexpr = 96
    w = tl.arange(0, W)
    bl = tl.arange(0, NB)
    for r0 in range(pid * ROWS, M, PROGS * ROWS):
        r = r0 + tl.arange(0, ROWS)
        rm = (r < M)[:, None]
        xb = X + r[:, None] * NH
        yb = Y + r[:, None] * NH
        sb = S + r[:, None] * NSP
        for k0 in range(0, NH, W):
            flat = tl.load(xb + (k0 + w)[None, :], mask=rm, other=0)
            xt = tl.reshape(flat, (ROWS * NB, MXC))
            xi = xt.to(tl.int16, bitcast=True)
            mg = (xi & MAG).to(tl.int16)
            mm = tl.max(mg, 1)
            af = mm.to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)
            ab = af.to(tl.int32, bitcast=True)
            ef = (ab >> 23) & 0xFF
            eb = ef - 8
            nan = ef == 0xFF
            flush = (eb <= 0) | nan
            ebs = tl.where(nan, 255, tl.maximum(eb, 0))
            rc = ((254 - tl.minimum(ebs, 254)) << 23).to(tl.float32, bitcast=True)
            cap = tl.where(flush, 0, (mm.to(tl.int32) & ~MANT) | FM)
            sgn = (xi & SGN).to(tl.int16)
            src = (tl.minimum(mg, cap[:, None].to(tl.int16)) | sgn).to(tl.int16)
            q = src.to(tl.bfloat16, bitcast=True).to(tl.float32) * rc[:, None]
            tl.store(
                yb + (k0 + w)[None, :],
                tl.reshape(q.to(tl.float8e4nv), (ROWS, W)),
                mask=rm,
            )
            tl.store(
                sb + (k0 // MXC + bl)[None, :],
                tl.reshape(ebs.to(tl.uint8), (ROWS, NB)),
                mask=rm,
            )


QUANT_TARGET_PROGS = 16
QUANT_MAX_ROWS = 8
QUANT_MAX_TILE = 32 * 1024  # upper bound on ROWS*NH to avoid UB overflow


def _pick_quant_rows(M: int, cores: int, NH: int) -> int:
    """Choose ROWS for _mxquant_kernel (rows consumed per program).

    Target progs=16; derive ROWS as ceil(M / 16) rounded down to a power of
    two, capped by QUANT_MAX_ROWS and the ROWS*NH tile budget.
    """
    rows = 1
    while (
        rows * 2 <= QUANT_MAX_ROWS
        and rows * 2 * NH <= QUANT_MAX_TILE
        and -(-M // (rows * 2)) >= QUANT_TARGET_PROGS
    ):
        rows *= 2
    return rows


def _cube_core_num(device=None) -> int:
    props = torch.npu.get_device_properties(device)
    for attr in ("num_ai_cores", "multi_processor_count"):
        v = getattr(props, attr, None)
        if v:
            return int(v)
    return 32


def _is_nz(t: torch.Tensor) -> bool:
    """NZ format is only visible in the ACL format descriptor, not in shape."""
    try:
        import torch_npu

        return torch_npu.get_npu_format(t) == 29
    except Exception:
        return False


def _pick_block_n(NH: int, prefer: int | None = None, block_m: int = 16) -> int:
    """Return the largest BLOCK_N that divides NH and fits two fp32 accumulators
    in the 256 KB L0C budget (2 * BLOCK_M * BLOCK_N * 4 <= L0C_BYTES).
    """
    cap = L0C_BYTES // (2 * block_m * 4)
    cap = min(cap, 1024)
    if prefer is not None:
        if NH % prefer:
            raise ValueError(f"block_n={prefer} does not divide NH={NH}")
        return prefer
    bn = 1
    while bn * 2 <= cap and NH % (bn * 2) == 0:
        bn *= 2
    if bn < MX_BLOCK:
        raise ValueError(f"NH={NH} is too narrow or not a multiple of 32")
    return bn


def _check(x, weight, weight_scale, x_scale, group_list):
    if x.dtype != torch.float8_e4m3fn:
        raise ValueError(f"x must be fp8_e4m3fn, got {x.dtype}")
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"weight must be fp8_e4m3fn, got {weight.dtype}")
    if x.ndim != 2 or weight.ndim != 3:
        raise ValueError(f"x must be 2-D, weight 3-D; got {x.ndim}/{weight.ndim}")
    M, K = x.shape
    E, Kw, N = weight.shape
    if Kw != K:
        raise ValueError(f"K mismatch: x {K} vs weight {Kw}")
    if K % 64:
        raise ValueError(f"K must be a multiple of 64 (e8m0 pair storage), got {K}")
    if N % 2:
        raise ValueError(f"N must be even (SwiGLU split), got {N}")
    if N % 64:
        raise ValueError(f"N must be 64-aligned for fp8 inputs, got {N}")
    if group_list.numel() != E:
        raise ValueError(f"group_list length {group_list.numel()} != E {E}")
    nkb = K // MX_BLOCK
    if x_scale.numel() < M * nkb:
        raise ValueError(f"x_scale numel {x_scale.numel()} < M*K/32 {M * nkb}")
    if weight_scale.numel() < E * nkb * N:
        raise ValueError(
            f"weight_scale numel {weight_scale.numel()} < E*K/32*N {E * nkb * N}"
        )
    return M, K, E, N


def grouped_matmul_swiglu_quant_v2(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int = 0,
    *,
    block_m: int = 16,
    block_n: int | None = None,
    block_k: int = 64,
    tile_threshold: int = 8,
    num_cores: int | None = None,
    weight_nz: bool | None = None,
    scale_alg: int = 0,
    b_cache: str = "",
    sb_cache: str = "",
    dump: bool = False,
    allow_slow_bk: bool = False,
    epi: int = 2,
    split: bool = True,
    quant_rows: int | None = None,
):
    """MX-FP8 GroupedMatmul + SwiGLU + MX quantisation.

    Args:
        x:            fp8_e4m3fn [M, K]
        weight:       fp8_e4m3fn [E, K, N]  (N = 2*NH, gate|up concatenated)
        weight_scale: uint8, numel E*(K/32)*N; accepts 4-D [E,K/64,N,2] or flat
        x_scale:      uint8, numel M*(K/32);   accepts 3-D [M,K/64,2] or flat
        group_list:   int32/64 [E]
        group_list_type: 0=cumsum (vllm default), 1=count
        block_k:      must be 64; larger values degrade B-side scale access.
        scale_alg:    0=OCP (matches vendor), 1=cuBLAS ceiling.
        split:        True (default) uses two kernels: kernel 1 does gmm+SwiGLU
                      and stores bf16; kernel 2 quantises the dense [M, NH]
                      buffer.  Both outputs are bit-exact with split=False.
        dump:         If True, return (y, y_scale, dg, du) where dg/du are the
                      bf16 gate/up accumulator dumps for diagnostics.

    Returns:
        (y, y_scale):
          y:       fp8_e4m3fn [M, NH]
          y_scale: uint8      [M, nblk/2, 2],  nblk = NH/32
        Rows outside grouplist are not written (matches vendor behaviour).
        y_scale dtype is uint8 here; vendor returns float8_e8m0fnu -- compare
        via vendor_scale.view(torch.uint8).
    """
    if group_list_type not in (0, 1):
        raise ValueError(f"group_list_type must be 0 or 1, got {group_list_type}")
    if scale_alg not in (0, 1):
        raise ValueError(f"scale_alg must be 0 (OCP) or 1 (cuBLAS), got {scale_alg}")
    M, K, E, N = _check(x, weight, weight_scale, x_scale, group_list)
    NH = N // 2
    BN = _pick_block_n(NH, block_n, block_m)
    if block_k % 64:
        raise ValueError(f"block_k must be a multiple of 64, got {block_k}")
    if block_k != 64 and not allow_slow_bk:
        raise ValueError(
            f"block_k={block_k} is order-of-magnitude slower than 64 on this path. "
            f"Pass allow_slow_bk=True to override."
        )
    if K % block_k:
        raise ValueError(f"block_k={block_k} does not divide K={K}")
    nz = _is_nz(weight) if weight_nz is None else weight_nz

    nkb = K // MX_BLOCK
    # Pad A and A-scale to M+BLOCK_M rows so inner-loop loads need no mask.
    # Out-of-bounds accumulator rows are dropped by the store mask.
    xp = torch.empty((M + block_m, K), device=x.device, dtype=x.dtype)
    xp[:M] = x
    sap = torch.empty((M + block_m, nkb), device=x.device, dtype=torch.uint8)
    sap[:M] = (
        x_scale.reshape(M, nkb).view(torch.uint8)
        if x_scale.dtype != torch.uint8
        else x_scale.reshape(M, nkb)
    )

    sb = (
        weight_scale.view(torch.uint8)
        if weight_scale.dtype != torch.uint8
        else weight_scale
    )
    nblk = NH // MX_BLOCK
    if split and epi == 2:
        epi = 3  # split path: kernel 1
    y = torch.empty((M, NH), device=x.device, dtype=torch.float8_e4m3fn)
    ys = torch.empty((M, nblk), device=x.device, dtype=torch.uint8)
    if epi == 3:
        # Intermediate bf16 buffer for split path.
        mid = torch.empty((M, NH), device=x.device, dtype=torch.bfloat16)
    else:
        mid = y  # placeholder; not accessed on non-split path

    if dump:
        dg = torch.zeros((M, NH), device=x.device, dtype=torch.bfloat16)
        du = torch.zeros((M, NH), device=x.device, dtype=torch.bfloat16)
    else:
        dg = du = y  # placeholders; not accessed when DUMP=False

    cores = num_cores or _cube_core_num(x.device)
    _gmm_swiglu_mxquant_kernel[(cores,)](
        xp,
        sap,
        weight,
        sb,
        mid if epi == 3 else y,
        ys,
        group_list,
        GROUPS=E,
        NH=NH,
        K=K,
        N=N,
        NSP=nblk,
        GROUP_LIST_TYPE=group_list_type,
        BLOCK_M=block_m,
        BLOCK_N=BN,
        BLOCK_K=block_k,
        TILE_THRESHOLD=tile_threshold,
        MX=MX_BLOCK,
        WEIGHT_NZ=nz,
        CEIL=(scale_alg == 1),
        B_CACHE=b_cache,
        SB_CACHE=sb_cache,
        EPI=int(epi),
        DUMP=dump,
        DG=dg,
        DU=du,
    )
    if epi == 3:
        rows = quant_rows or _pick_quant_rows(M, cores, NH)
        progs = min(cores * 2, max(1, -(-M // rows)))
        _mxquant_kernel[(progs,)](
            mid, y, ys, M, NH=NH, NSP=nblk, ROWS=rows, NB=nblk, PROGS=progs
        )
    if dump:
        return y, ys.reshape(M, nblk // 2, 2), dg, du
    return y, ys.reshape(M, nblk // 2, 2)


def grouped_matmul_swiglu_quant_v2_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scale: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int = 0,
    scale_alg: int = 0,
):
    """Pure-torch fp32 reference implementation for accuracy comparison.

    Scale index layout follows the same conventions as the kernel:
    x_scale is flat; weight_scale has a truly-interleaved trailing-2 dimension.
    """
    M, K = x.shape
    E, _, N = weight.shape
    NH = N // 2
    nkb = K // MX_BLOCK

    sa = x_scale.reshape(M, nkb)
    sa = (sa.view(torch.uint8) if sa.dtype != torch.uint8 else sa).to(torch.int32)
    sa_f = torch.where(
        sa == 0,
        torch.zeros_like(sa, dtype=torch.float32),
        torch.pow(2.0, sa.float() - 127.0),
    )

    sb = (
        weight_scale.view(torch.uint8)
        if weight_scale.dtype != torch.uint8
        else weight_scale
    )
    sb = sb.reshape(E, nkb // 2, N, 2).to(torch.int32)
    sb = sb.permute(0, 1, 3, 2).reshape(E, nkb, N)  # (b//2, b%2) -> b
    sb_f = torch.where(
        sb == 0,
        torch.zeros_like(sb, dtype=torch.float32),
        torch.pow(2.0, sb.float() - 127.0),
    )

    xf = x.to(torch.float32) * sa_f.repeat_interleave(MX_BLOCK, dim=1)
    wf = weight.to(torch.float32) * sb_f.repeat_interleave(MX_BLOCK, dim=1)

    gl = group_list.tolist()
    c = torch.zeros((M, N), device=x.device, dtype=torch.float32)
    start = 0
    for e in range(E):
        end = gl[e] if group_list_type == 0 else start + gl[e]
        if end > start:
            c[start:end] = xf[start:end] @ wf[e]
        start = end
    valid = start

    act, gate = c[:, :NH], c[:, NH:]
    s = torch.nn.functional.silu(act) * gate

    # MX quant: round to bf16 first, then block-quantise in groups of 32
    sbf = s.to(torch.bfloat16).to(torch.float32)
    blk = sbf.reshape(M, NH // MX_BLOCK, MX_BLOCK)
    amax = blk.abs().amax(dim=-1)
    bits = amax.view(torch.int32)
    eb = ((bits >> 23) & 0xFF) - E4M3_EMAX
    if scale_alg == 1:
        eb = eb + ((bits & 0x7FFFFF) > 0x600000).to(torch.int32)
    ebs = torch.clamp(eb, min=0)
    ebs = torch.where(amax == 0, torch.zeros_like(ebs), ebs)
    rc = torch.pow(2.0, -(ebs.float() - 127.0)).unsqueeze(-1)
    q = torch.clamp(blk * rc, -448.0, 448.0)
    yq = q.reshape(M, NH).to(torch.float8_e4m3fn)
    return yq, ebs.to(torch.uint8), valid
