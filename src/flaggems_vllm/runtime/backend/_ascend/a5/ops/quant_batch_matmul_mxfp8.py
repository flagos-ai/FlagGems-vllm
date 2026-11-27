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

"""MX-FP8 quant matmul for vllm-ascend's ``AscendW8A8MXFP8LinearMethod.apply``.

This is the A5 Triton replacement for the following vendor call
(``quantization/methods/w8a8/w8a8_mxfp8.py:143``)::

    torch_npu.npu_quant_matmul(
        quantized_x, layer.weight, layer.weight_scale,
        scale_dtype=torch_npu.float8_e8m0fnu,
        pertoken_scale=pertoken_scale,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        bias=bias, output_dtype=output_dtype, group_sizes=[1, 1, 32])

which profiles as ``aclnnQuantMatmulV5_QuantBatchMatmulV3_QuantBatchMatmulV3``
for ND weight and ``aclnnQuantMatmulWeightNz_...`` for NZ weight.  Both layouts
are implemented here and NZ is detected from the ACL descriptor, so callers
pass no extra argument.

x and weight are fp8_e4m3fn; every 32 K elements share one e8m0 scale carried
in uint8, where v encodes 2^(v-127) and v=0 marks a zero block::

    out[m, n] = sum_k  x[m,k] * sa[m, k//32] * w[k,n] * sb[k//32, n]

The layouts are the ones ``process_weights_after_loading`` produces, not the
checkpoint ones: weight is ``[K,N]``, weight_scale is physically ``[K/64,N,2]``
whose trailing 2 is truly interleaved (``flat = (b//2)*N*2 + n*2 + (b%2)`` with
``b = k//32``), while pertoken_scale ``[M,K/64,2]`` is simply flattened.  NZ
weight is physically ``[N/32,K/16,16,32]``, i.e.
``offs = (n//32)*(K*32) + k*32 + (n%32)``, and its logical shape is unchanged.

This operator is the E=1, group_list-free case of ``grouped_matmul_mxfp8``, so
the constraints measured there hold (do not relax without re-validating):
  * ``tl.dot_scaled`` lowers to a single hardware MX-MAC instruction, which is
    what makes the output bit-exact with CANN while split-K is off.
  * The cost of a mask depends on whether its bound is a compile-time constant.
    A runtime-range mask loses the regular access pattern (13x slower scale
    loads) and, on top of the split-K partials store, aborts
    bishengir-compile; a static mask is free.  M is therefore passed as a
    constexpr while ``M < BLOCK_M``, which removes the row-padding buffers and
    their pre-processing kernels entirely.
  * Index arithmetic on runtime values degrades access patterns, clamping
    included (``tl.minimum(offs_m, M-1)`` measured +67%), so ``//`` and ``%``
    stay on bare ``tl.arange`` and tile offsets enter as scalar terms.
  * BLOCK_K must be a multiple of 64 so the interleaved B-scale index folds
    into compile-time constants, and BLOCK_N must divide N so scale loads need
    no mask.

M is tiny in deployment (decode runs M=1..8 against a cube M granularity of
16), so the work is the weight stream rather than the MACs.  Parallelism comes
from split-K and not from a finer BLOCK_N: with SPLIT_K=1 the bottleneck is the
serial depth of the k-loop, not the program count (one program measured as fast
as four at N=512), and a finer BLOCK_N shortens the DMA bursts instead.
split-K reorders the K accumulation, so cases that enable it are not guaranteed
to stay bit-exact with CANN; measured deviations are far inside the BF16 output
tolerance.

Forward inference only: 2-D x, fp8_e4m3fn operands, BF16/FP16 output.  int8 /
int4 / FP4, 3-6 dim x and the asymmetric ``offset`` parameter are rejected
rather than silently falling back to torch.
"""

import torch
import torch_npu  # noqa: F401
import triton  # noqa: F401
import triton.language as tl

__all__ = ["quant_batch_matmul_mxfp8"]

MX_BLOCK = 32
ACL_FORMAT_FRACTAL_NZ = 29

# The 950 unified buffer is 256KB; the backend's own graph budget caps usable
# space at 80% of it (``graph_ub_budget_bytes_for_arch``).  Same accounting is
# used here to decide whether B can be double-buffered.
_UB_BUDGET_BYTES = 256 * 1024 * 80 // 100


def _cube_core_num(device=None) -> int:
    """dot runs on the Cube unit, so parallelism counts Cube cores (32 on 950DT)."""
    return int(torch.npu.get_device_properties(device).cube_core_num)


@triton.jit
def _e8m0_to_f32(v):
    """Move the biased exponent straight into FP32; zero denotes a zero block.

    Bit-exact with 2^(v-127), while ``tl.exp2(v-127)`` rounds at v=254.
    """
    return (v.to(tl.uint32) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _qbmm_kernel(
    A,  # fp8_e4m3fn [M_pad, K]
    SA,  # uint8      [M_pad, K/32]   A-side e8m0
    B,  # fp8_e4m3fn [K, N]          ND, or NZ with the same logical shape
    SB,  # uint8      [K/32, N] logically, interleaved as (b//2, n, b%2)
    C,  # [M, N] BF16/FP16 when SPLIT_K==1, else [SPLIT_K, M, N] FP32
    BIAS,  # fp32 [N] or a dummy pointer
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # multiple of 64, see the module docstring
    SPLIT_K: tl.constexpr,
    NUM_M_TILES: tl.constexpr,  # cdiv(M, BLOCK_M), computed on the host
    PSTRIDE: tl.constexpr,  # partials split stride, unused when SPLIT_K==1
    MX: tl.constexpr,
    WEIGHT_NZ: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    A_BCAST: tl.constexpr,  # M==1: A/SA rows are all row 0, no padding needed
    SB_NK: tl.constexpr,  # B scale pre-transposed to a plain [N, K/32]
    MASK_M: tl.constexpr,  # store needs a runtime row mask
    M_CE: tl.constexpr,  # >0: M is a compile-time constant, masks are static
):
    pid = tl.program_id(0)
    num_n_tiles: tl.constexpr = N // BLOCK_N  # the wrapper guarantees N % BN == 0
    n_kblk: tl.constexpr = K // MX  # total scale blocks
    chunk: tl.constexpr = n_kblk // SPLIT_K  # scale blocks per split

    # pid layout, matching the wrapper's grid:
    #   pid = sk * (num_m_tiles * num_n_tiles) + tile_m * num_n_tiles + tile_n
    # tile_n varies fastest so neighbouring programs read neighbouring N blocks.
    # num_m_tiles arrives as a constexpr instead of tl.cdiv(M, BLOCK_M): for M>1
    # that M is a runtime i32, which would make the whole tile_m -> offs_m ->
    # a_base address chain dynamic and abort bishengir-compile with
    # "Cannot calculate static size from mmadL1 shape".
    num_m_tiles: tl.constexpr = NUM_M_TILES
    tile_n = pid % num_n_tiles
    # With a single m-tile, fold tile_m to a compile-time 0 explicitly; the
    # compiler does not fold `% 1`, and the resulting runtime scalar combines
    # with the split-K k offset into two dynamic terms on A's address, which is
    # exactly what the mmadL1 descriptor cannot size.
    M_SINGLE: tl.constexpr = num_m_tiles == 1
    if M_SINGLE:
        tile_m = 0
        sk = pid // num_n_tiles
    else:
        tile_m = (pid // num_n_tiles) % num_m_tiles
        sk = pid // (num_n_tiles * num_m_tiles)

    offs_m0 = tl.arange(0, BLOCK_M).to(tl.int64)  # bare arange
    offs_m = tile_m * BLOCK_M + offs_m0  # only for the store address and mask
    offs_n0 = tl.arange(0, BLOCK_N).to(tl.int64)  # bare arange, needed by NZ
    offs_n = tile_n.to(tl.int64) * BLOCK_N + offs_n0
    offs_kk = tl.arange(0, BLOCK_K).to(tl.int64)
    offs_sk = tl.arange(0, BLOCK_K // MX).to(tl.int64)

    # A and A-scale loads carry no runtime mask.  Staying in bounds relies on
    # one of three things:
    #   A_BCAST (M==1): the row index is identically 0, so all BLOCK_M rows read
    #     row 0.  The extra rows compute duplicates whose accumulator rows the
    #     store mask drops, and no padding buffer is needed at all.  `offs_m0*0`
    #     is constant-folded, so this is not arithmetic on a runtime value.
    #   M_CE > 0 (1 < M < BLOCK_M): M is a compile-time constant, so the mask
    #     below is static and free, which also removes the padding buffers (two
    #     slice-assignment memcpys plus a zeros_like, 2.44us on the M3 shape).
    #   otherwise: the wrapper pads A/SA to cdiv(M,BLOCK_M)*BLOCK_M rows.
    # The tile_m offset is added as a scalar term rather than written as
    # `offs_m[:, None] * K`: a runtime offset vector plus the split-K k offset
    # gives A's address two dynamic terms, which aborts the compiler.
    ROW_MASK: tl.constexpr = M_CE > 0 and not A_BCAST
    row0 = offs_m0 * 0 if A_BCAST else offs_m0
    if A_BCAST or M_SINGLE:
        m_off_a = 0  # compile-time constant: k stays A's only dynamic term
        m_off_sa = 0
    else:
        m_off_a = tile_m.to(tl.int64) * (BLOCK_M * K)
        m_off_sa = tile_m.to(tl.int64) * (BLOCK_M * n_kblk)
    a_base = A + m_off_a + row0[:, None] * K
    sa_base = SA + m_off_sa + row0[:, None] * n_kblk

    if WEIGHT_NZ:
        # FRACTAL_NZ [N/32, K/16, 16, 32] flattens to
        #   offs = (n//32)*(K*32) + k*32 + (n%32)
        # because (k//16)*16 + k%16 == k.  // and % stay on a bare arange and
        # the tile offset is a scalar term.
        nz_g: tl.constexpr = BLOCK_N // 32
        nvec = (offs_n0 // 32) * (K * 32) + (offs_n0 % 32)
        b_base = B + tile_n.to(tl.int64) * (nz_g * K * 32) + nvec[None, :]
    else:
        b_base = B + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    kb_start = sk.to(tl.int64) * chunk
    for k0 in tl.range(0, chunk, BLOCK_K // MX):
        kb0 = kb_start + k0.to(tl.int64)  # first scale block of this step (even)
        koff = kb0 * MX  # first K element of this step

        if ROW_MASK:
            # Static mask (M_CE is a constexpr).  `other=` must match the
            # pointee dtype, since a bare 0 would be int32.  Padded rows read as
            # zero and their accumulator rows are dropped anyway.
            rm = offs_m0 < M_CE
            a = tl.load(
                a_base + koff + offs_kk[None, :],
                mask=rm[:, None],
                other=tl.zeros((1, 1), dtype=A.dtype.element_ty),
            )
            sa = tl.load(
                sa_base + kb0 + offs_sk[None, :],
                mask=rm[:, None],
                other=tl.zeros((1, 1), dtype=SA.dtype.element_ty),
            )
        else:
            a = tl.load(a_base + koff + offs_kk[None, :])
            sa = tl.load(sa_base + kb0 + offs_sk[None, :])
        if WEIGHT_NZ:
            b = tl.load(b_base + (koff + offs_kk)[:, None] * 32)
        else:
            b = tl.load(b_base + (koff + offs_kk)[:, None] * N)
        # The B scale tile must be [BLOCK_N, BLOCK_K/32], N first, as required
        # by semantic.py's verify_scaled_shape.
        if SB_NK:
            # Pre-transposed to a plain [N, K/32]: the index becomes a 2-D
            # stride and the // and % disappear.  In deployment this would be a
            # one-off weight-loading transform, like the NZ format cast.
            sb = tl.load(SB + offs_n[:, None] * n_kblk + kb0 + offs_sk[None, :])
        else:
            # Physical layout is flat = (b//2)*N*2 + n*2 + (b%2).  BLOCK_K being
            # a multiple of 64 makes kb0 even, so (kb0+j)//2 = kb0//2 + j//2 and
            # (kb0+j)%2 = j%2: // and % stay on the bare arange j.
            sb_off = (
                (kb0 // 2) * (N * 2)
                + offs_n[:, None] * 2
                + (offs_sk[None, :] // 2) * (N * 2)
                + (offs_sk[None, :] % 2)
            )
            sb = tl.load(SB + sb_off)
        acc = tl.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", acc)

    # The row mask `offs_m < M` is a runtime range unless M was specialised to a
    # constexpr.  A runtime-range mask on the split-K partials store is the
    # actual trigger of the mmadL1 abort, measured as:
    #   M=1  (constexpr) + SK=4 -> compiles
    #   M=8  (runtime)   + SK=1 -> compiles
    #   M=8  (runtime)   + SK=2 -> abort
    #   M=16 (runtime)   + SK=4 -> abort
    # So SPLIT_K>1 stores without a mask: partials belong to this operator, are
    # allocated with padded rows, and combine only reads the first M rows.
    if SPLIT_K == 1:
        if HAS_BIAS:
            acc += tl.load(BIAS + offs_n)[None, :]
        if M_CE > 0:
            tl.store(
                C + offs_m[:, None] * N + offs_n[None, :],
                acc.to(C.dtype.element_ty),
                mask=(offs_m < M_CE)[:, None],
            )
        elif MASK_M:
            tl.store(
                C + offs_m[:, None] * N + offs_n[None, :],
                acc.to(C.dtype.element_ty),
                mask=(offs_m < M)[:, None],
            )
        else:
            tl.store(
                C + offs_m[:, None] * N + offs_n[None, :],
                acc.to(C.dtype.element_ty),
            )
    else:
        # FP32 partial sums; combine adds them and applies the bias.  PSTRIDE is
        # a host-computed constexpr rather than `M.to(tl.int64)*N` because
        # Triton specialises a scalar argument equal to 1 into a Python int,
        # which has no `.to`.
        pbase = C + sk.to(tl.int64) * PSTRIDE + offs_m[:, None] * N + offs_n[None, :]
        if M_CE > 0:
            # A static mask is free, so partials can be the compact [SK, M, N]
            # instead of padded rows: 2MB -> 128KB on the GLM shape, and those
            # are bytes the main kernel really writes out.
            tl.store(pbase, acc, mask=(offs_m < M_CE)[:, None])
        elif MASK_M:
            tl.store(pbase, acc, mask=(offs_m < M)[:, None])
        else:
            tl.store(pbase, acc)


@triton.jit
def _combine_kernel(
    P,  # fp32 [SPLIT_K, M_rows, N]
    C,  # bf16/fp16 [M, N]
    BIAS,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PSTRIDE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """Add the split-K partial sums; one program per row and N block.

    The partials are loaded as one [SPLIT_K, BLOCK_N] tile and reduced along
    axis 0, not in a loop over SPLIT_K: SPLIT_K and PSTRIDE are constexprs, so
    this is one regular 2-D strided read, whereas the loop form emits an
    independent transfer per iteration and makes the combine time grow linearly
    with SPLIT_K.
    """
    m = tl.program_id(0)
    tile_n = tl.program_id(1)
    offs_n = (tile_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    offs_sk = tl.arange(0, SPLIT_K).to(tl.int64)
    base = m.to(tl.int64) * N + offs_n
    part = tl.load(P + offs_sk[:, None] * PSTRIDE + base[None, :])
    acc = tl.sum(part, axis=0)
    if HAS_BIAS:
        acc += tl.load(BIAS + offs_n)
    tl.store(C + base, acc.to(C.dtype.element_ty))


def _is_nz(t: torch.Tensor) -> bool:
    """Ask the ACL format: an NZ cast leaves torch's logical shape and numel
    unchanged, so the shape has no discriminating power."""
    return torch_npu.get_npu_format(t) == ACL_FORMAT_FRACTAL_NZ


def _divisors_pow2(n: int, lo: int, hi: int):
    """Powers of two in [lo, hi] that divide n, largest first."""
    out = []
    c = hi
    while c >= lo:
        if n % c == 0:
            out.append(c)
        c //= 2
    return out


def _pick_tiling(
    M: int,
    K: int,
    N: int,
    cores: int,
    block_n: int | None,
    split_k: int | None,
    block_k: int | None,
):
    """Choose (BLOCK_N, SPLIT_K, BLOCK_K).

    Hard constraints: ``N % BLOCK_N == 0`` (B-scale loads must stay mask-free),
    ``BLOCK_K % 64 == 0`` and dividing each split's K length (the interleaved
    B-scale index needs an even kb0), and ``(K/32) % SPLIT_K == 0`` (no K-side
    tail block).

    The rule is to take the largest power-of-two divisor of N up to 1024 and
    buy parallelism with split-K instead of a finer BLOCK_N.  Measured: at
    equal program counts the larger BLOCK_N always won, because BLOCK_N decides
    how much a single DMA burst covers and how much MAC work one B tile feeds.
    Replacing split-K with a finer BLOCK_N at identical byte counts was 2.0-2.9x
    slower.  BLOCK_N=2048 is excluded for a physical reason: its B tile alone is
    128KB and the FP32 accumulator another 128KB, which overflows the UB and
    raises a runtime aicore exception.

    SPLIT_K has two measured bounds, both with a physical meaning: programs must
    not exceed the core count (a second wave usually leaves cores idle, 48
    programs measured slower than 24 on the M3 shape), and each split keeps at
    least 4 BLOCK_K steps (below that the loop body is too short to hide its own
    start-up cost).

    The branch that steps BLOCK_N down is a fallback; no deployment or edge
    shape reaches it.  It only matters when K is too short for split-K and N is
    small as well, which is extrapolation rather than measurement.
    """
    num_m_tiles = (M + 15) // 16  # BLOCK_M is fixed at 16
    cands_n = _divisors_pow2(N, 32, 1024)
    if not cands_n:
        raise ValueError(f"N={N} needs a power-of-two divisor in [32,1024]")

    n_kblk = K // MX_BLOCK
    max_sk = max(1, n_kblk // 8)  # >= 4 BLOCK_K steps, i.e. 8 scale blocks

    # Compiler limit: a runtime-range row mask combined with the split-K
    # partials store aborts bishengir-compile ("Cannot calculate static size
    # from mmadL1 shape").  With a single m-tile, M is passed as a constexpr and
    # the mask is static, so split-K is usable; with more m-tiles the output
    # spans several tiles, the mask must be dynamic, and split-K is disabled.
    # The criterion is the m-tile count, not M > 1.
    sk_ok = num_m_tiles == 1

    def pick_sk(tiles: int) -> int:
        """Largest power of two that pushes the program count up to ``cores``."""
        if not sk_ok:
            return 1
        sk = 1
        while sk * 2 <= max_sk and tiles * sk * 2 <= cores and n_kblk % (sk * 2) == 0:
            sk *= 2
        return sk

    if block_n is not None:
        bn = block_n
    else:
        bn = cands_n[0]
        while num_m_tiles * (N // bn) * pick_sk(num_m_tiles * (N // bn)) < max(
            1, cores // 4
        ):
            nxt = [c for c in cands_n if c < bn]
            if not nxt:
                break
            bn = nxt[0]

    if split_k is not None:
        sk = split_k
        if sk > 1 and not sk_ok:
            raise ValueError(
                f"M={M} spans {num_m_tiles} m-tiles, so split_k={sk} is not "
                f"available: the output needs a runtime row mask, and a runtime "
                f"mask on the split-K partials store aborts bishengir-compile "
                f"with 'Cannot calculate static size from mmadL1 shape'"
            )
    else:
        sk = pick_sk(num_m_tiles * (N // bn))

    if n_kblk % sk != 0:
        raise ValueError(f"split_k={sk} must divide K/32={n_kblk} (no K tail block)")
    chunk = n_kblk // sk  # scale blocks per split

    if block_k is not None:
        bk = block_k
    else:
        # Multiple of 64 whose BK/32 divides chunk.  BK=64 is the measured sweet
        # spot under the interleaved scale layout; BK=128 was 5x slower.
        if chunk % 2 != 0:
            raise ValueError(
                f"chunk={chunk} is odd, so block_k cannot be a multiple of 64; "
                f"reduce split_k (currently {sk})"
            )
        bk = 64
    if bk % (2 * MX_BLOCK) != 0:
        raise ValueError(
            f"block_k={bk} must be a multiple of {2 * MX_BLOCK}: the B-scale "
            f"index folds its //2 into a constant only when kb0 is even"
        )
    if chunk % (bk // MX_BLOCK) != 0:
        raise ValueError(
            f"block_k={bk} must divide each split's K length {chunk * MX_BLOCK}"
        )
    return bn, sk, bk


def quant_batch_matmul_mxfp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    pertoken_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
    *,
    block_m: int = 16,
    block_n: int | None = None,
    block_k: int | None = None,
    split_k: int | None = None,
    num_cores: int | None = None,
    weight_nz: bool | None = None,
    a_bcast: bool | None = None,
    sb_nk: bool = False,
    m_const: int | None = None,
    multibuffer: bool | None = None,
) -> torch.Tensor:
    """Return the BF16/FP16 MX-FP8 matmul output ``[M, N]``.

    Arguments mirror the MX-FP8 form of ``torch_npu.npu_quant_matmul``:

      x               fp8_e4m3fn ``[M, K]``
      weight          fp8_e4m3fn ``[K, N]``, ND or FRACTAL_NZ (auto-detected)
      weight_scale    e8m0 bytes, ``(K/32)*N`` elements, 3-D ``[K/64,N,2]`` or
                      any equivalent flattening
      pertoken_scale  e8m0 bytes, ``M*(K/32)`` elements, 3-D ``[M,K/64,2]`` or
                      any equivalent flattening
      bias            optional fp32 ``[N]``
      output_dtype    torch.bfloat16 or torch.float16

    ``weight_nz=None`` detects the layout with ``torch_npu.get_npu_format``, so
    model code needs no change when vllm-ascend switches between ND and NZ.
    The remaining keyword arguments preserve the source implementation's tuning
    interface; production defaults are BLOCK_M=16, BLOCK_N and SPLIT_K from
    ``_pick_tiling``, BLOCK_K=64, and B double-buffering whenever the UB budget
    allows it.  No NVIDIA autotune is attached because this is A5-only.
    """
    if output_dtype not in (torch.bfloat16, torch.float16):
        raise NotImplementedError(f"output_dtype {output_dtype} is not supported")
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D [M,K], got {tuple(x.shape)}")
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D [K,N], got {tuple(weight.shape)}")
    M, K = x.shape
    Kw, N = weight.shape
    if K != Kw:
        raise ValueError(f"K mismatch: x {K} vs weight {Kw}")
    if K % MX_BLOCK != 0:
        raise ValueError(f"K={K} must be a multiple of {MX_BLOCK}")
    if x.dtype != torch.float8_e4m3fn or weight.dtype != torch.float8_e4m3fn:
        raise TypeError(f"x/weight must be fp8_e4m3fn, got {x.dtype}/{weight.dtype}")
    if x.device.type != "npu" or any(
        t.device != x.device for t in (weight, weight_scale, pertoken_scale)
    ):
        raise ValueError("all tensors must be on the same NPU")

    for name, t in (("weight_scale", weight_scale), ("pertoken_scale", pertoken_scale)):
        if t.dtype not in (torch.uint8, torch.float8_e8m0fnu):
            raise TypeError(f"{name} must contain e8m0 bytes, got {t.dtype}")
    n_kblk = K // MX_BLOCK
    if weight_scale.numel() != n_kblk * N:
        raise ValueError(
            f"weight_scale needs (K/32)*N = {n_kblk * N} elements, "
            f"got {weight_scale.numel()}"
        )
    if pertoken_scale.numel() < M * n_kblk:
        raise ValueError(
            f"pertoken_scale needs M*(K/32) = {M * n_kblk} elements, "
            f"got {pertoken_scale.numel()}"
        )
    for name, t in (
        ("x", x),
        ("weight_scale", weight_scale),
        ("pertoken_scale", pertoken_scale),
    ):
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous (indexed by flat offset)")

    # A byte view keeps the e8m0 representation without a compute cast.
    if weight_scale.dtype != torch.uint8:
        weight_scale = weight_scale.view(torch.uint8)
    if pertoken_scale.dtype != torch.uint8:
        pertoken_scale = pertoken_scale.view(torch.uint8)

    if weight_nz is None:
        weight_nz = _is_nz(weight)
    if weight_nz:
        if K % 16 != 0 or N % 32 != 0:
            raise ValueError(
                f"NZ requires K%16==0 and N%32==0 (16x32 fractal), got K={K} N={N}"
            )
    elif not weight.is_contiguous():
        # An NZ tensor's is_contiguous() reflects the logical shape only, so
        # this check applies to ND weight.
        raise ValueError("ND weight must be contiguous (indexed by flat offset)")

    if block_m != 16:
        raise ValueError("block_m is fixed at 16 (cube M granularity; use masks below)")
    cores = num_cores if num_cores is not None else _cube_core_num(x.device.index)
    if cores <= 0:
        raise ValueError(f"num_cores must be positive, got {cores}")
    bn, sk, bk = _pick_tiling(M, K, N, cores, block_n, split_k, block_k)
    if N % bn != 0:
        raise ValueError(f"block_n={bn} must divide N={N} for mask-free scale loads")

    num_m_tiles = (M + block_m - 1) // block_m
    m_pad = num_m_tiles * block_m

    # ---- B double-buffering ----
    # triton-ascend disables auto multi-buffer on 950 by default
    # (``NPUOptions.multibuffer = None``, on the grounds that the extra ring
    # buffer may overflow the UB), but a transfer-bound kernel is exactly what
    # needs it: measured -1.3% to -10.2% on the main kernel across the four
    # deployment shapes, with unchanged accuracy.  It cannot be enabled by trial
    # and error, because a UB overflow is a runtime aicore exception rather than
    # a compile failure, so the budget is checked on the host:
    #   B tile (double-buffered)  BLOCK_K * BLOCK_N * 1B * 2
    #   accumulator               BLOCK_M * BLOCK_N * 4B
    #   A tile                    BLOCK_M * BLOCK_K * 1B
    #   both scales               (BLOCK_M + BLOCK_N) * (BLOCK_K/32)
    # BLOCK_N=1024/BLOCK_K=64 needs 195KB <= 204.8KB; BLOCK_N=2048's B tile
    # alone is 128KB and double-buffering it overflows, which is also why 1024
    # is the BLOCK_N ceiling.
    launch_opts: dict[str, object] = {}
    if multibuffer is None:
        need = (
            bk * bn * 2
            + block_m * bn * 4
            + block_m * bk
            + (block_m + bn) * (bk // MX_BLOCK)
        )
        if need <= _UB_BUDGET_BYTES:
            launch_opts["multibuffer"] = True
    else:
        launch_opts["multibuffer"] = multibuffer

    # A/SA loads stay mask-free (a runtime mask on the scale load measured 13x
    # slower), so rows come from one of three paths: A_BCAST for M==1, a static
    # mask for 1 < M < BLOCK_M, and wrapper padding otherwise.
    if a_bcast is None:
        a_bcast = M == 1
    elif a_bcast and M != 1:
        raise ValueError(f"a_bcast only holds for M==1 (row index 0), got M={M}")

    # m_const=None picks the static-mask path automatically while M < BLOCK_M;
    # 0 disables it; any other value must equal M.  M >= BLOCK_M keeps padding,
    # because prefill's M varies widely and specialising would recompile often.
    # M==1 is included even though A_BCAST already needs no mask: a free static
    # mask lets the partials be the compact [SK, M, N].
    m_ce = M if (m_const is None and M < block_m) else (m_const or 0)
    if m_ce and not 0 < m_ce < block_m:
        raise ValueError(f"m_const only supports 0 < M < {block_m}, got {m_ce}")
    if m_ce and m_ce != M:
        raise ValueError(f"m_const={m_ce} must equal the actual M={M}")
    p_rows = M if m_ce else m_pad  # partials row count

    if a_bcast or m_pad == M or m_ce:
        ain, sa = x, pertoken_scale
    else:
        ain = torch.empty((m_pad, K), device=x.device, dtype=x.dtype)
        ain[:M] = x
        # Zero the whole scale buffer instead of filling only its tail: a tail
        # slice assignment dispatches to AiCPU's aclnnInplaceFillScalar and
        # costs far more than this small allocation.
        sa = torch.zeros(m_pad * n_kblk, device=x.device, dtype=torch.uint8)
        sa[: M * n_kblk] = pertoken_scale.reshape(-1)[: M * n_kblk]

    has_bias = bias is not None
    if has_bias:
        if bias.numel() != N:
            raise ValueError(f"bias needs N={N} elements, got {bias.numel()}")
        bias_t = bias.reshape(-1).to(torch.float32).contiguous()
    else:
        bias_t = pertoken_scale  # dummy, never read by the kernel

    grid = (num_m_tiles * (N // bn) * sk,)
    if sk == 1:
        # The output is the caller's [M, N], so padded rows must not be written:
        # a mask is needed whenever m_pad > M.
        out = torch.empty((M, N), device=x.device, dtype=output_dtype)
        _qbmm_kernel[grid](
            ain,
            sa,
            weight,
            weight_scale,
            out,
            bias_t,
            M,
            N=N,
            K=K,
            BLOCK_M=block_m,
            BLOCK_N=bn,
            BLOCK_K=bk,
            SPLIT_K=1,
            NUM_M_TILES=num_m_tiles,
            PSTRIDE=p_rows * N,
            MX=MX_BLOCK,
            WEIGHT_NZ=weight_nz,
            HAS_BIAS=has_bias,
            A_BCAST=a_bcast,
            SB_NK=sb_nk,
            MASK_M=m_pad != M,
            M_CE=m_ce,
            **launch_opts,
        )
        return out

    # Without a static mask the partials are allocated with padded rows so the
    # main kernel's store needs no mask at all: that runtime mask is the
    # mmadL1 abort trigger, and dropping it is faster anyway.  The extra rows
    # hold duplicates that combine's grid never reads.
    part = torch.empty((sk, p_rows, N), device=x.device, dtype=torch.float32)
    _qbmm_kernel[grid](
        ain,
        sa,
        weight,
        weight_scale,
        part,
        bias_t,
        M,
        N=N,
        K=K,
        BLOCK_M=block_m,
        BLOCK_N=bn,
        BLOCK_K=bk,
        SPLIT_K=sk,
        NUM_M_TILES=num_m_tiles,
        PSTRIDE=p_rows * N,
        MX=MX_BLOCK,
        WEIGHT_NZ=weight_nz,
        HAS_BIAS=has_bias,
        A_BCAST=a_bcast,
        SB_NK=sb_nk,
        MASK_M=False,
        M_CE=m_ce,
        **launch_opts,
    )
    out = torch.empty((M, N), device=x.device, dtype=output_dtype)
    # combine's BLOCK_N: largest power-of-two divisor of N up to 2048, which is
    # the measured sweet spot on both split-K shapes and is non-monotonic on
    # either side (too small spreads the launch floor over too many programs,
    # too large makes the [SK, cbn] FP32 working set spill out of the UB).  The
    # UB bound below keeps that working set at 64KB.
    cbn = next(
        (c for c in _divisors_pow2(N, 1, 2048) if sk * c * 4 <= 64 * 1024),
        _divisors_pow2(N, 1, 2048)[-1],
    )
    _combine_kernel[(M, N // cbn)](
        part,
        out,
        bias_t,
        N=N,
        SPLIT_K=sk,
        PSTRIDE=p_rows * N,
        BLOCK_N=cbn,
        HAS_BIAS=has_bias,
    )
    return out
