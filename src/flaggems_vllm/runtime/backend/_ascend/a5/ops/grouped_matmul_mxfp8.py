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

"""MX-FP8 grouped matmul for vllm-ascend's ``apply_gmm1/apply_gmm2``.

This is the A5 Triton replacement for the following vendor call (without the
SwiGLU epilogue; that fused operation has its own implementation)::

    torch_npu.npu_grouped_matmul(
        x=[x], weight=[w], scale=[weight_scale], per_token_scale=[x_scale],
        split_item=2, group_type=0, group_list=gl, group_list_type=glt,
        output_dtype=out_dtype, scale_dtype=torch_npu.float8_e8m0fnu,
        per_token_scale_dtype=torch_npu.float8_e8m0fnu)[0]

Inputs are fp8_e4m3fn ``x[M,K]`` and ND/NZ ``weight[E,K,N]``.  Both e8m0
scales are carried in uint8 (an e8m0 byte view is also accepted).  A scale is
logically ``[M,K/32]`` and B scale is ``[E,K/32,N]``.  Their physical shapes
end in two: A scale is simply flattened, while B scale ``[E,K/64,N,2]`` is
interleaved as ``(b//2, n, b%2)``.  NZ is detected through the ACL descriptor;
its physical layout is ``[E,N/32,K/16,16,32]`` with the logical shape intact.

The default hardware path uses ``tl.dot_scaled`` (MX-MAC) and is bit-exact
with CANN.  K that is only 32-aligned takes the validated software fallback;
it is BF16-output accurate but is not guaranteed to be bit-exact with CANN.

Constraints established by profiling (do not relax without re-validation):
  * Scale loads are mask-free.  A/A-scale are padded by BLOCK_M rows; padded
    output rows are removed by the store mask.
  * Integer division/modulo must operate on bare ``tl.arange`` vectors, not
    runtime-offset vectors, to preserve regular memory access patterns.
  * BLOCK_N must divide N; hardware-path BLOCK_K=64 (larger values can severely
    degrade the interleaved B-scale access).
  * Persistent scheduling avoids reading group_list back to the host.

Forward inference only; non-contiguous inputs, unsupported dtypes/layouts and
FP4 are rejected rather than silently falling back to torch.
"""

import torch
import torch_npu  # noqa: F401
import triton
import triton.experimental.tle as tle
import triton.language as tl

__all__ = ["grouped_matmul_mxfp8"]

MX_BLOCK = 32
ACL_FORMAT_FRACTAL_NZ = 29


def _cube_core_num(device=None) -> int:
    """Scheduling is persistent over Cube cores (normally 32 on 950DT)."""
    return int(torch.npu.get_device_properties(device).cube_core_num)


@triton.jit
def _e8m0_to_f32(v):
    """Move the biased exponent directly into FP32; zero denotes a zero block."""
    return (v.to(tl.uint32) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _replicate_sa_kernel(
    SA,
    SAB,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MX: tl.constexpr,
):
    """Expand A scale over N for the optional software-scale experiment."""
    m = tl.program_id(0)
    kb = tl.program_id(1)
    n_kblk: tl.constexpr = K // MX
    s = tl.load(SA + m.to(tl.int64) * n_kblk + kb)
    off = (m.to(tl.int64) * n_kblk + kb) * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(SAB + off, tl.full((BLOCK_N,), 1, tl.float32) * _e8m0_to_f32(s))


@triton.jit
def _dequant_a_kernel(
    A,
    SA,
    ABF,
    M,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MX: tl.constexpr,
):
    """Exactly fold the power-of-two A scale into BF16 for the slow path."""
    m = tl.program_id(0)
    n_kblk: tl.constexpr = K // MX
    row_a = A + m.to(tl.int64) * K
    row_s = SA + m.to(tl.int64) * n_kblk
    row_o = ABF + m.to(tl.int64) * K
    for k0 in tl.range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a = tl.load(row_a + offs_k, mask=mask_k, other=0.0)
        s = tl.load(row_s + offs_k // MX, mask=mask_k, other=0)
        tl.store(
            row_o + offs_k,
            (a.to(tl.float32) * _e8m0_to_f32(s)).to(tl.bfloat16),
            mask=mask_k,
        )


@triton.jit
def _tile_order(idx, num_m_tiles, num_n_tiles, THRESHOLD: tl.constexpr):
    if num_n_tiles >= THRESHOLD:
        tile_m = idx // num_n_tiles
        tile_n = idx % num_n_tiles
    else:
        tile_m = idx % num_m_tiles
        tile_n = idx // num_m_tiles
    return tile_m, tile_n


@triton.jit
def _grouped_matmul_mxfp8_kernel(
    AIN,
    SAB,
    SA2,
    B,
    SB,
    C,
    group_list,
    GROUPS: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_LIST_TYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILE_THRESHOLD: tl.constexpr,
    MX: tl.constexpr,
    VARIANT: tl.constexpr,
    WEIGHT_NZ: tl.constexpr,
    SLAB: tl.constexpr,
    B_L1: tl.constexpr,
    K_STAGES: tl.constexpr,
    K_UNROLL: tl.constexpr,
    B_CACHE: tl.constexpr,
    SB_CACHE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SB_NK: tl.constexpr,
    A_MASK: tl.constexpr,
):
    total_cores = tl.num_programs(0)
    core_idx = tl.program_id(0)
    num_n_tiles: tl.constexpr = N // BLOCK_N
    n_kblk: tl.constexpr = K // MX
    sb_stride_e: tl.constexpr = n_kblk * N
    SLAB_MUL: tl.constexpr = SLAB // MX

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
            offs_m = (group_start + tile_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(
                tl.int64
            )
            offs_n0 = tl.arange(0, BLOCK_N).to(tl.int64)
            offs_n = (tile_n * BLOCK_N + offs_n0).to(tl.int64)
            offs_k = tl.arange(0, MX).to(tl.int64)
            offs_ks = tl.arange(0, SLAB).to(tl.int64)
            mask_m = offs_m < group_end

            g64 = group_idx.to(tl.int64)
            a_base = AIN + offs_m[:, None] * K + offs_k[None, :]
            if WEIGHT_NZ:
                # All // and % below operate on a bare arange.
                nz_g: tl.constexpr = BLOCK_N // 32
                nvec = (offs_n0 // 32) * (K * 32) + (offs_n0 % 32)
                b_base = (
                    B
                    + g64 * (K * N)
                    + tile_n.to(tl.int64) * (nz_g * K * 32)
                    + nvec[None, :]
                    + offs_ks[:, None] * 32
                )
            else:
                b_base = B + g64 * (K * N) + offs_ks[:, None] * N + offs_n[None, :]
            sb_base = SB + g64 * sb_stride_e
            sab_base = (
                SAB
                + offs_m[:, None] * (n_kblk * BLOCK_N)
                + tl.arange(0, BLOCK_N)[None, :]
            )

            a2_base = AIN + offs_m[:, None] * K
            if WEIGHT_NZ:
                nz_g2: tl.constexpr = BLOCK_N // 32
                nvec2 = (offs_n0 // 32) * (K * 32) + (offs_n0 % 32)
                b2_base = (
                    B
                    + g64 * (K * N)
                    + tile_n.to(tl.int64) * (nz_g2 * K * 32)
                    + nvec2[None, :]
                )
            else:
                b2_base = B + g64 * (K * N) + offs_n[None, :]

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            if VARIANT == 2:
                # Both fp8 operands and both e8m0 scales enter one MX-MAC.
                offs_kk = tl.arange(0, BLOCK_K).to(tl.int64)
                offs_sk = tl.arange(0, BLOCK_K // MX).to(tl.int64)
                sa2_base = SA2 + offs_m[:, None] * n_kblk + offs_sk[None, :]
                for k0 in tl.range(
                    0,
                    n_kblk,
                    BLOCK_K // MX,
                    num_stages=K_STAGES,
                    loop_unroll_factor=K_UNROLL,
                ):
                    kb0 = k0.to(tl.int64)
                    koff = kb0 * MX
                    if A_MASK:
                        a = tl.load(
                            a2_base + koff + offs_kk[None, :],
                            mask=mask_m[:, None],
                            other=0.0,
                        )
                    else:
                        a = tl.load(a2_base + koff + offs_kk[None, :])
                    if WEIGHT_NZ:
                        b = tl.load(
                            b2_base + (koff + offs_kk)[:, None] * 32,
                            cache_modifier=B_CACHE,
                        )
                    else:
                        b = tl.load(
                            b2_base + (koff + offs_kk)[:, None] * N,
                            cache_modifier=B_CACHE,
                        )
                    sa = tl.load(sa2_base + kb0)
                    if SB_NK:
                        sb = tl.load(
                            sb_base + offs_n[:, None] * n_kblk + kb0 + offs_sk[None, :],
                            cache_modifier=SB_CACHE,
                        )
                    else:
                        # kb0 is even for BLOCK_K=64; // and % stay on arange.
                        sb_off = (
                            (kb0 // 2) * (N * 2)
                            + offs_n[:, None] * 2
                            + (offs_sk[None, :] // 2) * (N * 2)
                            + (offs_sk[None, :] % 2)
                        )
                        sb = tl.load(sb_base + sb_off, cache_modifier=SB_CACHE)
                    acc = tl.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", acc)
            else:
                if B_L1:
                    b_l1 = tle.dsa.alloc(
                        [SLAB, BLOCK_N],
                        dtype=B.dtype.element_ty,
                        mem_addr_space=tle.dsa.ascend.L1,
                    )
                    b_bp = tl.make_block_ptr(
                        B + g64 * (K * N),
                        (K, N),
                        (N, 1),
                        (0, tile_n * BLOCK_N),
                        (SLAB, BLOCK_N),
                        (1, 0),
                    )

                for kb in tl.range(
                    n_kblk // SLAB_MUL,
                    num_stages=K_STAGES,
                    loop_unroll_factor=K_UNROLL,
                ):
                    kb64 = kb.to(tl.int64)
                    k_off = kb64 * (MX * SLAB_MUL)
                    b_off = k_off * 32 if WEIGHT_NZ else k_off * N
                    if B_L1 == 1:
                        # Experimental: known unsupported in the current compiler.
                        tle.dsa.copy(b_bp, b_l1, [SLAB, BLOCK_N])
                        b_slab = tle.dsa.to_tensor(b_l1, writable=False)
                        b_bp = tl.advance(b_bp, [SLAB, 0])
                    elif B_L1 == 2:
                        b_full = tle.dsa.to_tensor(b_l1, writable=True)
                        b_slab = tle.dsa.insert_slice(
                            b_full,
                            tl.load(b_bp),
                            offsets=[0, 0],
                            sizes=[SLAB, BLOCK_N],
                            strides=[1, 1],
                        )
                        b_bp = tl.advance(b_bp, [SLAB, 0])
                    else:
                        b_slab = tl.load(b_base + b_off, cache_modifier=B_CACHE)

                    for j in tl.static_range(SLAB_MUL):
                        jj = j * MX
                        kbj = kb64 * SLAB_MUL + j
                        a = tl.load(
                            a_base + k_off + jj, mask=mask_m[:, None], other=0.0
                        )
                        if SLAB_MUL == 1:
                            b = b_slab
                        else:
                            b = tle.dsa.extract_slice(
                                b_slab, [jj, 0], [MX, BLOCK_N], [1, 1]
                            )
                        sb_off = (kbj // 2) * (N * 2) + offs_n * 2 + (kbj % 2)
                        sb_raw = tl.load(sb_base + sb_off, cache_modifier=SB_CACHE)
                        if VARIANT == 1:
                            sa_b = tl.load(sab_base + kbj * BLOCK_N)
                            acc += tl.dot(a, b) * sa_b * _e8m0_to_f32(sb_raw)[None, :]
                        else:
                            acc += (
                                tl.dot(a, b.to(tl.bfloat16))
                                * _e8m0_to_f32(sb_raw)[None, :]
                            )

            tl.store(
                C + offs_m[:, None] * N + offs_n[None, :],
                acc.to(C.dtype.element_ty),
                mask=mask_m[:, None],
            )

        last_count = current_count % total_cores
        group_start = group_end


def _check(x, weight, weight_scale, per_token_scale, group_list, group_list_type):
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D [M,K], got {tuple(x.shape)}")
    if weight.ndim != 3:
        raise ValueError(f"weight must be 3-D [E,K,N], got {tuple(weight.shape)}")
    M, K = x.shape
    E, Kw, N = weight.shape
    if K != Kw:
        raise ValueError(f"K mismatch: x {K} vs weight {Kw}")
    if K % MX_BLOCK:
        raise ValueError(f"K={K} must be a multiple of {MX_BLOCK}")
    if x.dtype != torch.float8_e4m3fn or weight.dtype != torch.float8_e4m3fn:
        raise TypeError(f"x/weight must be fp8_e4m3fn, got {x.dtype}/{weight.dtype}")
    for name, t in (
        ("weight_scale", weight_scale),
        ("per_token_scale", per_token_scale),
    ):
        if t.dtype not in (torch.uint8, torch.float8_e8m0fnu):
            raise TypeError(f"{name} must contain e8m0 bytes, got {t.dtype}")
    if group_list.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"group_list must be int32/int64, got {group_list.dtype}")
    n_kblk = K // MX_BLOCK
    if weight_scale.numel() != E * n_kblk * N:
        raise ValueError(
            f"weight_scale needs {E * n_kblk * N} elements, got {weight_scale.numel()}"
        )
    if per_token_scale.numel() != M * n_kblk:
        raise ValueError(
            f"per_token_scale needs {M * n_kblk} elements, got {per_token_scale.numel()}"
        )
    if group_list_type not in (0, 1):
        raise ValueError("group_list_type must be 0 (cumsum) or 1 (counts)")
    if group_list.numel() != E:
        raise ValueError(f"group_list length must be E={E}, got {group_list.numel()}")
    return M, K, E, N


def _is_nz(t: torch.Tensor) -> bool:
    return torch_npu.get_npu_format(t) == ACL_FORMAT_FRACTAL_NZ


def _pick_block_n(N: int, prefer: int | None = None, variant: int = 2) -> int:
    """Choose a divisor of N; scale loads intentionally have no mask."""
    if prefer is None:
        prefer = 1024 if variant == 2 else 512
    if N % prefer == 0:
        return prefer
    for cand in (1024, 512, 256, 128, 64, 32):
        if cand <= prefer and N % cand == 0:
            return cand
    for cand in (1024, 512, 256, 128, 64, 32):
        if N % cand == 0:
            return cand
    raise ValueError(f"N={N} needs a power-of-two divisor >= 32")


def grouped_matmul_mxfp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    per_token_scale: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int = 1,
    output_dtype: torch.dtype = torch.bfloat16,
    *,
    block_m: int = 16,
    block_n: int | None = None,
    tile_threshold: int = 8,
    num_cores: int | None = None,
    variant: int = 2,
    weight_nz: bool | None = None,
    slab: int = MX_BLOCK,
    b_l1: int = 0,
    k_stages: int | None = None,
    k_unroll: int | None = None,
    launch_num_stages: int | None = None,
    multibuffer_mode: dict[str, int] | None = None,
    b_cache: str = "",
    sb_cache: str = "",
    block_k: int | None = None,
    sb_nk: bool = False,
    a_mask: bool | None = None,
) -> torch.Tensor:
    """Return the BF16/FP16 MX-FP8 grouped matmul output ``[M,N]``.

    ``group_list_type=1`` means per-expert row counts; 0 means cumulative ends.
    The optional tuning parameters preserve the source implementation's
    experiment interface.  Production defaults select variant=2, BM=16,
    BN=1024 (or its largest suitable divisor), BK=64, and mask-free A loads.
    No NVIDIA autotune is attached because this is an A5-only implementation.
    """
    if output_dtype not in (torch.bfloat16, torch.float16):
        raise NotImplementedError(f"output_dtype {output_dtype} is not supported")
    M, K, E, N = _check(
        x, weight, weight_scale, per_token_scale, group_list, group_list_type
    )
    if x.device.type != "npu" or any(
        t.device != x.device
        for t in (weight, weight_scale, per_token_scale, group_list)
    ):
        raise ValueError("all tensors must be on the same NPU")
    if weight_nz is None:
        weight_nz = _is_nz(weight)
    if weight_nz and (K % 16 != 0 or N % 32 != 0):
        raise ValueError(f"NZ requires K%16==0 and N%32==0, got K={K} N={N}")

    for name, t in (
        ("x", x),
        ("weight_scale", weight_scale),
        ("per_token_scale", per_token_scale),
        ("group_list", group_list),
    ):
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if not weight_nz and not weight.is_contiguous():
        raise ValueError("ND weight must be contiguous")

    # Byte views preserve the e8m0 representation without a compute cast.
    if weight_scale.dtype != torch.uint8:
        weight_scale = weight_scale.view(torch.uint8)
    if per_token_scale.dtype != torch.uint8:
        per_token_scale = per_token_scale.view(torch.uint8)

    bn = block_n if block_n is not None else _pick_block_n(N, variant=variant)
    if N % bn != 0:
        raise ValueError(f"block_n={bn} must divide N={N} for mask-free scale loads")

    cores = num_cores if num_cores is not None else _cube_core_num(x.device.index)
    if cores <= 0:
        raise ValueError(f"num_cores must be positive, got {cores}")
    n_kblk = K // MX_BLOCK
    bk = block_k if block_k is not None else min(64, K)
    if variant == 2 and block_k is None and (K % 64 != 0 or K < 64):
        variant = 0
        bn = block_n if block_n is not None else _pick_block_n(N, variant=0)
    if variant == 2:
        if bk % (2 * MX_BLOCK) != 0:
            raise ValueError(f"variant=2 block_k={bk} must be a multiple of 64")
        if K % bk != 0:
            raise ValueError(f"block_k={bk} must divide K={K}")
    if slab % MX_BLOCK != 0:
        raise ValueError(f"slab={slab} must be a multiple of {MX_BLOCK}")
    if b_l1 and slab == MX_BLOCK:
        raise ValueError("b_l1 requires slab > MX_BLOCK")
    if b_l1 and weight_nz:
        raise ValueError("b_l1 does not support NZ weight")
    if (K // MX_BLOCK) % (slab // MX_BLOCK) != 0:
        raise ValueError(f"slab={slab} must divide K={K}")

    dummy = per_token_scale
    sa2 = dummy
    if variant == 2:
        # Whole-buffer zero is intentional: filling only the tail dispatches to
        # AiCPU and costs substantially more than this small scale allocation.
        sa2 = torch.zeros((M + block_m) * n_kblk, device=x.device, dtype=torch.uint8)
        sa2[: M * n_kblk] = per_token_scale.reshape(-1)[: M * n_kblk]
        sab = dummy
        ain = torch.empty((M + block_m, K), device=x.device, dtype=x.dtype)
        ain[:M] = x
        # Padded A rows may contain arbitrary fp8: only the affected output rows
        # are removed by the store mask.  They cannot contaminate valid rows.
        if a_mask is None:
            a_mask = False
        if sb_nk:
            weight_scale = (
                weight_scale.reshape(E, n_kblk // 2, N, 2)
                .permute(0, 1, 3, 2)
                .reshape(E, n_kblk, N)
                .permute(0, 2, 1)
                .contiguous()
            )
    elif variant == 1:
        sab = torch.empty(
            (M + block_m, n_kblk, bn), device=x.device, dtype=torch.float32
        )
        _replicate_sa_kernel[(M, n_kblk)](
            per_token_scale, sab, K=K, BLOCK_N=bn, MX=MX_BLOCK
        )
        ain = x
    else:
        ain = torch.empty((M, K), device=x.device, dtype=torch.bfloat16)
        _dequant_a_kernel[(M,)](
            x,
            per_token_scale,
            ain,
            M,
            K=K,
            BLOCK_K=min(512, triton.next_power_of_2(K)),
            MX=MX_BLOCK,
        )
        sab = dummy

    out = torch.empty((M, N), device=x.device, dtype=output_dtype)
    if a_mask is None:
        a_mask = True
    extra = {} if multibuffer_mode is None else {"multibuffer_mode": multibuffer_mode}
    if launch_num_stages is not None:
        if multibuffer_mode is not None:
            raise ValueError(
                "launch_num_stages and multibuffer_mode are mutually exclusive"
            )
        extra["num_stages"] = launch_num_stages
    _grouped_matmul_mxfp8_kernel[(cores,)](
        ain,
        sab,
        sa2,
        weight,
        weight_scale,
        out,
        group_list,
        GROUPS=E,
        N=N,
        K=K,
        GROUP_LIST_TYPE=group_list_type,
        BLOCK_M=block_m,
        BLOCK_N=bn,
        TILE_THRESHOLD=tile_threshold,
        MX=MX_BLOCK,
        VARIANT=variant,
        WEIGHT_NZ=weight_nz,
        SLAB=slab,
        B_L1=b_l1,
        K_STAGES=k_stages,
        K_UNROLL=k_unroll,
        B_CACHE=b_cache,
        SB_CACHE=sb_cache,
        BLOCK_K=bk,
        SB_NK=sb_nk,
        A_MASK=a_mask,
        **extra,
    )
    return out
