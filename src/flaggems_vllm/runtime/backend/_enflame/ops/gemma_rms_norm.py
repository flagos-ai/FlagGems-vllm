import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Whole-row register-resident kernel covers up to this next_power_of_2(N); larger N
# falls back to the streaming two-pass kernel (register footprint would not fit).
_WHOLE_ROW_MAX_BLOCK_N = 16384

# L600 (GCU400): 48 SPs. The persistent row-stream below launches one program per
# SP (grid = min(M, 48)) instead of one CTA per row.
_PERSIST_GRID = 48

# num_warps sweet spots measured on L600 (fp16 bench matrix, do_bench median):
#   BLOCK_N <= 2048 -> nw 1 ; BLOCK_N >= 4096 -> nw 4 (nw1 does not compile at
#   BLOCK_N 8192 on the current flagtree toolchain).
_NW_SMALL_BLOCK = 1
_NW_LARGE_BLOCK = 4
_UNROLL2_MAX_BLOCK_N = 512


@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


# GCU rewrite of the whole-row kernel as a persistent 1D row-stream:
#   * one program per SP walks rows in strides of num_programs, removing the
#     per-row CTA relaunch chain (~1.3us "load dispatch + reduction tree") that
#     dominates every M > 48 shape of the one-CTA-per-row form;
#   * addressing is scalar row base + 1D column arange with a 1D mask only —
#     2D pointer tiles (offs = m[:, None] * N + n[None, :]) lower to
#     memref_ext.slice_pad_start ops the current flagtree toolchain rejects on
#     fresh compiles (compiler_issues/vllm_gemma_rms_norm.md);
#   * the weight row is loaded once per program and pre-scaled to (1 + w) so
#     the per-row body is load -> fp32 square-sum tree -> scale -> store;
#   * UNROLL2 keeps a second row's load in flight across the tree of the
#     first, winning ~20-30% on narrow rows (BLOCK_N <= 512); the odd-row
#     guard keeps every M correct (rows are covered exactly once).
@triton.jit(do_not_specialize=["eps"])
def _gemma_rms_norm_persist_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    eps,
    BLOCK_N: tl.constexpr,
    UNROLL2: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_N)
    cmask = cols < N
    wp1 = 1.0 + tl.load(w_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
    if UNROLL2:
        for row0 in range(pid * 2, M, nprog * 2):
            r1 = row0 + 1
            base0 = row0 * N
            x0 = tl.load(x_ptr + base0 + cols, mask=cmask, other=0.0)
            if r1 < M:
                base1 = r1 * N
                x1 = tl.load(x_ptr + base1 + cols, mask=cmask, other=0.0)
                xf1 = x1.to(tl.float32)
                inv1 = 1.0 / tl.sqrt(tl.sum(xf1 * xf1, axis=0) / N + eps)
                tl.store(out_ptr + base1 + cols, xf1 * inv1 * wp1, mask=cmask)
            xf0 = x0.to(tl.float32)
            inv0 = 1.0 / tl.sqrt(tl.sum(xf0 * xf0, axis=0) / N + eps)
            tl.store(out_ptr + base0 + cols, xf0 * inv0 * wp1, mask=cmask)
    else:
        for row in range(pid, M, nprog):
            base = row * N
            x = tl.load(x_ptr + base + cols, mask=cmask, other=0.0)
            xf = x.to(tl.float32)
            inv_rms = 1.0 / tl.sqrt(tl.sum(xf * xf, axis=0) / N + eps)
            tl.store(out_ptr + base + cols, xf * inv_rms * wp1, mask=cmask)


@triton.autotune(
    [
        triton.Config(kwargs={"TILE_N": tile_n}, num_warps=num_warps)
        for tile_n in [1024, 2048, 4096, 8192]
        for num_warps in [4, 8, 16]
    ],
    key=["M", "N"],
)
@triton.jit(do_not_specialize=["eps"])
def _gemma_rms_norm_loop_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    M,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    pid = tl.program_id(0)

    acc = tl.zeros((1,), dtype=tl.float32)
    num_steps = tl.cdiv(N, TILE_N)

    for step in range(0, num_steps - 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets).to(tl.float32)
        acc += tl.sum(x * x, axis=0)

    start_n = (num_steps - 1) * TILE_N
    n_offsets = start_n + tl.arange(0, TILE_N)
    mask = n_offsets < N
    x = tl.load(in_ptr + pid * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
    acc += tl.sum(x * x, axis=0)

    var = tl.sum(acc) / N
    rrms = 1 / tl.sqrt(var + eps)

    prev_multiple = prev_multiple_of(N, TILE_N)

    for start_n in range(0, TILE_N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        w = tl.load(w_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
        y = x * rrms * (1.0 + w)
        tl.store(out_ptr + pid * N + n_offsets, y, mask=mask)

    for start_n in range(TILE_N, N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            eviction_policy="evict_first",
        ).to(tl.float32)
        w = tl.load(w_ptr + n_offsets).to(tl.float32)
        y = x * rrms * (1.0 + w)
        tl.store(out_ptr + pid * N + n_offsets, y)


def gemma_rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    logger.debug("GEMS GEMMA_RMSNORM")

    if x.ndim == 0:
        raise ValueError("gemma_rms_norm expects an input with at least one dimension")
    orig_shape = x.shape
    N = orig_shape[-1]
    if w.ndim != 1 or w.shape[0] != N:
        raise ValueError(f"weight must have shape ({N},), got {tuple(w.shape)}")

    if any(dim == 0 for dim in orig_shape):
        return torch.empty_like(x)

    if not x.is_contiguous() or not w.is_contiguous():
        raise NotImplementedError("gemma_rms_norm requires contiguous tensors")

    M = 1
    for dim in orig_shape[:-1]:
        M *= dim
    out = torch.empty_like(x)

    if triton.next_power_of_2(N) <= _WHOLE_ROW_MAX_BLOCK_N:
        block_n = triton.next_power_of_2(N)
        grid_n = min(M, _PERSIST_GRID)
        if M > 1 and block_n <= _UNROLL2_MAX_BLOCK_N:
            _gemma_rms_norm_persist_kernel[(grid_n,)](
                x,
                w,
                out,
                M,
                N,
                eps,
                BLOCK_N=block_n,
                UNROLL2=True,
                num_warps=_NW_SMALL_BLOCK,
            )
        else:
            num_warps = _NW_SMALL_BLOCK if block_n <= 2048 else _NW_LARGE_BLOCK
            _gemma_rms_norm_persist_kernel[(grid_n,)](
                x,
                w,
                out,
                M,
                N,
                eps,
                BLOCK_N=block_n,
                UNROLL2=False,
                num_warps=num_warps,
            )
    else:
        _gemma_rms_norm_loop_kernel[M,](out, x, w, M, N, eps)

    return out.view(orig_shape)
