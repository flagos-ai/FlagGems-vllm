import torch
import triton
import triton.language as tl

# Autotune candidates: BLOCK_N x num_warps (row) / BLOCK_M x num_warps (multirow).
# BLOCK_N only applies to the masked (non-exact decomposition) path. Kept small:
# every (M, N) shape sweeps the whole list once.
_ROW_CONFIGS = [
    triton.Config(kwargs={"BLOCK_N": bn}, num_warps=nw)
    for bn in (2048, 8192)
    for nw in (4, 8, 16)
]

_MULTIROW_CONFIGS = [
    triton.Config(kwargs={"BLOCK_M": bm}, num_warps=nw)
    for bm in (2, 4, 8, 16)
    for nw in (4, 8, 16)
]

_LOOP_CONFIGS = [
    triton.Config(kwargs={"TILE_N": tn}, num_warps=nw)
    for tn in (2048, 8192, 16384)
    for nw in (8, 16)
]


def _pow2_chunks(n):
    """Greedy decomposition of n into at most three power-of-two chunks.

    Returns (chunks, exact) where exact is True iff the chunks sum to n
    (i.e. n has at most three bits set).
    """
    chunks, rem = [], n
    while rem > 0 and len(chunks) < 3:
        c = 1 << (rem.bit_length() - 1)
        chunks.append(c)
        rem -= c
    return chunks, rem == 0


@triton.autotune(configs=_ROW_CONFIGS, key=["M", "N"])
@triton.jit
def _gemma_rmsnorm_row_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    eps,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EXACT: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * N
    if EXACT:
        # exact pow2 chunks, C0 + C1 + C2 == N: no mask anywhere
        o0 = tl.arange(0, C0)
        x0 = tl.load(x_ptr + base + o0).to(tl.float32)
        w0 = tl.load(w_ptr + o0).to(tl.float32)
        ssq = tl.sum(x0 * x0)
        if C1 > 0:
            o1 = C0 + tl.arange(0, C1)
            x1 = tl.load(x_ptr + base + o1).to(tl.float32)
            ssq += tl.sum(x1 * x1)
        if C2 > 0:
            o2 = C0 + C1 + tl.arange(0, C2)
            x2 = tl.load(x_ptr + base + o2).to(tl.float32)
            ssq += tl.sum(x2 * x2)
        rrms = tl.rsqrt(ssq / N + eps)
        y0 = x0 * rrms * (1.0 + w0)
        tl.store(out_ptr + base + o0, y0.to(out_ptr.dtype.element_ty))
        if C1 > 0:
            w1 = tl.load(w_ptr + o1).to(tl.float32)
            y1 = x1 * rrms * (1.0 + w1)
            tl.store(out_ptr + base + o1, y1.to(out_ptr.dtype.element_ty))
        if C2 > 0:
            w2 = tl.load(w_ptr + o2).to(tl.float32)
            y2 = x2 * rrms * (1.0 + w2)
            tl.store(out_ptr + base + o2, y2.to(out_ptr.dtype.element_ty))
    else:
        # decomposition not exact (>3 bits set): ordinary masked tiles,
        # looped so every autotuned BLOCK_N stays correct even when < N
        tile = tl.arange(0, BLOCK_N)
        acc = tl.zeros((), dtype=tl.float32)
        for start in range(0, N, BLOCK_N):
            offs = start + tile
            x = tl.load(x_ptr + base + offs, mask=offs < N, other=0.0).to(tl.float32)
            acc += tl.sum(x * x)
        rrms = tl.rsqrt(acc / N + eps)
        for start in range(0, N, BLOCK_N):
            offs = start + tile
            mask = offs < N
            x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            y = x * rrms * (1.0 + w)
            tl.store(out_ptr + base + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.autotune(configs=_MULTIROW_CONFIGS, key=["M", "N"])
@triton.jit
def _gemma_rmsnorm_multirow_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EXACT: tl.constexpr,
):
    m_offs = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M
    if EXACT:
        n_offs = tl.arange(0, N)
        ld_mask = m_mask[:, None]
    else:
        n_offs = tl.arange(0, BLOCK_N)
        ld_mask = m_mask[:, None] & (n_offs < N)[None, :]
    ptrs = m_offs[:, None] * N + n_offs[None, :]
    x = tl.load(x_ptr + ptrs, mask=ld_mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + n_offs, mask=n_offs < N, other=0.0).to(tl.float32)
    rrms = tl.rsqrt(tl.sum(x * x, axis=1) / N + eps)
    y = x * rrms[:, None] * (1.0 + w)[None, :]
    tl.store(out_ptr + ptrs, y.to(out_ptr.dtype.element_ty), mask=ld_mask)


@triton.autotune(configs=_LOOP_CONFIGS, key=["M", "N"])
@triton.jit
def _gemma_rmsnorm_loop_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * N
    tile = tl.arange(0, TILE_N)

    # pass 1: sum of squares
    acc = tl.zeros((), dtype=tl.float32)
    for start in range(0, N, TILE_N):
        offs = start + tile
        x = tl.load(x_ptr + base + offs, mask=offs < N, other=0.0).to(tl.float32)
        acc += tl.sum(x * x)
    rrms = tl.rsqrt(acc / N + eps)

    # pass 2: scale and store
    for start in range(0, N, TILE_N):
        offs = start + tile
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = x * rrms * (1.0 + w)
        tl.store(out_ptr + base + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def gemma_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps=1e-6) -> torch.Tensor:
    assert x.is_contiguous()
    assert w.is_contiguous()
    n = x.shape[-1]
    if n == 0:
        return torch.empty_like(x)
    m = x.numel() // n
    out = torch.empty_like(x)
    if m == 0:
        return out

    block_n = triton.next_power_of_2(n)

    if block_n <= 1024 and m >= 256:
        # large-M x small-N: rows share one weight load, fewer programs
        _gemma_rmsnorm_multirow_kernel[lambda META: (triton.cdiv(m, META["BLOCK_M"]),)](
            x.view(m, n),
            w,
            out.view(m, n),
            m,
            n,
            eps,
            BLOCK_N=block_n,
            EXACT=n == block_n,
        )
        return out

    if block_n <= 16384:
        chunks, exact = _pow2_chunks(n)
        _gemma_rmsnorm_row_kernel[(m,)](
            x.view(m, n),
            w,
            out.view(m, n),
            m,
            n,
            eps,
            C0=chunks[0],
            C1=chunks[1] if len(chunks) > 1 else 0,
            C2=chunks[2] if len(chunks) > 2 else 0,
            EXACT=exact,
        )
        return out

    _gemma_rmsnorm_loop_kernel[(m,)](
        x.view(m, n),
        w,
        out.view(m, n),
        m,
        n,
        eps,
    )
    return out
