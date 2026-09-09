import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


# Widest N handled by the single-pass chunked kernel (chunks fit the 192KB
# per-core unified buffer); wider rows use the two-pass loop kernel.
_CHUNKED_N_MAX = 10240

# Widest N handled by the 2-row block kernel (2 * N * 4B fp32 live values).
_BM2_N_MAX = 6144

# Above this row count the grid is split finer (2x the vector cores) which
# wins for very large batches.
_FINE_GRID_M = 3072

_CACHED_CORE_NUM = None


def _get_core_num():
    global _CACHED_CORE_NUM
    if _CACHED_CORE_NUM is None:
        try:
            import torch_npu  # noqa: F401

            current_device = torch.npu.current_device()
            torch.npu.set_device(current_device)
            cores_dict = torch.npu.get_device_limit(current_device)
            _CACHED_CORE_NUM = cores_dict["vector_core_num"]
        except (ImportError, AttributeError, KeyError, TypeError):
            _CACHED_CORE_NUM = None
    return _CACHED_CORE_NUM


def _chunk_params(n):
    """Decompose n into at most 3 power-of-two chunks.

    Returns (C0, C1, C2, exact). When n needs more than 3 chunks (e.g. 3840)
    the caller falls back to the next power of two with a masked chunk 0
    (exact=False).
    """
    chunks = []
    rem = n
    while rem > 0 and len(chunks) < 3:
        c = 1 << (rem.bit_length() - 1)
        chunks.append(c)
        rem -= c
    if rem == 0:
        while len(chunks) < 3:
            chunks.append(1)
        return chunks[0], chunks[1], chunks[2], True
    return triton.next_power_of_2(n), 1, 1, False


@triton.jit(do_not_specialize=["eps"])
def _gemma_rmsnorm_kernel(
    y_ptr,
    x_ptr,
    w_ptr,
    MAX_ROWS,
    eps,
    N_ROWS,
    N,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    EXACT0: tl.constexpr,
):
    # One row per iteration; chunks C0(+C1)(+C2) tile the row exactly when
    # EXACT0, otherwise chunk 0 is next_pow2(N) and masked. Padded chunks
    # (C* == 1) are compiled out by the constexpr guards below.
    pid = tl.program_id(0)
    y_ptr += pid * N * N_ROWS
    x_ptr += pid * N * N_ROWS
    base_row = pid * N_ROWS
    rows = min(base_row + N_ROWS, MAX_ROWS) - base_row

    o0 = tl.arange(0, C0)
    o1 = C0 + tl.arange(0, C1)
    o2 = C0 + C1 + tl.arange(0, C2)
    m0 = o0 < N
    m1 = o1 < N
    m2 = o2 < N
    # preload (1 + w) once per program; reused by every row below
    if EXACT0:
        w0 = (1.0 + tl.load(w_ptr + o0).to(tl.float32)) * 1.0
    else:
        w0 = (1.0 + tl.load(w_ptr + o0, mask=m0, other=0.0).to(tl.float32)) * 1.0
    if C1 > 1:
        w1 = (1.0 + tl.load(w_ptr + o1, mask=m1, other=0.0).to(tl.float32)) * 1.0
    if C2 > 1:
        w2 = (1.0 + tl.load(w_ptr + o2, mask=m2, other=0.0).to(tl.float32)) * 1.0

    for row_off in tl.range(0, rows, 1):
        b = row_off * N
        if EXACT0:
            x0 = tl.load(x_ptr + b + o0).to(tl.float32)
        else:
            x0 = tl.load(x_ptr + b + o0, mask=m0, other=0.0).to(tl.float32)
        ssq = tl.sum(x0 * x0)
        if C1 > 1:
            x1 = tl.load(x_ptr + b + o1, mask=m1, other=0.0).to(tl.float32)
            ssq += tl.sum(x1 * x1)
        if C2 > 1:
            x2 = tl.load(x_ptr + b + o2, mask=m2, other=0.0).to(tl.float32)
            ssq += tl.sum(x2 * x2)
        rrms = tl.rsqrt(ssq / N + eps)
        if EXACT0:
            tl.store(y_ptr + b + o0, (x0 * rrms * w0).to(y_ptr.dtype.element_ty))
        else:
            tl.store(
                y_ptr + b + o0, (x0 * rrms * w0).to(y_ptr.dtype.element_ty), mask=m0
            )
        if C1 > 1:
            tl.store(
                y_ptr + b + o1, (x1 * rrms * w1).to(y_ptr.dtype.element_ty), mask=m1
            )
        if C2 > 1:
            tl.store(
                y_ptr + b + o2, (x2 * rrms * w2).to(y_ptr.dtype.element_ty), mask=m2
            )


@triton.jit(do_not_specialize=["eps"])
def _gemma_rmsnorm_bm2_kernel(
    y_ptr,
    x_ptr,
    w_ptr,
    MAX_ROWS,
    eps,
    N_ROWS,
    N,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    EXACT0: tl.constexpr,
):
    # Two rows per iteration as a 2D block: doubles the outstanding loads and
    # hides the cross-lane reduction latency; wins for large row counts.
    pid = tl.program_id(0)
    y_ptr += pid * N * N_ROWS
    x_ptr += pid * N * N_ROWS
    base_row = pid * N_ROWS
    rows = min(base_row + N_ROWS, MAX_ROWS) - base_row

    r = tl.arange(0, 2)
    o0 = tl.arange(0, C0)
    o1 = C0 + tl.arange(0, C1)
    o2 = C0 + C1 + tl.arange(0, C2)
    m0 = o0 < N
    m1 = o1 < N
    m2 = o2 < N
    if EXACT0:
        w0 = (1.0 + tl.load(w_ptr + o0).to(tl.float32)) * 1.0
    else:
        w0 = (1.0 + tl.load(w_ptr + o0, mask=m0, other=0.0).to(tl.float32)) * 1.0
    if C1 > 1:
        w1 = (1.0 + tl.load(w_ptr + o1, mask=m1, other=0.0).to(tl.float32)) * 1.0
    if C2 > 1:
        w2 = (1.0 + tl.load(w_ptr + o2, mask=m2, other=0.0).to(tl.float32)) * 1.0

    for row_off in tl.range(0, rows, 2):
        rb = row_off + r
        rm = rb < rows
        if EXACT0:
            x0 = tl.load(
                x_ptr + rb[:, None] * N + o0[None, :], mask=rm[:, None], other=0.0
            ).to(tl.float32)
        else:
            x0 = tl.load(
                x_ptr + rb[:, None] * N + o0[None, :],
                mask=rm[:, None] & m0[None, :],
                other=0.0,
            ).to(tl.float32)
        ssq = tl.sum(x0 * x0, axis=1)
        if C1 > 1:
            x1 = tl.load(
                x_ptr + rb[:, None] * N + o1[None, :],
                mask=rm[:, None] & m1[None, :],
                other=0.0,
            ).to(tl.float32)
            ssq += tl.sum(x1 * x1, axis=1)
        if C2 > 1:
            x2 = tl.load(
                x_ptr + rb[:, None] * N + o2[None, :],
                mask=rm[:, None] & m2[None, :],
                other=0.0,
            ).to(tl.float32)
            ssq += tl.sum(x2 * x2, axis=1)
        rrms = tl.rsqrt(ssq / N + eps)
        if EXACT0:
            tl.store(
                y_ptr + rb[:, None] * N + o0[None, :],
                (x0 * rrms[:, None] * w0[None, :]).to(y_ptr.dtype.element_ty),
                mask=rm[:, None],
            )
        else:
            tl.store(
                y_ptr + rb[:, None] * N + o0[None, :],
                (x0 * rrms[:, None] * w0[None, :]).to(y_ptr.dtype.element_ty),
                mask=rm[:, None] & m0[None, :],
            )
        if C1 > 1:
            tl.store(
                y_ptr + rb[:, None] * N + o1[None, :],
                (x1 * rrms[:, None] * w1[None, :]).to(y_ptr.dtype.element_ty),
                mask=rm[:, None] & m1[None, :],
            )
        if C2 > 1:
            tl.store(
                y_ptr + rb[:, None] * N + o2[None, :],
                (x2 * rrms[:, None] * w2[None, :]).to(y_ptr.dtype.element_ty),
                mask=rm[:, None] & m2[None, :],
            )


@triton.jit(do_not_specialize=["eps"])
def _gemma_rmsnorm_wide_kernel(
    y_ptr,
    x_ptr,
    w_ptr,
    MAX_ROWS,
    eps,
    N_ROWS,
    N,
    BLOCK: tl.constexpr,
):
    # Wide rows: two passes over the row in BLOCK-wide tiles (the whole row
    # does not fit the unified buffer as a single block).
    pid = tl.program_id(0)
    y_ptr += pid * N * N_ROWS
    x_ptr += pid * N * N_ROWS
    base_row = pid * N_ROWS
    rows = min(base_row + N_ROWS, MAX_ROWS) - base_row
    for row_off in tl.range(0, rows, 1):
        b = row_off * N
        ssq = 0.0
        for start in tl.range(0, N, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            x = tl.load(x_ptr + b + offs, mask=offs < N, other=0.0).to(tl.float32)
            ssq += tl.sum(x * x)
        rrms = tl.rsqrt(ssq / N + eps)
        for start in tl.range(0, N, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            m = offs < N
            x = tl.load(x_ptr + b + offs, mask=m, other=0.0).to(tl.float32)
            w = (1.0 + tl.load(w_ptr + offs, mask=m, other=0.0).to(tl.float32)) * 1.0
            tl.store(
                y_ptr + b + offs, (x * rrms * w).to(y_ptr.dtype.element_ty), mask=m
            )


def gemma_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    logger.debug("GEMS_ASCEND GEMMA_RMSNORM [shape]: %s", tuple(x.shape))

    if x.ndim == 0:
        raise ValueError("gemma_rmsnorm expects an input with at least one dimension")
    orig_shape = x.shape
    N = orig_shape[-1]
    if w.ndim != 1 or w.shape[0] != N:
        raise ValueError(f"weight must have shape ({N},), got {tuple(w.shape)}")

    if any(dim == 0 for dim in orig_shape):
        return torch.empty_like(x)

    x = x.view(-1, N)
    M = x.shape[0]
    out = torch.empty_like(x)

    cores = _get_core_num()
    CORES = 24 if cores is None else cores

    # Tuned launch parameters (see the notes at the top of this file).
    if N <= _CHUNKED_N_MAX:
        C0, C1, C2, exact = _chunk_params(N)
        if M <= CORES:
            # one program per row: no weight reuse to amortize
            grid, n_rows, num_warps = M, 1, 8
            kernel = _gemma_rmsnorm_kernel
        else:
            target = 2 * CORES if M >= _FINE_GRID_M else CORES
            n_rows = triton.cdiv(M, target)
            grid = triton.cdiv(M, n_rows)
            if exact and N <= _BM2_N_MAX:
                kernel = _gemma_rmsnorm_bm2_kernel
                num_warps = 8
            else:
                kernel = _gemma_rmsnorm_kernel
                num_warps = 4
        kernel[(grid,)](
            out,
            x,
            w,
            M,
            eps,
            N_ROWS=n_rows,
            N=N,
            C0=C0,
            C1=C1,
            C2=C2,
            EXACT0=exact,
            num_warps=num_warps,
            multibuffer=True,
            limit_auto_multi_buffer_only_for_local_buffer=False,
            limit_auto_multi_buffer_of_local_buffer="no-limit",
        )
    else:
        if M <= CORES:
            n_rows, grid = 1, M
        else:
            n_rows = triton.cdiv(M, CORES)
            grid = triton.cdiv(M, n_rows)
        # BLOCK tuned per width: 8192 covers 16384 in two exact steps;
        # smaller wide rows (12288) prefer 4096
        BLOCK = 8192 if N >= 16384 else 4096
        _gemma_rmsnorm_wide_kernel[(grid,)](
            out,
            x,
            w,
            M,
            eps,
            N_ROWS=n_rows,
            N=N,
            BLOCK=BLOCK,
            num_warps=4,
            multibuffer=True,
            limit_auto_multi_buffer_only_for_local_buffer=False,
            limit_auto_multi_buffer_of_local_buffer="no-limit",
        )

    return out.view(orig_shape)
