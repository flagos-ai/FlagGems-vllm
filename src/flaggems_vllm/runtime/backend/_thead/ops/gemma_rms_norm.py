import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


_gemma_rms_norm_may_2d_configs = [
    triton.Config(kwargs={"BLOCK_M": 1}, num_warps=1),
    triton.Config(kwargs={"BLOCK_M": 1}, num_warps=2),
    triton.Config(kwargs={"BLOCK_M": 1}, num_warps=4),
    triton.Config(kwargs={"BLOCK_M": 1}, num_warps=8),
    triton.Config(kwargs={"BLOCK_M": 1}, num_warps=16),
    triton.Config(kwargs={"BLOCK_M": 2}, num_warps=1),
    triton.Config(kwargs={"BLOCK_M": 2}, num_warps=2),
    triton.Config(kwargs={"BLOCK_M": 2}, num_warps=4),
    triton.Config(kwargs={"BLOCK_M": 2}, num_warps=8),
    triton.Config(kwargs={"BLOCK_M": 2}, num_warps=16),
    triton.Config(kwargs={"BLOCK_M": 4}, num_warps=4),
    triton.Config(kwargs={"BLOCK_M": 4}, num_warps=8),
    triton.Config(kwargs={"BLOCK_M": 4}, num_warps=16),
    triton.Config(kwargs={"BLOCK_M": 8}, num_warps=4),
    triton.Config(kwargs={"BLOCK_M": 8}, num_warps=8),
    triton.Config(kwargs={"BLOCK_M": 8}, num_warps=16),
]

_gemma_rms_norm_loop_configs = [
    triton.Config(kwargs={"TILE_N": tile_n}, num_warps=num_warps)
    for tile_n in [512, 1024, 2048, 4096, 8192, 16384]
    for num_warps in [4, 8, 16]
]


@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


@triton.autotune(_gemma_rms_norm_may_2d_configs, key=["M", "N"])
@triton.jit
def _gemma_rms_norm_may_2d_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if BLOCK_M != 1:
        m_block_id = tl.program_id(0)
        m_offs = m_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        n_offs = tl.arange(0, BLOCK_N)
        offs = m_offs[:, None] * N + n_offs[None, :]

        m_mask = m_offs < M
        n_mask = n_offs < N
        mask = m_mask[:, None] & n_mask[None, :]

        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + n_offs, mask=n_mask, other=0.0).to(tl.float32)

        inv_rms = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / N + eps)
        y = x * inv_rms[:, None] * (1.0 + w[None, :])
        tl.store(out_ptr + offs, y, mask=mask)
    else:
        m = tl.program_id(0)
        n_offs = tl.arange(0, BLOCK_N)
        offs = m * N + n_offs

        n_mask = n_offs < N
        x = tl.load(x_ptr + offs, mask=n_mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + n_offs, mask=n_mask, other=0.0).to(tl.float32)

        inv_rms = 1.0 / tl.sqrt(tl.sum(x * x, axis=-1) / N + eps)
        y = x * inv_rms * (1.0 + w)
        tl.store(out_ptr + offs, y, mask=n_mask)


@triton.autotune(_gemma_rms_norm_loop_configs, key=["M", "N"])
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

    x.is_contiguous()
    w.is_contiguous()

    x = x.view(-1, N)
    M = x.shape[0]
    out = torch.empty_like(x)

    if N <= 8192:
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        _gemma_rms_norm_may_2d_kernel[grid](
            x, w, out, M, N, eps, BLOCK_N=triton.next_power_of_2(N)
        )
    else:
        _gemma_rms_norm_loop_kernel[M,](out, x, w, M, N, eps)

    return out.view(orig_shape)
