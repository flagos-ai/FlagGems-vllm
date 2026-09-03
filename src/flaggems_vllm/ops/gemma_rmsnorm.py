import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


_gemma_rmsnorm_may_2d_configs = [
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


@triton.autotune(_gemma_rmsnorm_may_2d_configs, key=["M", "N"])
@triton.jit
def _gemma_rmsnorm_may_2d_kernel(
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
        offs = m * N + n_offs[None, :]

        n_mask = n_offs < N
        x = tl.load(x_ptr + offs, mask=n_mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + n_offs, mask=n_mask, other=0.0).to(tl.float32)

        inv_rms = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / N + eps)
        y = x * inv_rms * (1.0 + w)
        tl.store(out_ptr + offs, y, mask=n_mask)


def gemma_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    logger.debug("GEMS GEMMA_RMSNORM")
    x = x.contiguous()
    w = w.contiguous()
    orig_shape = x.shape
    N = orig_shape[-1]
    x = x.reshape(-1, N)
    M = x.shape[0]
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    _gemma_rmsnorm_may_2d_kernel[grid](
        x, w, out, M, N, eps, BLOCK_N=triton.next_power_of_2(N)
    )
    return out.reshape(orig_shape)
