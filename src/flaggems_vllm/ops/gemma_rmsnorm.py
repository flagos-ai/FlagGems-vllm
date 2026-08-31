import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def get_gemma_rmsnorm_may_2d_splitn_configs():
    configs = []
    for block_m in [1, 4, 8]:
        for block_n in [128, 256, 512]:
            configs.append(
                triton.Config(
                    {"BLOCK_M": block_m, "BLOCK_N": block_n},
                    num_warps=4,
                    num_stages=2,
                )
            )

    for block_m in [1, 4, 8]:
        configs.append(
            triton.Config(
                {"BLOCK_M": block_m, "BLOCK_N": 1024},
                num_warps=8,
                num_stages=2,
            )
        )

    for block_m in [1, 4, 8]:
        configs.append(
            triton.Config(
                {"BLOCK_M": block_m, "BLOCK_N": 2048},
                num_warps=16,
                num_stages=2,
            )
        )

    for block_m in [16, 32]:
        for block_n in [256, 512]:
            configs.append(
                triton.Config(
                    {"BLOCK_M": block_m, "BLOCK_N": block_n},
                    num_warps=8,
                    num_stages=2,
                )
            )

    for block_m in [4, 8]:
        configs.append(
            triton.Config(
                {"BLOCK_M": block_m, "BLOCK_N": 1024},
                num_warps=16,
                num_stages=2,
            )
        )
    return configs


@triton.autotune(
    configs=get_gemma_rmsnorm_may_2d_splitn_configs(),
    key=["M", "N"],
)
@triton.jit
def gemma_rmsnorm_may_2d_splitn_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    eps: float,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    m_block_id = tl.program_id(0)
    m_offs = m_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M

    offsets_base = m_offs[:, None] * N
    sum_sq = tl.zeros([BLOCK_M], dtype=tl.float32)

    # pass‑1: accumulate sum‑of‑squares, split over N dimension
    for n_chunk in range(0, N, BLOCK_N):
        n_offs = n_chunk + tl.arange(0, BLOCK_N)
        mn_mask = m_mask[:, None] & (n_offs[None, :] < N)
        x = tl.load(x_ptr + offsets_base + n_offs[None, :], mask=mn_mask, other=0.0).to(
            tl.float32
        )
        xq = x * x
        sum_sq += tl.sum(xq, axis=1)

    xq_mean = sum_sq / N
    rrms = tl.rsqrt(xq_mean + eps)

    # pass‑2: normalize & apply weight, split over N dimension
    for n_chunk in range(0, N, BLOCK_N):
        n_offs = n_chunk + tl.arange(0, BLOCK_N)
        mn_mask = m_mask[:, None] & (n_offs[None, :] < N)
        n_mask = n_offs < N

        x = tl.load(x_ptr + offsets_base + n_offs[None, :], mask=mn_mask, other=0.0).to(
            tl.float32
        )
        w = tl.load(w_ptr + n_offs, mask=n_mask, other=0.0).to(tl.float32)

        out = (1.0 + w[None, :]) * x * rrms[:, None]
        tl.store(out_ptr + offsets_base + n_offs[None, :], out, mask=mn_mask)


def get_gemma_rmsnorm_may_2d_no_splitn_configs():
    configs = []
    for block_m in [1, 4, 8, 16]:
        for num_warps in [1, 2, 4, 8, 16]:
            configs.append(
                triton.Config(
                    {"BLOCK_M": block_m},
                    num_warps=num_warps,
                    num_stages=2,
                )
            )
    return configs


@triton.autotune(
    configs=get_gemma_rmsnorm_may_2d_no_splitn_configs(),
    key=["M", "N"],
)
@triton.jit
def gemma_rmsnorm_may_2d_no_splitn_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    eps: float,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if BLOCK_M == 1:
        m = tl.program_id(0)
        n_offs = tl.arange(0, BLOCK_N)
        n_mask = n_offs < N
        x = tl.load(x_ptr + m * N + n_offs, mask=n_mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + n_offs, mask=n_mask, other=0.0).to(tl.float32)
        xq = x * x
        xq_mean = tl.sum(xq, axis=-1, keep_dims=True) / N  # tensor<1xf32>
        rrms = tl.rsqrt(xq_mean + eps)
        out = (1 + w) * x * rrms
        tl.store(out_ptr + m * N + n_offs, out, mask=n_mask)
    else:
        m_block_id = tl.program_id(0)
        m = m_block_id * BLOCK_M
        n_offs = tl.arange(0, BLOCK_N)
        m_offs = m + tl.arange(0, BLOCK_M)
        n_mask = n_offs < N
        m_mask = m_offs < M
        mn_mask = n_mask[None, :] & m_mask[:, None]
        x = tl.load(
            x_ptr + m_offs[:, None] * N + n_offs[None, :], mask=mn_mask, other=0.0
        ).to(tl.float32)
        w = tl.load(w_ptr + n_offs, mask=n_mask, other=0.0).to(tl.float32)
        xq = x * x
        xq_mean = tl.sum(xq, axis=-1, keep_dims=True) / N  # tensor<BLOCK_Mx1xf32>
        rrms = tl.rsqrt(xq_mean + eps)
        out = (1 + w) * x * rrms
        tl.store(out_ptr + m_offs[:, None] * N + n_offs[None, :], out, mask=mn_mask)


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
    if M == 1:
        gemma_rmsnorm_may_2d_no_splitn_kernel[grid](
            x, w, out, M, N, eps, BLOCK_N=triton.next_power_of_2(N)
        )
    else:
        gemma_rmsnorm_may_2d_splitn_kernel[grid](
            x,
            w,
            out,
            M,
            N,
            eps,
        )
    return out.reshape(orig_shape)
