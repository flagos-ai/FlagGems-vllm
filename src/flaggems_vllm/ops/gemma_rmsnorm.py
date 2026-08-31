import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def get_gemma_rmsnorm_configs():
    """Generate autotune configurations for gemma_rmsnorm.

    Inspired by flashinfer's adaptive hyperparameter selection:
    - Small N (≤1024): smaller BLOCK_N, more BLOCK_M parallelism
    - Medium N (1024-4096): balanced tiling
    - Large N (>4096): larger BLOCK_N, fewer warps for register pressure
    """
    configs = []

    # Small N: prioritize M parallelism
    for block_m in [1, 4, 8, 16, 32]:
        for block_n in [256, 512]:
            for num_warps in [4, 8]:
                configs.append(
                    triton.Config(
                        {"BLOCK_M": block_m, "BLOCK_N": block_n},
                        num_warps=num_warps,
                        num_stages=2,
                    )
                )

    # Medium N: balanced
    for block_m in [1, 4, 8, 16]:
        for block_n in [1024, 2048]:
            for num_warps in [4, 8, 16]:
                configs.append(
                    triton.Config(
                        {"BLOCK_M": block_m, "BLOCK_N": block_n},
                        num_warps=num_warps,
                        num_stages=2,
                    )
                )

    # Large N: prioritize N coverage, reduce M to control register pressure
    for block_m in [1, 4, 8]:
        for block_n in [4096, 8192, 16384]:
            for num_warps in [8, 16, 32]:
                configs.append(
                    triton.Config(
                        {"BLOCK_M": block_m, "BLOCK_N": block_n},
                        num_warps=num_warps,
                        num_stages=2,
                    )
                )

    return configs


@triton.autotune(
    configs=get_gemma_rmsnorm_configs(),
    key=["M", "N"],
)
@triton.jit
def gemma_rmsnorm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Process BLOCK_M rows at a time, with N chunked into BLOCK_N tiles
    # This avoids loading the entire row at once for large N
    pid_m = tl.program_id(0)

    # Row indices for this M-block
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M

    # Compute base offsets for these rows
    offsets_base = rows[:, None] * N

    # First pass: compute sum of squares across all N (in chunks)
    sum_sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    for n_chunk in range(0, N, BLOCK_N):
        chunk_cols = n_chunk + tl.arange(0, BLOCK_N)
        chunk_mask = row_mask[:, None] & (chunk_cols[None, :] < N)
        chunk_offsets = offsets_base + chunk_cols[None, :]
        x_chunk = tl.load(x_ptr + chunk_offsets, mask=chunk_mask, other=0.0).to(
            tl.float32
        )
        sum_sq += tl.sum(x_chunk * x_chunk, axis=1)

    # Compute RMS normalization factor: 1 / sqrt(mean(x^2) + eps)
    rrms = 1.0 / tl.sqrt(sum_sq / N + eps)

    # Second pass: apply normalization (in chunks)
    for n_chunk in range(0, N, BLOCK_N):
        chunk_cols = n_chunk + tl.arange(0, BLOCK_N)
        chunk_mask = row_mask[:, None] & (chunk_cols[None, :] < N)
        chunk_offsets = offsets_base + chunk_cols[None, :]

        # Load input and weight for this chunk
        x_chunk = tl.load(x_ptr + chunk_offsets, mask=chunk_mask, other=0.0).to(
            tl.float32
        )
        weight_chunk = tl.load(w_ptr + chunk_cols, mask=(chunk_cols < N), other=0.0).to(
            tl.float32
        )

        # Apply normalization: y = x * (1/RMS) * (1 + weight)
        y_chunk = x_chunk * rrms[:, None] * (1.0 + weight_chunk[None, :])

        # Store output
        tl.store(out_ptr + chunk_offsets, y_chunk, mask=chunk_mask)


def gemma_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    logger.debug("GEMS GEMMA_RMSNORM")
    x = x.contiguous()
    w = w.contiguous()

    orig_shape = x.shape
    N = orig_shape[-1]
    x = x.reshape(-1, N)
    M = x.shape[0]
    out = torch.empty_like(x)

    # Grid is 1D over M dimension only
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    gemma_rmsnorm_kernel[grid](
        x,
        w,
        out,
        M,
        N,
        eps,
    )

    return out.reshape(orig_shape)
