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

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_WHOLE_ROW_MAX_BLOCK_N = 16384

_WHOLE_ROW_CONFIG_M_THRESHOLD = 128
_WHOLE_ROW_MAX_N = 4096

_gemma_rms_norm_configs = [
    triton.Config({"BLOCK_M": 1}, num_warps=2),
    triton.Config({"BLOCK_M": 1}, num_warps=4),
    triton.Config({"BLOCK_M": 1}, num_warps=8),
    triton.Config({"BLOCK_M": 1}, num_warps=16),
    triton.Config({"BLOCK_M": 1}, num_warps=32),
    triton.Config({"BLOCK_M": 2}, num_warps=2),
    triton.Config({"BLOCK_M": 2}, num_warps=4),
    triton.Config({"BLOCK_M": 2}, num_warps=8),
    triton.Config({"BLOCK_M": 2}, num_warps=16),
    triton.Config({"BLOCK_M": 2}, num_warps=32),
    triton.Config({"BLOCK_M": 4}, num_warps=4),
]

_gemma_rms_norm_loop_configs = [
    triton.Config(kwargs={"TILE_N": 2048}, num_warps=4, num_stages=2),
    triton.Config(kwargs={"TILE_N": 2048}, num_warps=8, num_stages=1),
    triton.Config(kwargs={"TILE_N": 4096}, num_warps=8, num_stages=2),
    triton.Config(kwargs={"TILE_N": 4096}, num_warps=16, num_stages=2),
    triton.Config(kwargs={"TILE_N": 8192}, num_warps=16, num_stages=1),
]


@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


@triton.autotune(_gemma_rms_norm_configs, key=["M", "N"])
@triton.jit(do_not_specialize=["eps"])
def _gemma_rms_norm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    m_offs = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    offs = m_offs[:, None] * N + n_offs[None, :]
    mask = (m_offs < M)[:, None] & (n_offs < N)[None, :]

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    w = tl.load(w_ptr + n_offs, mask=n_offs < N, other=0.0)

    xf = x.to(tl.float32)
    inv_rms = 1.0 / tl.sqrt(tl.sum(xf * xf, axis=1) / N + eps)
    y = xf * inv_rms[:, None] * (1.0 + w.to(tl.float32)[None, :])
    tl.store(out_ptr + offs, y, mask=mask)


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

    if not x.is_contiguous() or not w.is_contiguous():
        raise NotImplementedError("gemma_rms_norm requires contiguous tensors")

    M = 1
    for dim in orig_shape[:-1]:
        M *= dim
    out = torch.empty_like(x)

    np2 = triton.next_power_of_2(N)
    if np2 <= _WHOLE_ROW_MAX_BLOCK_N and (
        M < _WHOLE_ROW_CONFIG_M_THRESHOLD or np2 <= _WHOLE_ROW_MAX_N
    ):
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        _gemma_rms_norm_kernel[grid](x, w, out, M, N, eps, BLOCK_N=np2)
    else:
        _gemma_rms_norm_loop_kernel[(M,)](out, x, w, M, N, eps)

    return out.view(orig_shape)
