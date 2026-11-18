# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 FlagOS Contributors
"""Signed-square-root gate for normalized Hy4 Flash Engram key/query rows."""

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.hy4_flash_engram_lookup_norm import (
    _ENGRAM_DTYPES,
    _check_engram_tensors,
)
from flaggems_vllm.utils import libentry, libtuner


@libentry()
@libtuner(configs=runtime.get_tuned_config("hy4_flash_engram_row"), key=["N"])
@triton.jit
def _engram_gate(
    Key,
    Query,
    Out,
    N: tl.constexpr,
    K_S0: tl.constexpr,
    K_S1: tl.constexpr,
    Q_S0: tl.constexpr,
    Q_S1: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    k = tl.load(Key + row * K_S0 + col * K_S1, col < N, other=0).to(tl.float32)
    q = tl.load(Query + row * Q_S0 + col * Q_S1, col < N, other=0).to(tl.float32)
    # Keep the original multiply, reduction, scale and sqrt activation casts.
    dtype: tl.constexpr = Key.dtype.element_ty
    p = (k * q).to(dtype).to(tl.float32)
    dot = tl.sum(p, 0).to(dtype).to(tl.float32)
    score = (dot * SCALE).to(dtype).to(tl.float32)
    min_value = tl.full((), 1e-6, dtype).to(tl.float32)
    root = tl.sqrt(tl.maximum(tl.abs(score), min_value)).to(dtype).to(tl.float32)
    signed = tl.where(score < 0, -root, tl.where(score > 0, root, 0))
    sigmoid = tl.where(score != score, score, tl.sigmoid(signed))
    tl.store(Out + row, sigmoid)


def hy4_flash_engram_gate(key, query):
    """Return [tokens] gates from already normalized [tokens, hidden] inputs."""
    _check_engram_tensors(key, query)
    if (
        key.ndim != 2
        or query.shape != key.shape
        or query.dtype != key.dtype
        or key.dtype not in _ENGRAM_DTYPES
        or not 0 < key.shape[-1] <= 8192
    ):
        raise ValueError("Engram gate needs matching floating [tokens, hidden] inputs")
    m, n = key.shape
    out = torch.empty((m,), dtype=key.dtype, device=key.device)
    if m:
        _engram_gate[(m,)](
            key,
            query,
            out,
            n,
            *key.stride(),
            *query.stride(),
            n**-0.5,
            triton.next_power_of_2(n),
            enable_fp_fusion=False,
        )
    return out
