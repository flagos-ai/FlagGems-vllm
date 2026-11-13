# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Qwen4 HC down projection with the production BF16 rounding boundary."""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@triton.jit
def _hc_down_silu_epilogue(acc, columns, RANK: tl.constexpr, HC: tl.constexpr):
    # The materialized down projection is BF16 before the FP32 SiLU stage.
    rounded = acc.to(tl.bfloat16).to(tl.float32)
    z = rounded * (1.0 / HC)
    return tl.where(columns < RANK, z * tl.sigmoid(z), rounded)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hc_down_silu_gemv"),
    key=["SHAPE"],
    strategy=str,
    warmup=5,
    rep=20,
    use_cuda_graph=False,
)
@triton.jit
def _hc_down_silu_gemv(
    X,
    W,
    O,
    SHAPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    K: tl.constexpr = SHAPE[1]
    N: tl.constexpr = SHAPE[2]
    RANK: tl.constexpr = SHAPE[3]
    HC: tl.constexpr = SHAPE[4]
    SX: tl.constexpr = SHAPE[5]
    SW: tl.constexpr = SHAPE[6]
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = tl.arange(0, BLOCK_K)
    partial = tl.zeros((BLOCK_N, BLOCK_K), tl.float32)
    # Bound the reduction tile even for long K. All partials stay in FP32;
    # no BF16 conversion occurs until the entire dot product is reduced.
    for start in range(tl.cdiv(K, BLOCK_K)):
        k = start * BLOCK_K + offsets
        x = tl.load(X + row * SX + k, k < K, other=0).to(tl.float32)
        weight = tl.load(
            W + columns[:, None] * SW + k[None, :],
            (columns[:, None] < RANK + HC) & (k[None, :] < K),
            other=0,
        ).to(tl.float32)
        partial += weight * x[None, :]
    acc = tl.sum(partial, 1)
    value = _hc_down_silu_epilogue(acc, columns, RANK, HC)
    tl.store(O + row * N + columns, value, columns < RANK + HC)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("hc_down_silu_gemm"),
    key=["SHAPE"],
    strategy=str,
    warmup=5,
    rep=20,
    use_cuda_graph=False,
)
@triton.jit
def _hc_down_silu_gemm(
    X,
    W,
    O,
    SHAPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    M: tl.constexpr = SHAPE[0]
    K: tl.constexpr = SHAPE[1]
    N: tl.constexpr = SHAPE[2]
    RANK: tl.constexpr = SHAPE[3]
    HC: tl.constexpr = SHAPE[4]
    SX: tl.constexpr = SHAPE[5]
    SW: tl.constexpr = SHAPE[6]
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start in range(tl.cdiv(K, BLOCK_K)):
        k = start * BLOCK_K + offsets
        x = tl.load(
            X + rows[:, None] * SX + k[None, :],
            (rows[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        weight = tl.load(
            W + columns[None, :] * SW + k[:, None],
            (columns[None, :] < RANK + HC) & (k[:, None] < K),
            other=0,
        )
        acc = tl.dot(x, weight, acc)
    value = _hc_down_silu_epilogue(acc, columns[None, :], RANK, HC)
    tl.store(
        O + rows[:, None] * N + columns[None, :],
        value,
        (rows[:, None] < M) & (columns[None, :] < RANK + HC),
    )


@libentry()
@libtuner(
    # Current MACA vector loads fail on this large-span path. Scalar K tiles
    # retain the same FP32 reduction and BF16 epilogue without that lowering.
    configs=runtime.get_tuned_config(
        "hc_down_silu_wide_gemv"
        if runtime.device.vendor_name == "metax"
        else "hc_down_silu_gemv"
    ),
    key=["SHAPE"],
    strategy=str,
    warmup=5,
    rep=20,
    use_cuda_graph=False,
)
@triton.jit
def _hc_down_silu_wide_gemv(
    X,
    W,
    O,
    SHAPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Large read-only row strides need widening before multiplication, not
    # after an i32 product has already overflowed. Keep the usual path in i32.
    K: tl.constexpr = SHAPE[1]
    N: tl.constexpr = SHAPE[2]
    RANK: tl.constexpr = SHAPE[3]
    HC: tl.constexpr = SHAPE[4]
    SX: tl.constexpr = SHAPE[5]
    SW: tl.constexpr = SHAPE[6]
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1).to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N).to(
        tl.int64
    )
    offsets = tl.arange(0, BLOCK_K).to(tl.int64)
    partial = tl.zeros((BLOCK_N, BLOCK_K), tl.float32)
    for start in range(tl.cdiv(K, BLOCK_K)):
        k = start.to(tl.int64) * BLOCK_K + offsets
        x = tl.load(X + row * SX + k, k < K, other=0).to(tl.float32)
        weight = tl.load(
            W + columns[:, None] * SW + k[None, :],
            (columns[:, None] < RANK + HC) & (k[None, :] < K),
            other=0,
        ).to(tl.float32)
        partial += weight * x[None, :]
    acc = tl.sum(partial, 1)
    value = _hc_down_silu_epilogue(acc, columns, RANK, HC)
    tl.store(O + row * N + columns, value, columns < RANK + HC)


def hc_down_silu(
    x: torch.Tensor, weight: torch.Tensor, rank: int, hc: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return BF16 HC activations and injection logits from BF16 inputs.

    ``x[M, K]`` and ``weight[N, K]`` have positive K, rank and hc, with
    ``N >= rank + hc``. The FP32 dot product rounds to BF16 before the
    activation applies FP32 ``silu(value / hc)``. Injection logits retain
    that rounded dot product. Inputs are read only.

    Both outputs are views into one fresh packed ``[M, N]`` allocation;
    padding columns are neither computed nor read. Unit inner stride with
    arbitrary nonnegative row strides is a candidate extension beyond the
    contiguous native workload. Empty M also preserves these packed views.
    """
    logger.debug("GEMS HC_DOWN_SILU")
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise TypeError("hc_down_silu requires tensor inputs")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("hc_down_silu requires a positive integer rank")
    if isinstance(hc, bool) or not isinstance(hc, int) or hc <= 0:
        raise ValueError("hc_down_silu requires a positive integer hc")
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("hc_down_silu requires rank-2 inputs")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("hc_down_silu supports only bfloat16 inputs")
    if x.device != weight.device:
        raise ValueError("hc_down_silu requires same-device inputs")
    if x.layout != torch.strided or weight.layout != torch.strided:
        raise ValueError("hc_down_silu requires strided inputs")
    m, k = x.shape
    n, weight_k = weight.shape
    if k <= 0 or k != weight_k:
        raise ValueError("hc_down_silu requires matching positive input widths")
    if n < rank + hc:
        raise ValueError("hc_down_silu requires weight rows >= rank + hc")
    sx, sw = x.stride(), weight.stride()
    if sx[1] != 1 or sw[1] != 1 or sx[0] < 0 or sw[0] < 0:
        raise ValueError("hc_down_silu requires unit inner and nonnegative row strides")
    if x.device.type in ("cpu", "meta"):
        raise NotImplementedError("hc_down_silu requires accelerator inputs")
    output = torch.empty((m, n), dtype=x.dtype, device=x.device)
    activation, injection_logits = output[:, :rank], output[:, rank : rank + hc]
    if m == 0:
        return activation, injection_logits
    wide = (
        max(
            (m - 1) * sx[0] + k - 1,
            (rank + hc - 1) * sw[0] + k - 1,
            m * n - 1,
        )
        > 2**31 - 1
    )
    use_gemv = m <= 4 or k < 2048
    # Measured crossovers at the model workload: H20 favors vectors through
    # M=16, while S5000 favors them throughout the fused model domain and M=49.
    # Keep these choices on the measured backends and workload.
    if k == 10240 and rank + hc == 324:
        if runtime.device.vendor_name == "nvidia":
            use_gemv = m <= 16
        elif runtime.device.vendor_name == "mthreads":
            use_gemv = m <= 49
    shape = (m, k, n, rank, hc, sx[0], sw[0], int(use_gemv))
    with runtime.torch_device_fn.device(x.device.index):
        if wide:
            _hc_down_silu_wide_gemv[
                lambda meta: (m, triton.cdiv(rank + hc, meta["BLOCK_N"]))
            ](x, weight, output, (m, k, n, rank, hc, sx[0], sw[0], 2))
        elif use_gemv:
            _hc_down_silu_gemv[
                lambda meta: (m, triton.cdiv(rank + hc, meta["BLOCK_N"]))
            ](x, weight, output, shape)
        else:
            _hc_down_silu_gemm[
                lambda meta: (
                    triton.cdiv(m, meta["BLOCK_M"]),
                    triton.cdiv(rank + hc, meta["BLOCK_N"]),
                )
            ](x, weight, output, shape)
    return activation, injection_logits
