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

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._thead.ops.fused_moe import (
    _supports_int8_moe,
    invoke_fused_moe_triton_kernel,
)
from flaggems_vllm.utils import libentry, libtuner

SMALL_ROW_LIMIT = 64
SHORT_REDUCTION_LIMIT = 1024
GEMV_ROW_LIMIT = 128


@triton.jit
def packed_gemv_step(
    a_ptrs, b_ptrs, acc, mask_k, mask_n, CACHE_B: tl.constexpr, EVICT_B: tl.constexpr
):
    a = tl.load(a_ptrs, mask_k, other=0)
    b = tl.load(
        b_ptrs,
        mask_n[:, None] & mask_k[None, :],
        other=0,
        cache_modifier=CACHE_B,
        eviction_policy=EVICT_B,
    )
    return tl.inline_asm_elementwise(
        "dp4a.s32.s32 $0, $1, $2, $3;",
        constraints="=r,r,r,r",
        args=[a[None, :], b, acc],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def grouped_gemv_int8_kernel(
    A,
    As,
    B,
    Bs,
    C,
    RowExperts,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K4: tl.constexpr,
    STAGES: tl.constexpr,
    WARPS: tl.constexpr,
    TLE_PIPELINE: tl.constexpr,
    CACHE_B: tl.constexpr = "",
    EVICT_B: tl.constexpr = "",
):
    tl.static_assert(K % 4 == 0 and WARPS > 0)
    row = tl.program_id(1).to(tl.int64)
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    expert = tl.load(RowExperts + row).to(tl.int64)
    words = tl.arange(0, BLOCK_K4)
    a_words = A.to(tl.pointer_type(tl.int32)) + row * (K // 4) + words
    b_words = B.to(tl.pointer_type(tl.int32)) + expert * N * (K // 4)
    b_words += cols[:, None] * (K // 4) + words[None, :]
    acc = tl.zeros((BLOCK_N, BLOCK_K4), tl.int32)
    if TLE_PIPELINE:
        for tile in tle.gpu.pipeline(0, tl.cdiv(K // 4, BLOCK_K4), num_stages=STAGES):
            offset = tile * BLOCK_K4
            acc = packed_gemv_step(
                a_words + offset,
                b_words + offset,
                acc,
                words + offset < K // 4,
                cols < N,
                CACHE_B,
                EVICT_B,
            )
    else:
        for tile in range(tl.cdiv(K // 4, BLOCK_K4)):
            offset = tile * BLOCK_K4
            acc = packed_gemv_step(
                a_words + offset,
                b_words + offset,
                acc,
                words + offset < K // 4,
                cols < N,
                CACHE_B,
                EVICT_B,
            )
    dot = tl.sum(acc, 1).to(tl.float32)
    scale_a = tl.load(As + row)
    scale_b = tl.load(Bs + expert * N + cols, cols < N, other=0)
    tl.store(C + row * N + cols, dot * scale_a * scale_b, cols < N)


@triton.jit
def grouped_dot_step(
    a_ptr,
    b_ptr,
    acc,
    BLOCK_K: tl.constexpr,
    SWAP: tl.constexpr,
    ASYNC_A: tl.constexpr,
    ASYNC_B: tl.constexpr,
):
    a = tle.load(a_ptr, boundary_check=(0, 1), padding_option="zero", is_async=ASYNC_A)
    b = tle.load(b_ptr, boundary_check=(0, 1), padding_option="zero", is_async=ASYNC_B)
    if SWAP:
        acc = tl.dot(b, tl.trans(a), acc, out_dtype=tl.int32)
    else:
        acc = tl.dot(a, tl.trans(b), acc, out_dtype=tl.int32)
    return tl.advance(a_ptr, (0, BLOCK_K)), tl.advance(b_ptr, (0, BLOCK_K)), acc


@triton.jit
def grouped_gemm_compute_tile(
    A,
    As,
    B,
    Bs,
    C,
    SortedRows,
    row_offset,
    first_row,
    expert,
    tile_n,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
    WARPS: tl.constexpr,
    SWAP: tl.constexpr,
    TLE_PIPELINE: tl.constexpr,
    NAIVE: tl.constexpr,
    ASYNC_A: tl.constexpr,
    ASYNC_B: tl.constexpr,
):
    lanes = tl.arange(0, BLOCK_M)
    if NAIVE:
        rows = tl.where(lanes == 0, first_row, M)
    else:
        rows = tl.load(SortedRows + row_offset + lanes)
    # Extra rows in the physical block are ignored by the output mask. The
    # descriptor's full allocation extent keeps the final group in bounds.
    a_ptr = tl.make_block_ptr(
        A,
        shape=(M, K),
        strides=(K, 1),
        offsets=(first_row, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_ptr = tl.make_block_ptr(
        B + expert * N * K,
        shape=(N, K),
        strides=(K, 1),
        offsets=(tile_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )
    if SWAP:
        acc = tl.zeros((BLOCK_N, BLOCK_M), tl.int32)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.int32)
    if TLE_PIPELINE:
        for _ in tle.gpu.pipeline(0, tl.cdiv(K, BLOCK_K), num_stages=STAGES):
            a_ptr, b_ptr, acc = grouped_dot_step(
                a_ptr, b_ptr, acc, BLOCK_K, SWAP, ASYNC_A, ASYNC_B
            )
    else:
        for _ in range(tl.cdiv(K, BLOCK_K)):
            a_ptr, b_ptr, acc = grouped_dot_step(
                a_ptr, b_ptr, acc, BLOCK_K, SWAP, ASYNC_A, ASYNC_B
            )
    if SWAP:
        acc = tl.trans(acc)
    cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    a_scale = tl.load(As + rows, rows < M, other=0)
    b_scale = tl.load(Bs + expert * N + cols, cols < N, other=0)
    value = acc.to(tl.float32) * a_scale[:, None] * b_scale[None, :]
    tl.store(
        C + rows[:, None].to(tl.int64) * N + cols[None, :],
        value,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


@triton.jit
def grouped_gemm_tiled_kernel(
    A,
    As,
    B,
    Bs,
    C,
    RowExperts,
    Rows,
    SortedRows,
    BlockExperts,
    PaddedRows,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    STAGES: tl.constexpr,
    WARPS: tl.constexpr,
    SWAP: tl.constexpr,
    TLE_PIPELINE: tl.constexpr,
    NAIVE: tl.constexpr,
    GROUP_M: tl.constexpr = 0,
    ADAPTIVE_M: tl.constexpr = False,
    ASYNC_A: tl.constexpr = False,
    ASYNC_B: tl.constexpr = False,
    SCALAR_PROBE: tl.constexpr = False,
):
    tl.static_assert(WARPS > 0)
    if GROUP_M == 0:
        tile_m = tl.program_id(0)
        tile_n = tl.program_id(1)
        if tile_m * BLOCK_M >= tl.load(PaddedRows):
            return
    else:
        pid = tl.program_id(0)
        num_n = tl.cdiv(N, BLOCK_N)
        tile_m = pid // (GROUP_M * num_n) * GROUP_M + pid % GROUP_M
        tile_n = pid // GROUP_M % num_n
        if tile_m * BLOCK_M >= tl.load(PaddedRows):
            return
    row_offset = tile_m * BLOCK_M
    if NAIVE:
        first_row = tile_m
        expert = tl.load(RowExperts + tile_m).to(tl.int64)
    else:
        first_row = tl.load(SortedRows + row_offset)
        expert = tl.load(BlockExperts + tile_m).to(tl.int64)
    if ADAPTIVE_M and not NAIVE and BLOCK_M > 32:
        if SCALAR_PROBE:
            has16 = tl.load(SortedRows + row_offset + 16) < M
            has32 = tl.load(SortedRows + row_offset + 32) < M
            if BLOCK_M > 64:
                has64 = tl.load(SortedRows + row_offset + 64) < M
            else:
                has64 = False
            valid_rows = tl.where(
                has64, 128, tl.where(has32, 64, tl.where(has16, 32, 16))
            )
        else:
            rows = tl.load(SortedRows + row_offset + tl.arange(0, BLOCK_M))
            valid_rows = tl.sum((rows < M).to(tl.int32), 0)
        if valid_rows <= 16:
            grouped_gemm_compute_tile(
                A,
                As,
                B,
                Bs,
                C,
                SortedRows,
                row_offset,
                first_row,
                expert,
                tile_n,
                M,
                N,
                K,
                16,
                BLOCK_N,
                BLOCK_K,
                STAGES,
                WARPS,
                SWAP,
                TLE_PIPELINE,
                NAIVE,
                ASYNC_A,
                ASYNC_B,
            )
        elif valid_rows <= 32:
            grouped_gemm_compute_tile(
                A,
                As,
                B,
                Bs,
                C,
                SortedRows,
                row_offset,
                first_row,
                expert,
                tile_n,
                M,
                N,
                K,
                32,
                BLOCK_N,
                BLOCK_K,
                STAGES,
                WARPS,
                SWAP,
                TLE_PIPELINE,
                NAIVE,
                ASYNC_A,
                ASYNC_B,
            )
        elif BLOCK_M > 64 and valid_rows <= 64:
            grouped_gemm_compute_tile(
                A,
                As,
                B,
                Bs,
                C,
                SortedRows,
                row_offset,
                first_row,
                expert,
                tile_n,
                M,
                N,
                K,
                64,
                BLOCK_N,
                BLOCK_K,
                STAGES,
                WARPS,
                SWAP,
                TLE_PIPELINE,
                NAIVE,
                ASYNC_A,
                ASYNC_B,
            )
        else:
            grouped_gemm_compute_tile(
                A,
                As,
                B,
                Bs,
                C,
                SortedRows,
                row_offset,
                first_row,
                expert,
                tile_n,
                M,
                N,
                K,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                STAGES,
                WARPS,
                SWAP,
                TLE_PIPELINE,
                NAIVE,
                ASYNC_A,
                ASYNC_B,
            )
    else:
        grouped_gemm_compute_tile(
            A,
            As,
            B,
            Bs,
            C,
            SortedRows,
            row_offset,
            first_row,
            expert,
            tile_n,
            M,
            N,
            K,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            STAGES,
            WARPS,
            SWAP,
            TLE_PIPELINE,
            NAIVE,
            ASYNC_A,
            ASYNC_B,
        )


@libentry()
@triton.jit
def grouped_gemm_metadata(
    RowExperts,
    Rows,
    SortedRows,
    BlockExperts,
    PaddedRows,
    M: tl.constexpr,
    E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    NAIVE: tl.constexpr = False,
):
    if NAIVE:
        tl.store(PaddedRows, M * BLOCK_M)
    else:
        expert = tl.program_id(0)
        groups = tl.arange(0, BLOCK_E)
        counts = tl.load(Rows + groups, groups < E, other=0)
        padded_counts = tl.cdiv(counts, BLOCK_M) * BLOCK_M
        start = tl.sum(tl.where(groups < expert, counts, 0), 0)
        padded_start = tl.sum(tl.where(groups < expert, padded_counts, 0), 0)
        count = tl.load(Rows + expert)
        expert_id = tl.load(RowExperts + start, count > 0, other=-1)
        if expert == 0:
            tl.store(PaddedRows, tl.sum(padded_counts, 0))
        lanes = tl.arange(0, BLOCK_M)
        for block in range(tl.cdiv(count, BLOCK_M)):
            offsets = block * BLOCK_M + lanes
            rows = tl.where(offsets < count, start + offsets, M)
            tl.store(SortedRows + padded_start + offsets, rows)
            tl.store(BlockExperts + padded_start // BLOCK_M + block, expert_id)


def prune_grouped_configs(configs, named_args, **kwargs):
    m, n, k = named_args["M"], named_args["N"], named_args["K"]
    return [
        config
        for config in configs
        if (not config.kwargs["NAIVE"] or m <= 512)
        and (
            not (config.kwargs["ASYNC_A"] or config.kwargs["ASYNC_B"])
            or (n % config.kwargs["BLOCK_N"] == 0 and k % config.kwargs["BLOCK_K"] == 0)
        )
    ]


grouped_gemm_configs = runtime.get_tuned_config("grouped_gemm_w8a8_gemm")
MAX_GROUPED_BLOCK_M = max(config.kwargs["BLOCK_M"] for config in grouped_gemm_configs)
MIN_GROUPED_BLOCK_M = min(config.kwargs["BLOCK_M"] for config in grouped_gemm_configs)

grouped_gemm_tuner = libtuner(
    configs=grouped_gemm_configs,
    key=["M", "N", "K", "E"],
    prune_configs_by={"early_config_prune": prune_grouped_configs},
    warmup=5,
    rep=20,
)(grouped_gemm_tiled_kernel)
autotuned_grouped_gemm = libentry()(grouped_gemm_tuner)
autotuned_grouped_gemv = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("grouped_gemm_w8a8_gemv"),
        key=["M", "N", "K", "E"],
        warmup=5,
        rep=20,
    )(grouped_gemv_int8_kernel)
)


def grouped_gemm_w8a8(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
    m_indices: torch.Tensor,
    m_rows: torch.Tensor,
) -> None:
    """Write per-channel INT8 grouped NT GEMM into the supplied BF16 output.

    Rows must be contiguous groups in ascending expert order. m_rows contains
    nonnegative lengths summing to M; m_indices identifies each row's expert.
    These device-value preconditions are supplied by the caller without a host
    synchronization. Only routing metadata is padded; A, B and C stay unpadded.
    """
    if A.ndim != 2 or B.ndim != 3:
        raise ValueError("A must be [M,K] and B must be [E,N,K]")
    M, K = A.shape
    E, N, weight_k = B.shape
    tensors = (A, As, B, Bs, C, m_indices, m_rows)
    expected_dtypes = (
        torch.int8,
        torch.float32,
        torch.int8,
        torch.float32,
        torch.bfloat16,
        torch.int32,
        torch.int32,
    )
    if any(t.dtype != dtype for t, dtype in zip(tensors, expected_dtypes)):
        raise NotImplementedError(
            "requires INT8 A/B, FP32 scales, BF16 C, INT32 indices"
        )
    if A.device.type != "cuda" or any(t.device != A.device for t in tensors):
        raise ValueError("all tensors must be on the same PPU device")
    if not all(t.is_contiguous() for t in tensors):
        raise NotImplementedError("only contiguous tensors are supported")
    expected_shapes = ((M, 1), (E, N, 1), (M, N), (M,), (E,))
    if weight_k != K or any(
        tuple(t.shape) != shape
        for t, shape in zip((As, Bs, C, m_indices, m_rows), expected_shapes)
    ):
        raise ValueError("incompatible scales, output or expert metadata shapes")
    if K > 131071 or (M > 0 and E == 0):
        raise ValueError("requires K <= 131071 and experts for nonempty input")
    c_start = C.data_ptr()
    c_end = c_start + C.numel() * C.element_size()
    for tensor in (A, As, B, Bs, m_indices, m_rows):
        start = tensor.data_ptr()
        end = start + tensor.numel() * tensor.element_size()
        if c_start < end and start < c_end:
            raise ValueError("C must not overlap an input")
    if M == 0 or N == 0:
        return
    if not _supports_int8_moe(A.device.index):
        raise NotImplementedError("grouped INT8 GEMM requires PPU-ZW810E")

    if (
        M <= GEMV_ROW_LIMIT
        and K > 0
        and K % 4 == 0
        and A.data_ptr() % 4 == 0
        and B.data_ptr() % 4 == 0
    ):
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), M)
        with torch_device_fn.device(A.device):
            autotuned_grouped_gemv[grid](A, As, B, Bs, C, m_indices, M, N, K, E)
        return

    if (
        K > 0
        and K % 128 == 0
        and N % 64 == 0
        and A.data_ptr() % 128 == 0
        and B.data_ptr() % 128 == 0
    ):
        sorted_rows = torch.empty(
            (M + E * (MAX_GROUPED_BLOCK_M - 1),), dtype=torch.int32, device=A.device
        )
        block_experts = torch.empty(
            (triton.cdiv(M, MIN_GROUPED_BLOCK_M) + E,),
            dtype=torch.int32,
            device=A.device,
        )
        padded_rows = torch.empty((1,), dtype=torch.int32, device=A.device)

        def _grid(meta):
            # Grid callbacks retain the selected tile size on cached launches and tuning trials.
            grouped_gemm_metadata[(1 if meta["NAIVE"] else E,)](
                m_indices,
                m_rows,
                sorted_rows,
                block_experts,
                padded_rows,
                M,
                E,
                BLOCK_M=meta["BLOCK_M"],
                BLOCK_E=triton.next_power_of_2(E),
                NAIVE=meta["NAIVE"],
                num_warps=4,
            )
            tiles_m = M if meta["NAIVE"] else triton.cdiv(M, meta["BLOCK_M"]) + E - 1
            tiles_n = triton.cdiv(N, meta["BLOCK_N"])
            if meta["GROUP_M"] == 0:
                return tiles_m, tiles_n
            else:
                # Pad the last group to keep the kernel's index divisions constant.
                tiles_m = triton.cdiv(tiles_m, meta["GROUP_M"]) * meta["GROUP_M"]
                return (tiles_m * tiles_n,)

        with torch_device_fn.device(A.device):
            autotuned_grouped_gemm[_grid](
                A,
                As,
                B,
                Bs,
                C,
                m_indices,
                m_rows,
                sorted_rows,
                block_experts,
                padded_rows,
                M,
                N,
                K,
                E,
                triton.next_power_of_2(E),
            )
        return

    block_m = 16 if M <= 64 * E else 64
    # Larger reductions and batches benefit from reusing weights across grouped rows.
    use_naive = M <= min(E, SMALL_ROW_LIMIT) and K <= SHORT_REDUCTION_LIMIT
    if use_naive:
        sorted_rows = None
        block_experts = m_indices
    else:
        capacity = triton.cdiv(M + E * (block_m - 1), block_m) * block_m
        sorted_rows = torch.empty((capacity,), dtype=torch.int32, device=A.device)
        block_experts = torch.empty(
            (capacity // block_m,), dtype=torch.int32, device=A.device
        )
    padded_rows = torch.empty((1,), dtype=torch.int32, device=A.device)
    config = {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "INT8_ACCUMULATOR": True,
        "num_warps": 4,
        "num_stages": 2,
    }
    with torch_device_fn.device(A.device):
        grouped_gemm_metadata[(1 if use_naive else E,)](
            m_indices,
            m_rows,
            sorted_rows,
            block_experts,
            padded_rows,
            M,
            E,
            BLOCK_M=block_m,
            BLOCK_E=triton.next_power_of_2(E),
            NAIVE=use_naive,
            num_warps=4,
        )
        invoke_fused_moe_triton_kernel(
            A,
            B,
            C.view(1, M, N),
            As,
            Bs,
            None,
            sorted_rows,
            block_experts,
            padded_rows,
            False,
            1,
            config,
            tl.bfloat16,
            use_int8_w8a8=True,
            per_channel_quant=True,
        )
