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


# GCU400 dispatch floor is ~52 ns/CTA, so latency of fine-grained kernels is
# dominated by the program count, not the memory traffic (measured on
# (4096,1,4096) fp16: 131072 programs -> 6.6 ms, 8192 -> 430 us, 48 -> 52 us).
# Keep the mt-kernel launch at one program per SP (48) and let each program
# stride over the flattened tile space; this also keeps the grid far below
# the GCU launch limits (grid.x <= 65535, grid.y <= 255) by construction.
_MT_GRID_TARGET = 48


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
    ],
    key=["hidden_size", "topk", "ELEM_SIZE"],
)
@triton.jit
def _moe_sum_pair_kernel(
    input_ptr,
    output_ptr,
    num_tokens,
    topk: tl.constexpr,
    hidden_size,
    input_stride_token,
    input_stride_topk,
    output_stride_token,
    BLOCK_SIZE: tl.constexpr,
    ELEM_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    hidden_start = block_idx * BLOCK_SIZE
    hidden_offsets = hidden_start + tl.arange(0, BLOCK_SIZE)

    hidden_offsets = tl.max_contiguous(
        tl.multiple_of(hidden_offsets, BLOCK_SIZE), BLOCK_SIZE
    )

    hidden_mask = hidden_offsets < hidden_size
    if token_idx >= num_tokens:
        return
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    input_base = input_ptr + token_idx * input_stride_token + hidden_offsets

    for expert_idx in tl.static_range(0, topk, 2):
        expert_data0 = tl.load(
            input_base + expert_idx * input_stride_topk,
            mask=hidden_mask,
            other=0.0,
        )
        if expert_idx + 1 < topk:
            expert_data1 = tl.load(
                input_base + (expert_idx + 1) * input_stride_topk,
                mask=hidden_mask,
                other=0.0,
            )
            acc += expert_data0.to(tl.float32) + expert_data1.to(tl.float32)
        else:
            acc += expert_data0.to(tl.float32)

    output_ptr_pos = output_ptr + token_idx * output_stride_token + hidden_offsets
    tl.store(
        output_ptr_pos,
        acc.to(tl.float16) if input_ptr.dtype.element_ty == tl.float16 else acc,
        mask=hidden_mask,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 128}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_H": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_H": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_H": 1024}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_H": 2048}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_H": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_H": 8192}, num_warps=8, num_stages=3),
    ],
    key=["hidden_size", "topk", "token_bucket", "ELEM_SIZE"],
)
@triton.jit
def _moe_sum_mt_kernel(
    input_ptr,
    output_ptr,
    num_tokens,
    hidden_size: tl.constexpr,
    input_stride_token,
    input_stride_topk,
    output_stride_token,
    token_bucket,
    topk: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ELEM_SIZE: tl.constexpr,
):
    # The (hidden block, token) tile space is flattened into a single index;
    # one loop iteration handles one token row x BLOCK_H. The original layout
    # put token blocks on grid.y, which exceeds the GCU grid.y <= 255 limit
    # for large num_tokens (e.g. cdiv(4096, 8) = 512); with the launch capped
    # at _MT_GRID_TARGET programs the grid limits can never bind, and striding
    # beats one-program-per-tile anyway (see the _MT_GRID_TARGET comment).
    #
    # NOTE: kept 1-D on purpose. The GCU toolchain installed on 2026-11-07
    # failed to compile any 2-D tile access (memref_ext.slice_pad_start
    # operand-segment mismatch, see compiler_issues/vllm_moe_sum.md); the
    # 2026-11-10 toolchain realignment fixed that (md5 87b94b98), and on
    # 2026-11-11 the 2-D revisit was measured for the general path: a
    # (M, BLOCK_H) tile load lowers to per-row 1-D segments, so the memory-op
    # count does not drop and the tile form is slower than the 1-D row loop
    # (g6 acc2d M=32: 1390.6us vs 1134.5us on (1024,1024,1024) fp16); a
    # contiguous wide load + tl.reshape + tl.sum fold is pathological
    # (M=8: 218.3us vs 40.1us on (64,512,512) fp16). Do not retry these
    # forms for this op without a compiler-side fix to the fold lowering.
    NUM_H_BLOCKS: tl.constexpr = (hidden_size + BLOCK_H - 1) // BLOCK_H
    total_tiles = NUM_H_BLOCKS * num_tokens

    for flat_idx in range(tl.program_id(0), total_tiles, tl.num_programs(0)):
        block_idx = flat_idx % NUM_H_BLOCKS
        token_idx = flat_idx // NUM_H_BLOCKS

        hidden_offsets = block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
        hidden_offsets = tl.max_contiguous(
            tl.multiple_of(hidden_offsets, BLOCK_H), BLOCK_H
        )
        hidden_mask = hidden_offsets < hidden_size

        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        input_base = input_ptr + token_idx * input_stride_token + hidden_offsets
        for expert_idx in tl.static_range(topk):
            expert_data = tl.load(
                input_base + expert_idx * input_stride_topk,
                mask=hidden_mask,
                other=0.0,
            )
            acc += expert_data.to(tl.float32)

        output_ptr_pos = output_ptr + token_idx * output_stride_token + hidden_offsets
        tl.store(
            output_ptr_pos,
            acc.to(tl.float16) if input_ptr.dtype.element_ty == tl.float16 else acc,
            mask=hidden_mask,
        )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 256, "U": 16}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_H": 512, "U": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_H": 512, "U": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_H": 1024, "U": 16}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_H": 1024, "U": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_H": 2048, "U": 16}, num_warps=8, num_stages=3),
    ],
    key=["hidden_size", "topk", "ELEM_SIZE"],
)
@triton.jit
def _moe_sum_general_kernel(
    input_ptr,
    output_ptr,
    num_tokens,
    topk: tl.constexpr,
    hidden_size: tl.constexpr,
    input_stride_token,
    input_stride_topk,
    input_stride_hidden,
    output_stride_token,
    output_stride_hidden,
    BLOCK_H: tl.constexpr,
    U: tl.constexpr,
    CONTIG_H: tl.constexpr,
    ELEM_SIZE: tl.constexpr,
):
    # topk > 16 or non-contiguous input/output (fully stride-aware path).
    # The reduction over experts is a K-times strided (stride = hidden) row
    # load chain; on GCU400 the dynamic-range loop of the seed kernel leaves
    # those loads unpipelined (0.22 TB/s on (1024,1024,1024) fp16), so the
    # loop is chunked and the U loads of each chunk are unrolled statically
    # (constexpr TOPK) to put U rows in flight (playbook "topk 归约类 K 次
    # 跨步 load 的 constexpr TOPK 保序展开"; measured 2.2~8.8x on the bench
    # shapes). EXACT drops the hidden predicate when BLOCK_H divides
    # hidden_size; CONTIG_H drops the h*stride vector multiplies. The k
    # ascending accumulation order is identical to the seed kernel, so the
    # fp32 results are bitwise equal. U depth is bounded by the DSM budget
    # U*BLOCK_H*(itemsize+4)+BLOCK_H*4 <= ~448KB; U >= 48 at BLOCK_H=1024
    # regresses (measured 4626us vs 1099us on (1024,1024,1024) fp16).
    # Persistent grid-stride form (2026-11-11): the kernel used to launch one
    # CTA per (token, hidden-block) tile. GCU400 charges ~52 ns per CTA in the
    # dispatch pipe and the per-SP memory-op issue budget (~52-60 ns per 1D
    # row op, width-insensitive) does not improve with more CTAs once the 48
    # SPs are saturated, so a large grid only paid dispatch: measured
    # (1024,1024,1024) fp16 1024 CTAs -> 1195.3us vs 48 striding CTAs ->
    # 1134.5us (-5.1%); (64,512,512) fp16 64 CTAs -> 40.1us vs 36.1us
    # (-10%); fp32 shapes -1~2%. Capping the launch at min(tiles, 48)
    # programs also removes the old grid.x <= 65535 exposure for very large
    # num_tokens. The per-tile work, U chunking and k ascending order are
    # unchanged, so results stay bitwise identical.
    NUM_H_BLOCKS: tl.constexpr = (hidden_size + BLOCK_H - 1) // BLOCK_H
    total_tiles = num_tokens * NUM_H_BLOCKS

    for flat_idx in range(tl.program_id(0), total_tiles, tl.num_programs(0)):
        token_idx = flat_idx // NUM_H_BLOCKS
        block_idx = flat_idx % NUM_H_BLOCKS
        hidden_offsets = block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
        hidden_offsets = tl.max_contiguous(
            tl.multiple_of(hidden_offsets, BLOCK_H), BLOCK_H
        )
        hidden_mask = hidden_offsets < hidden_size
        EXACT: tl.constexpr = hidden_size % BLOCK_H == 0
        if CONTIG_H:
            input_base = input_ptr + token_idx * input_stride_token + hidden_offsets
            output_pos = output_ptr + token_idx * output_stride_token + hidden_offsets
        else:
            input_base = (
                input_ptr
                + token_idx * input_stride_token
                + hidden_offsets * input_stride_hidden
            )
            output_pos = (
                output_ptr
                + token_idx * output_stride_token
                + hidden_offsets * output_stride_hidden
            )
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        NUM_FULL: tl.constexpr = topk // U
        for chunk_idx in range(0, NUM_FULL):
            k_base = chunk_idx * U
            for u in tl.static_range(U):
                if EXACT:
                    expert_data = tl.load(input_base + (k_base + u) * input_stride_topk)
                else:
                    expert_data = tl.load(
                        input_base + (k_base + u) * input_stride_topk,
                        mask=hidden_mask,
                        other=0.0,
                    )
                acc += expert_data.to(tl.float32)
        if topk % U != 0:
            for expert_idx in range(NUM_FULL * U, topk):
                if EXACT:
                    expert_data = tl.load(input_base + expert_idx * input_stride_topk)
                else:
                    expert_data = tl.load(
                        input_base + expert_idx * input_stride_topk,
                        mask=hidden_mask,
                        other=0.0,
                    )
                acc += expert_data.to(tl.float32)
        if EXACT:
            tl.store(output_pos, acc.to(output_ptr.dtype.element_ty))
        else:
            tl.store(output_pos, acc.to(output_ptr.dtype.element_ty), mask=hidden_mask)


@triton.jit
def _moe_sum_copy_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    # topk == 1 fast path: summing the single expert row is the identity, so
    # the whole op degenerates to a flat contiguous copy of the
    # [num_tokens, hidden] stream with no fp32 round-trip (bitwise identical
    # to the reduction, and it sidesteps the slow fp32->fp16 store conversion
    # that made the mt path 85.7us vs 51.5us for bf16 on (4096,1,4096)).
    # One program per SP with a grid-stride loop (dispatch floor ~52 ns/CTA);
    # wide blocks win because each iteration issues one wide DMA per pointer
    # (measured on (4096,1,4096) fp16: BLOCK 256/1024/4096/16384/65536 ->
    # 1063/148/46/28/28 us).
    step = tl.num_programs(0) * BLOCK
    for offset in range(tl.program_id(0) * BLOCK, n_elements, step):
        offsets = offset + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        tl.store(
            output_ptr + offsets, tl.load(input_ptr + offsets, mask=mask), mask=mask
        )


def _token_bucket(num_tokens: int) -> int:
    if num_tokens < 1024:
        return 0
    if num_tokens < 8192:
        return 1
    return 2


def _copy_block(n_elements: int) -> int:
    # Keep ~one block per SP (48) in the launch and let each program stream
    # the rest: pick the largest power-of-two block that still yields >= 48
    # blocks, clamped to the measured 4096..65536 sweet spot (per-iteration
    # DMA width; bigger blocks cut per-iteration overhead, <48 programs lose
    # device occupancy). prev_pow2(n // 48) hits the measured optima:
    # 4096 / 16384 / 65536 elements for the 4K / 1M / 16.7M-element bench
    # shapes.
    if n_elements <= 0:
        return 1
    per_program = n_elements // _MT_GRID_TARGET
    block = 1 << max(per_program.bit_length() - 1, 0)
    return min(65536, max(4096, block))


def _check_moe_sum_inputs(input: torch.Tensor, output: torch.Tensor):
    assert input.dim() == 3, (
        f"moe_sum: input must be 3D [num_tokens, topk, hidden], "
        f"got {input.dim()}D shape {tuple(input.shape)}"
    )
    assert output.dim() == 2, (
        f"moe_sum: output must be 2D [num_tokens, hidden], "
        f"got {output.dim()}D shape {tuple(output.shape)}"
    )
    num_tokens, topk, hidden_size = input.shape
    assert topk >= 1, f"moe_sum: topk must be >= 1, got {topk}"
    assert output.shape == (num_tokens, hidden_size), (
        f"moe_sum: output shape {tuple(output.shape)} mismatch, "
        f"expected ({num_tokens}, {hidden_size}) from input shape"
    )
    assert (
        input.dtype == output.dtype
    ), f"moe_sum: dtype mismatch, input {input.dtype} vs output {output.dtype}"
    assert input.dtype in (torch.float16, torch.bfloat16, torch.float32), (
        f"moe_sum: unsupported dtype {input.dtype}, "
        f"expected float16 / bfloat16 / float32"
    )
    assert (
        input.device == output.device
    ), f"moe_sum: device mismatch, input {input.device} vs output {output.device}"


def _min_vec_elems(dtype: torch.dtype, vec_bytes: int = 16) -> int:
    return max(1, vec_bytes // torch.tensor([], dtype=dtype).element_size())


def moe_sum(
    input: torch.Tensor,
    output: torch.Tensor,
):
    logger.debug("GEMS MOE SUM")
    _check_moe_sum_inputs(input, output)
    num_tokens, topk, hidden_size = input.shape

    input_stride = input.stride()
    output_stride = output.stride()
    elem_size = input.element_size()

    vec_ok = hidden_size % _min_vec_elems(input.dtype, vec_bytes=16) == 0
    contiguous = input.is_contiguous() and output.is_contiguous()

    if num_tokens == 0 or hidden_size == 0:
        return

    if contiguous and topk == 1:
        n_elements = num_tokens * hidden_size
        block = _copy_block(n_elements)
        grid = (min(triton.cdiv(n_elements, block), _MT_GRID_TARGET),)
        _moe_sum_copy_kernel[grid](
            input,
            output,
            n_elements,
            BLOCK=block,
        )
    elif contiguous and topk <= 16 and num_tokens >= 128 and vec_ok:

        def grid(meta):
            # One program per SP; the kernel strides over the flattened tile
            # space, so the launch stays small (fast on the ~52 ns/CTA GCU
            # dispatch floor) and inside the GCU grid limits by construction.
            num_h_blocks = triton.cdiv(hidden_size, meta["BLOCK_H"])
            total_tiles = num_h_blocks * num_tokens
            return (min(total_tiles, _MT_GRID_TARGET),)

        _moe_sum_mt_kernel[grid](
            input,
            output,
            num_tokens,
            hidden_size,
            input_stride[0],
            input_stride[1],
            output_stride[0],
            _token_bucket(num_tokens),
            topk,
            ELEM_SIZE=elem_size,
        )
    elif contiguous and topk <= 16 and num_tokens < 128:
        grid = lambda meta: (num_tokens, triton.cdiv(hidden_size, meta["BLOCK_SIZE"]))
        _moe_sum_pair_kernel[grid](
            input,
            output,
            num_tokens,
            topk,
            hidden_size,
            input_stride[0],
            input_stride[1],
            output_stride[0],
            ELEM_SIZE=elem_size,
        )
    else:
        # topk > 16 or non-contiguous input/output (fully stride-aware path).
        # Persistent launch: one program per SP striding over the flattened
        # (token, hidden-block) tile space (see the kernel comment); the grid
        # stays <= _MT_GRID_TARGET by construction, so the GCU launch limits
        # (grid.x <= 65535, grid.y <= 255) can never bind.
        def grid(meta):
            num_h_blocks = triton.cdiv(hidden_size, meta["BLOCK_H"])
            total_tiles = num_tokens * num_h_blocks
            return (min(total_tiles, _MT_GRID_TARGET),)

        _moe_sum_general_kernel[grid](
            input,
            output,
            num_tokens,
            topk,
            hidden_size,
            input_stride[0],
            input_stride[1],
            input_stride[2],
            output_stride[0],
            output_stride[1],
            CONTIG_H=contiguous,
            ELEM_SIZE=elem_size,
        )
