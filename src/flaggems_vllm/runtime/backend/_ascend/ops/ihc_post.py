# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Ascend iHC post with a bounded launch grid for every autotune candidate."""

import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

_MAX_CORE_DIM = 65535


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("triton_ihc_post"),
    key=["NUM_TOKENS", "HC_MULT", "HIDDEN_SIZE", "DTYPE_TAG"],
    warmup=5,
    rep=10,
    use_cuda_graph=False,
)
@triton.jit
def _ihc_post_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    output_ptr,
    NUM_TOKENS: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    DTYPE_TAG: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TOKEN_MAJOR: tl.constexpr,
):
    hidden_blocks: tl.constexpr = tl.cdiv(HIDDEN_SIZE, BLOCK_D)
    channels = tl.arange(0, HC_POW2)
    lanes = tl.arange(0, BLOCK_D)
    start = tl.program_id(0).to(tl.int64)
    for tile in range(start, NUM_TOKENS * hidden_blocks, tl.num_programs(0)):
        if TOKEN_MAJOR:
            token = tile % NUM_TOKENS
            block = tile // NUM_TOKENS
        else:
            token = tile // hidden_blocks
            block = tile % hidden_blocks
        hidden = block * BLOCK_D + lanes
        row = token * HC_MULT + channels
        valid = (channels[:, None] < HC_MULT) & (hidden[None, :] < HIDDEN_SIZE)
        offset = row[:, None] * HIDDEN_SIZE + hidden[None, :]
        x = tl.load(x_ptr + token * HIDDEN_SIZE + hidden, hidden < HIDDEN_SIZE, 0)
        gate = tl.load(post_ptr + row, channels < HC_MULT, 0)
        residual = tl.load(residual_ptr + offset, valid, 0)
        value = gate[:, None] * x[None, :].to(tl.float32) + residual.to(tl.float32)
        tl.store(output_ptr + offset, value, valid)


def _launch_ihc_post(x, residual, post, output):
    tokens, channels, hidden = residual.shape
    props = runtime.torch_device_fn.get_device_properties(x.device)
    # Some devices omit a separate vector core count. The launch limit still
    # bounds the grid, and the kernel covers all tiles at either grid size.
    vector_cores = getattr(props, "vector_core_num", 0) or _MAX_CORE_DIM

    def grid(meta):
        tiles = tokens * triton.cdiv(hidden, meta["BLOCK_D"])
        return (min(tiles, vector_cores, _MAX_CORE_DIM),)

    _ihc_post_kernel[grid](
        x,
        residual,
        post,
        output,
        NUM_TOKENS=tokens,
        HC_MULT=channels,
        HC_POW2=triton.next_power_of_2(channels),
        HIDDEN_SIZE=hidden,
        DTYPE_TAG=str(x.dtype),
        enable_fp_fusion=False,
    )
