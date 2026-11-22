# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend row-streaming sampling with shared RNG and score helpers.

Loaded by the public sampler after its verified Ascend capability/shape gate.
"""

import triton
import triton.language as tl
from triton.runtime import driver as _ascend_driver

from flaggems_vllm import runtime
from flaggems_vllm.ops.gumbel_sample import (
    _ASCEND_351,
    _compile_plan_part,
    _gumbel_from_uniform,
    _masked_argmax,
    _rng_token_offsets,
    _sample_scores,
    murmur3_uniform32,
)
from flaggems_vllm.utils import libentry


@triton.jit
def _gumbel_streaming_row_kernel(
    LOGITS,
    MAPPING,
    TEMPERATURE,
    SEEDS,
    POSITIONS,
    SAMPLED,
    stride_logits,
    stride_mapping,
    stride_temperature,
    stride_seeds,
    stride_positions,
    VOCAB_SIZE: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    IS_DRAFTING: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    req = tl.load(MAPPING + row * stride_mapping).to(tl.int64)
    valid_req = req >= 0
    temperature = tl.load(TEMPERATURE + req * stride_temperature, valid_req, 0.0).to(
        tl.float32
    )
    seed = tl.load(SEEDS + req * stride_seeds, valid_req, 0).to(tl.int64)
    position = tl.load(POSITIONS + row * stride_positions).to(tl.int64)
    if IS_DRAFTING:
        position = position + (1 << 30)
    row_logits = LOGITS + row * stride_logits
    best_value = tl.full((), float("-inf"), tl.float32)
    best_index = tl.full((), 0x7FFFFFFF, tl.int32)
    for tile in range(0, tl.cdiv(VOCAB_SIZE, TILE_SIZE)):
        indices = tile * TILE_SIZE + tl.arange(0, TILE_SIZE)
        mask = indices < VOCAB_SIZE
        logits = tl.load(row_logits + indices, mask, float("-inf"))
        noise = tl.full((TILE_SIZE,), 0.0, tl.float32)
        if temperature != 0.0:
            rng_indices = _rng_token_offsets(indices, tile, TILE_SIZE, False)
            uniform = murmur3_uniform32(seed, position, rng_indices)
            noise = _gumbel_from_uniform(uniform, False, FINITE_INPUT=True)
        scores = _sample_scores(logits, noise, temperature, APPLY_TEMPERATURE, False)
        tile_value, tile_index = _masked_argmax(scores, indices, mask)
        take = (tile_value > best_value) | (
            (tile_value == best_value) & (tile_index < best_index)
        )
        best_value = tl.where(take, tile_value, best_value)
        best_index = tl.where(take, tile_index, best_index)
    tl.store(SAMPLED + row, best_index.to(tl.int64))


_streaming_plans = {}
_streaming_entry = libentry()(_gumbel_streaming_row_kernel)
_ascend_stream_config = (
    runtime.get_tuned_config("gumbel_sample_streaming_row")[0] if _ASCEND_351 else None
)


def _sample_streaming_row(
    logits,
    mapping,
    temperature,
    seed,
    pos,
    sampled,
    metadata,
    apply_temperature,
    is_drafting,
):
    tile_size = _ascend_stream_config.kwargs["TILE_SIZE"]
    args = (logits, mapping, temperature, seed, pos, sampled, *metadata[2:])
    tensor_device = logits.device
    device_index = tensor_device.index if tensor_device.type == "npu" else None
    with runtime.torch_device_fn.device(
        tensor_device if device_index is None else device_index
    ):
        device = (
            runtime.torch_device_fn.current_device()
            if device_index is None
            else device_index
        )
        key = (
            "streaming_row_v1",
            device,
            metadata,
            tile_size,
            _ascend_stream_config.num_warps,
            _ascend_stream_config.num_stages,
            tuple((tensor.dtype, tensor.data_ptr() % 16 == 0) for tensor in args[:6]),
            apply_temperature,
            is_drafting,
            False,
        )
        plan = _streaming_plans.get(key)
        if plan is None:
            runner, tail = _compile_plan_part(
                _streaming_entry,
                (metadata[0],),
                args,
                dict(
                    VOCAB_SIZE=metadata[1],
                    APPLY_TEMPERATURE=apply_temperature,
                    IS_DRAFTING=is_drafting,
                    TILE_SIZE=tile_size,
                    num_warps=_ascend_stream_config.num_warps,
                    num_stages=_ascend_stream_config.num_stages,
                    enable_auto_blockify=False,
                ),
            )
            _streaming_plans[key] = runner, tail
        else:
            runner, tail = plan
            stream = _ascend_driver.active.get_current_stream(device)
            runner(*args, *tail, stream=stream)
    return sampled
