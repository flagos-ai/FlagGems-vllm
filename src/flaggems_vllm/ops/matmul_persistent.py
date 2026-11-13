# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 FlagOS Contributors
"""vLLM batch-invariant matmul with a bounded H20 small-matrix path.

Kernels and tuned BF16 tables originate from the frozen vLLM port. This module
exports only matmul, never global ATen overrides or unrelated reductions.
"""

import math
from collections.abc import Callable
from functools import cache
from typing import Any, NamedTuple

import torch
import triton
import triton.language as tl

__all__ = ["matmul_persistent"]


class _MatmulMConfig(NamedTuple):
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int


class _MatmulShapeConfig(NamedTuple):
    block_k: int
    m_buckets: tuple[tuple[int, _MatmulMConfig], ...]


_BATCH_INVARIANT_MATMUL_TUNED_CONFIGS: dict[
    str, dict[tuple[int, int], _MatmulShapeConfig]
] = {
    "ada": {
        (12288, 2048): _MatmulShapeConfig(
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 128, 8, 2)),
                (4, _MatmulMConfig(16, 64, 8, 5)),
                (8, _MatmulMConfig(16, 128, 8, 4)),
                (16, _MatmulMConfig(16, 128, 4, 5)),
                (32, _MatmulMConfig(32, 128, 4, 5)),
                (64, _MatmulMConfig(32, 256, 4, 3)),
                (256, _MatmulMConfig(128, 128, 8, 3)),
                (512, _MatmulMConfig(128, 64, 4, 4)),
                (1024, _MatmulMConfig(128, 128, 8, 3)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (2048, 6144): _MatmulShapeConfig(
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 64, 4, 4)),
                (4, _MatmulMConfig(16, 64, 8, 4)),
                (8, _MatmulMConfig(16, 32, 8, 4)),
                (16, _MatmulMConfig(16, 32, 8, 4)),
                (32, _MatmulMConfig(16, 64, 8, 3)),
                (64, _MatmulMConfig(32, 64, 4, 4)),
                (256, _MatmulMConfig(32, 64, 4, 4)),
                (512, _MatmulMConfig(64, 64, 4, 3)),
                (1024, _MatmulMConfig(64, 64, 4, 3)),
                (2048, _MatmulMConfig(64, 64, 4, 3)),
            ),
        ),
        (4096, 2048): _MatmulShapeConfig(
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 256, 8, 3)),
                (4, _MatmulMConfig(16, 64, 4, 3)),
                (8, _MatmulMConfig(16, 128, 8, 2)),
                (16, _MatmulMConfig(16, 32, 4, 5)),
                (32, _MatmulMConfig(16, 64, 4, 4)),
                (64, _MatmulMConfig(64, 64, 8, 3)),
                (256, _MatmulMConfig(64, 64, 4, 5)),
                (512, _MatmulMConfig(64, 64, 4, 5)),
                (1024, _MatmulMConfig(128, 64, 4, 4)),
                (2048, _MatmulMConfig(128, 64, 8, 4)),
            ),
        ),
        (151936, 2048): _MatmulShapeConfig(
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 32, 8, 5)),
                (4, _MatmulMConfig(16, 32, 8, 5)),
                (8, _MatmulMConfig(16, 32, 8, 5)),
                (16, _MatmulMConfig(16, 32, 4, 4)),
                (32, _MatmulMConfig(16, 64, 4, 4)),
                (64, _MatmulMConfig(32, 64, 4, 4)),
                (256, _MatmulMConfig(128, 128, 4, 3)),
                (512, _MatmulMConfig(128, 128, 4, 3)),
                (1024, _MatmulMConfig(128, 128, 4, 3)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (2048, 2048): _MatmulShapeConfig(
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(32, 128, 8, 5)),
                (4, _MatmulMConfig(16, 256, 8, 3)),
                (8, _MatmulMConfig(32, 128, 8, 3)),
                (16, _MatmulMConfig(32, 128, 8, 3)),
                (32, _MatmulMConfig(16, 128, 8, 5)),
                (64, _MatmulMConfig(16, 256, 4, 3)),
                (256, _MatmulMConfig(64, 128, 8, 4)),
                (512, _MatmulMConfig(64, 64, 4, 5)),
                (1024, _MatmulMConfig(64, 64, 4, 5)),
                (2048, _MatmulMConfig(64, 64, 4, 5)),
            ),
        ),
    },
    "hopper": {
        (12288, 2048): _MatmulShapeConfig(
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 256, 4, 3)),
                (4, _MatmulMConfig(16, 256, 4, 4)),
                (8, _MatmulMConfig(16, 64, 4, 4)),
                (16, _MatmulMConfig(16, 64, 4, 4)),
                (32, _MatmulMConfig(32, 64, 8, 5)),
                (64, _MatmulMConfig(64, 64, 4, 4)),
                (256, _MatmulMConfig(64, 128, 4, 3)),
                (512, _MatmulMConfig(128, 128, 8, 3)),
                (1024, _MatmulMConfig(128, 256, 8, 2)),
                (2048, _MatmulMConfig(128, 256, 8, 2)),
            ),
        ),
        (2048, 6144): _MatmulShapeConfig(
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 32, 4, 4)),
                (4, _MatmulMConfig(16, 64, 4, 4)),
                (8, _MatmulMConfig(16, 64, 4, 4)),
                (16, _MatmulMConfig(16, 32, 4, 3)),
                (32, _MatmulMConfig(32, 32, 8, 3)),
                (64, _MatmulMConfig(64, 32, 4, 5)),
                (256, _MatmulMConfig(64, 128, 4, 3)),
                (512, _MatmulMConfig(128, 128, 4, 3)),
                (1024, _MatmulMConfig(64, 64, 4, 5)),
                (2048, _MatmulMConfig(64, 128, 4, 4)),
            ),
        ),
        (4096, 2048): _MatmulShapeConfig(
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 64, 8, 5)),
                (4, _MatmulMConfig(16, 64, 4, 5)),
                (8, _MatmulMConfig(16, 128, 4, 3)),
                (16, _MatmulMConfig(16, 64, 4, 5)),
                (32, _MatmulMConfig(16, 64, 4, 4)),
                (64, _MatmulMConfig(64, 32, 4, 3)),
                (256, _MatmulMConfig(64, 256, 4, 2)),
                (512, _MatmulMConfig(64, 64, 4, 3)),
                (1024, _MatmulMConfig(64, 128, 4, 3)),
                (2048, _MatmulMConfig(64, 64, 4, 4)),
            ),
        ),
        (151936, 2048): _MatmulShapeConfig(
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 256, 4, 4)),
                (4, _MatmulMConfig(16, 256, 4, 4)),
                (8, _MatmulMConfig(16, 256, 4, 3)),
                (16, _MatmulMConfig(16, 256, 4, 3)),
                (32, _MatmulMConfig(32, 256, 8, 3)),
                (64, _MatmulMConfig(64, 128, 4, 3)),
                (256, _MatmulMConfig(64, 128, 4, 3)),
                (512, _MatmulMConfig(128, 128, 8, 3)),
                (1024, _MatmulMConfig(128, 256, 8, 2)),
                (2048, _MatmulMConfig(128, 256, 8, 2)),
            ),
        ),
        (2048, 2048): _MatmulShapeConfig(
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 64, 4, 5)),
                (4, _MatmulMConfig(16, 128, 8, 3)),
                (8, _MatmulMConfig(16, 32, 4, 3)),
                (16, _MatmulMConfig(32, 64, 8, 5)),
                (32, _MatmulMConfig(16, 128, 4, 5)),
                (64, _MatmulMConfig(32, 64, 4, 2)),
                (256, _MatmulMConfig(128, 64, 4, 4)),
                (512, _MatmulMConfig(64, 256, 4, 2)),
                (1024, _MatmulMConfig(64, 64, 4, 3)),
                (2048, _MatmulMConfig(64, 128, 4, 3)),
            ),
        ),
    },
    "blackwell": {
        (12288, 2048): _MatmulShapeConfig(
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 128, 8, 5)),
                (4, _MatmulMConfig(16, 128, 4, 5)),
                (8, _MatmulMConfig(16, 128, 4, 5)),
                (16, _MatmulMConfig(16, 128, 4, 5)),
                (32, _MatmulMConfig(32, 128, 4, 5)),
                (64, _MatmulMConfig(64, 128, 8, 5)),
                (256, _MatmulMConfig(128, 256, 4, 4)),
                (512, _MatmulMConfig(128, 256, 4, 4)),
                (1024, _MatmulMConfig(128, 256, 4, 4)),
                (2048, _MatmulMConfig(128, 256, 4, 4)),
            ),
        ),
        (2048, 6144): _MatmulShapeConfig(
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 32, 4, 5)),
                (4, _MatmulMConfig(16, 32, 4, 5)),
                (8, _MatmulMConfig(16, 32, 4, 5)),
                (16, _MatmulMConfig(16, 32, 4, 5)),
                (32, _MatmulMConfig(16, 32, 4, 5)),
                (64, _MatmulMConfig(32, 32, 4, 5)),
                (256, _MatmulMConfig(64, 64, 4, 5)),
                (512, _MatmulMConfig(128, 64, 4, 4)),
                (1024, _MatmulMConfig(128, 128, 8, 3)),
                (2048, _MatmulMConfig(128, 128, 8, 3)),
            ),
        ),
        (4096, 2048): _MatmulShapeConfig(
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 64, 4, 5)),
                (4, _MatmulMConfig(16, 64, 4, 5)),
                (8, _MatmulMConfig(16, 32, 4, 5)),
                (16, _MatmulMConfig(16, 32, 4, 5)),
                (32, _MatmulMConfig(32, 32, 4, 5)),
                (64, _MatmulMConfig(32, 64, 4, 4)),
                (256, _MatmulMConfig(128, 64, 4, 4)),
                (512, _MatmulMConfig(128, 128, 8, 3)),
                (1024, _MatmulMConfig(128, 128, 8, 3)),
                (2048, _MatmulMConfig(128, 128, 8, 3)),
            ),
        ),
        (151936, 2048): _MatmulShapeConfig(
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 256, 4, 5)),
                (4, _MatmulMConfig(16, 256, 4, 5)),
                (8, _MatmulMConfig(16, 256, 4, 5)),
                (16, _MatmulMConfig(16, 256, 4, 5)),
                (32, _MatmulMConfig(64, 256, 4, 5)),
                (64, _MatmulMConfig(64, 256, 4, 5)),
                (256, _MatmulMConfig(128, 256, 4, 4)),
                (512, _MatmulMConfig(128, 256, 4, 4)),
                (1024, _MatmulMConfig(128, 256, 4, 4)),
                (2048, _MatmulMConfig(128, 256, 4, 4)),
            ),
        ),
        (2048, 2048): _MatmulShapeConfig(
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 32, 4, 4)),
                (4, _MatmulMConfig(16, 32, 4, 4)),
                (8, _MatmulMConfig(16, 32, 4, 5)),
                (16, _MatmulMConfig(16, 32, 8, 5)),
                (32, _MatmulMConfig(16, 32, 4, 4)),
                (64, _MatmulMConfig(32, 32, 4, 5)),
                (256, _MatmulMConfig(64, 64, 4, 4)),
                (512, _MatmulMConfig(128, 64, 4, 4)),
                (1024, _MatmulMConfig(128, 128, 4, 3)),
                (2048, _MatmulMConfig(128, 128, 8, 3)),
            ),
        ),
    },
    "sm120": {
        (12288, 2048): _MatmulShapeConfig(  # Qwen3-1.7B gate_up_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 128, 8, 5)),
                (4, _MatmulMConfig(16, 128, 8, 4)),
                (8, _MatmulMConfig(16, 128, 8, 4)),
                (16, _MatmulMConfig(16, 128, 4, 4)),
                (32, _MatmulMConfig(32, 128, 8, 5)),
                (64, _MatmulMConfig(64, 128, 4, 3)),
                (256, _MatmulMConfig(64, 64, 4, 4)),
                (512, _MatmulMConfig(64, 128, 4, 5)),
                (1024, _MatmulMConfig(128, 128, 4, 3)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (2048, 6144): _MatmulShapeConfig(  # Qwen3-1.7B down_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 32, 8, 5)),
                (4, _MatmulMConfig(16, 32, 8, 3)),
                (8, _MatmulMConfig(16, 32, 8, 4)),
                (16, _MatmulMConfig(16, 32, 4, 5)),
                (32, _MatmulMConfig(32, 32, 4, 5)),
                (64, _MatmulMConfig(32, 32, 4, 4)),
                (256, _MatmulMConfig(64, 64, 4, 5)),
                (512, _MatmulMConfig(128, 64, 4, 3)),
                (1024, _MatmulMConfig(128, 128, 4, 3)),
                (2048, _MatmulMConfig(128, 64, 8, 4)),
            ),
        ),
        (4096, 2048): _MatmulShapeConfig(  # Qwen3-1.7B qkv_proj
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 64, 4, 4)),
                (4, _MatmulMConfig(16, 64, 8, 5)),
                (8, _MatmulMConfig(16, 64, 8, 5)),
                (16, _MatmulMConfig(16, 64, 8, 3)),
                (32, _MatmulMConfig(16, 64, 8, 5)),
                (64, _MatmulMConfig(32, 64, 4, 2)),
                (256, _MatmulMConfig(64, 128, 4, 3)),
                (512, _MatmulMConfig(64, 64, 4, 3)),
                (1024, _MatmulMConfig(64, 128, 4, 3)),
                (2048, _MatmulMConfig(64, 128, 4, 3)),
            ),
        ),
        (151936, 2048): _MatmulShapeConfig(  # Qwen3-1.7B lm_head
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 64, 8, 5)),
                (4, _MatmulMConfig(16, 128, 4, 4)),
                (8, _MatmulMConfig(16, 128, 4, 5)),
                (16, _MatmulMConfig(16, 128, 8, 3)),
                (32, _MatmulMConfig(32, 128, 8, 3)),
                (64, _MatmulMConfig(32, 128, 8, 5)),
                (256, _MatmulMConfig(128, 128, 4, 3)),
                (512, _MatmulMConfig(128, 128, 8, 3)),
                (1024, _MatmulMConfig(128, 128, 4, 3)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (2048, 2048): _MatmulShapeConfig(  # Qwen3-1.7B o_proj
            block_k=32,
            m_buckets=(
                (1, _MatmulMConfig(16, 128, 4, 3)),
                (4, _MatmulMConfig(16, 128, 4, 4)),
                (8, _MatmulMConfig(16, 64, 4, 5)),
                (16, _MatmulMConfig(32, 32, 8, 4)),
                (32, _MatmulMConfig(16, 64, 4, 5)),
                (64, _MatmulMConfig(32, 32, 4, 5)),
                (256, _MatmulMConfig(128, 32, 4, 5)),
                (512, _MatmulMConfig(128, 64, 4, 4)),
                (1024, _MatmulMConfig(128, 128, 8, 5)),
                (2048, _MatmulMConfig(128, 256, 8, 4)),
            ),
        ),
        (19456, 2560): _MatmulShapeConfig(  # Qwen3-4B gate_up_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 128, 8, 4)),
                (4, _MatmulMConfig(16, 128, 8, 3)),
                (8, _MatmulMConfig(16, 128, 8, 5)),
                (16, _MatmulMConfig(16, 128, 8, 5)),
                (32, _MatmulMConfig(32, 128, 4, 4)),
                (64, _MatmulMConfig(64, 128, 8, 4)),
                (256, _MatmulMConfig(128, 128, 4, 3)),
                (512, _MatmulMConfig(64, 128, 4, 5)),
                (1024, _MatmulMConfig(128, 128, 4, 3)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (2560, 9728): _MatmulShapeConfig(  # Qwen3-4B down_proj
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 64, 4, 5)),
                (4, _MatmulMConfig(16, 64, 4, 4)),
                (8, _MatmulMConfig(16, 64, 4, 4)),
                (16, _MatmulMConfig(16, 32, 4, 5)),
                (32, _MatmulMConfig(32, 32, 4, 3)),
                (64, _MatmulMConfig(32, 64, 8, 4)),
                (256, _MatmulMConfig(64, 64, 4, 3)),
                (512, _MatmulMConfig(64, 128, 4, 3)),
                (1024, _MatmulMConfig(64, 128, 4, 3)),
                (2048, _MatmulMConfig(64, 128, 4, 3)),
            ),
        ),
        (6144, 2560): _MatmulShapeConfig(  # Qwen3-4B qkv_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 128, 8, 4)),
                (4, _MatmulMConfig(16, 256, 4, 3)),
                (8, _MatmulMConfig(16, 128, 4, 5)),
                (16, _MatmulMConfig(16, 64, 8, 3)),
                (32, _MatmulMConfig(16, 128, 8, 4)),
                (64, _MatmulMConfig(64, 64, 4, 5)),
                (256, _MatmulMConfig(128, 128, 8, 3)),
                (512, _MatmulMConfig(64, 64, 4, 5)),
                (1024, _MatmulMConfig(64, 128, 8, 4)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (151936, 2560): _MatmulShapeConfig(  # Qwen3-4B lm_head
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 64, 8, 5)),
                (4, _MatmulMConfig(16, 64, 8, 5)),
                (8, _MatmulMConfig(16, 64, 4, 5)),
                (16, _MatmulMConfig(16, 64, 4, 5)),
                (32, _MatmulMConfig(32, 64, 4, 5)),
                (64, _MatmulMConfig(64, 64, 4, 5)),
                (256, _MatmulMConfig(128, 128, 4, 3)),
                (512, _MatmulMConfig(128, 128, 4, 3)),
                (1024, _MatmulMConfig(128, 128, 8, 3)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (2560, 4096): _MatmulShapeConfig(  # Qwen3-4B o_proj
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 64, 8, 3)),
                (4, _MatmulMConfig(16, 64, 8, 4)),
                (8, _MatmulMConfig(16, 64, 8, 5)),
                (16, _MatmulMConfig(16, 64, 4, 4)),
                (32, _MatmulMConfig(16, 32, 4, 4)),
                (64, _MatmulMConfig(16, 64, 8, 3)),
                (256, _MatmulMConfig(64, 64, 8, 3)),
                (512, _MatmulMConfig(64, 128, 4, 3)),
                (1024, _MatmulMConfig(64, 128, 4, 3)),
                (2048, _MatmulMConfig(64, 128, 4, 3)),
            ),
        ),
        (24576, 4096): _MatmulShapeConfig(  # Qwen3-8B gate_up_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 256, 8, 3)),
                (4, _MatmulMConfig(16, 256, 8, 3)),
                (8, _MatmulMConfig(16, 256, 8, 3)),
                (16, _MatmulMConfig(16, 256, 8, 3)),
                (32, _MatmulMConfig(32, 256, 8, 3)),
                (64, _MatmulMConfig(64, 256, 4, 3)),
                (256, _MatmulMConfig(64, 128, 4, 5)),
                (512, _MatmulMConfig(128, 128, 4, 3)),
                (1024, _MatmulMConfig(128, 128, 4, 3)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (4096, 12288): _MatmulShapeConfig(  # Qwen3-8B down_proj
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 64, 4, 5)),
                (4, _MatmulMConfig(16, 64, 4, 5)),
                (8, _MatmulMConfig(16, 64, 4, 4)),
                (16, _MatmulMConfig(16, 32, 4, 4)),
                (32, _MatmulMConfig(32, 32, 8, 5)),
                (64, _MatmulMConfig(32, 64, 4, 5)),
                (256, _MatmulMConfig(64, 128, 4, 3)),
                (512, _MatmulMConfig(64, 64, 4, 3)),
                (1024, _MatmulMConfig(64, 128, 4, 3)),
                (2048, _MatmulMConfig(64, 128, 4, 3)),
            ),
        ),
        (6144, 4096): _MatmulShapeConfig(  # Qwen3-8B qkv_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 64, 8, 4)),
                (4, _MatmulMConfig(16, 128, 8, 4)),
                (8, _MatmulMConfig(16, 128, 4, 4)),
                (16, _MatmulMConfig(16, 64, 4, 5)),
                (32, _MatmulMConfig(16, 128, 8, 5)),
                (64, _MatmulMConfig(64, 64, 4, 4)),
                (256, _MatmulMConfig(128, 128, 8, 3)),
                (512, _MatmulMConfig(64, 64, 4, 5)),
                (1024, _MatmulMConfig(128, 64, 4, 3)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (151936, 4096): _MatmulShapeConfig(  # Qwen3-8B lm_head
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 64, 8, 5)),
                (4, _MatmulMConfig(16, 64, 8, 5)),
                (8, _MatmulMConfig(16, 64, 4, 5)),
                (16, _MatmulMConfig(16, 64, 4, 5)),
                (32, _MatmulMConfig(32, 64, 4, 5)),
                (64, _MatmulMConfig(32, 128, 8, 5)),
                (256, _MatmulMConfig(128, 128, 4, 3)),
                (512, _MatmulMConfig(128, 128, 8, 3)),
                (1024, _MatmulMConfig(128, 128, 4, 3)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (4096, 4096): _MatmulShapeConfig(  # Qwen3-8B o_proj
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 64, 8, 4)),
                (4, _MatmulMConfig(16, 64, 4, 3)),
                (8, _MatmulMConfig(16, 32, 4, 4)),
                (16, _MatmulMConfig(16, 32, 4, 3)),
                (32, _MatmulMConfig(16, 64, 4, 5)),
                (64, _MatmulMConfig(64, 32, 8, 4)),
                (256, _MatmulMConfig(64, 128, 4, 3)),
                (512, _MatmulMConfig(64, 64, 4, 3)),
                (1024, _MatmulMConfig(64, 128, 4, 3)),
                (2048, _MatmulMConfig(64, 128, 4, 3)),
            ),
        ),
        (6144, 2048): _MatmulShapeConfig(  # Qwen3-1.7B tp2_gate_up_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 32, 8, 3)),
                (4, _MatmulMConfig(16, 128, 8, 5)),
                (8, _MatmulMConfig(16, 128, 4, 5)),
                (16, _MatmulMConfig(16, 32, 4, 5)),
                (32, _MatmulMConfig(32, 128, 8, 5)),
                (64, _MatmulMConfig(32, 128, 4, 4)),
                (256, _MatmulMConfig(128, 128, 4, 3)),
                (512, _MatmulMConfig(64, 64, 4, 5)),
                (1024, _MatmulMConfig(128, 64, 4, 3)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (2048, 3072): _MatmulShapeConfig(  # Qwen3-1.7B tp2_down_proj
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 32, 8, 5)),
                (4, _MatmulMConfig(16, 64, 4, 3)),
                (8, _MatmulMConfig(16, 64, 8, 4)),
                (16, _MatmulMConfig(16, 64, 8, 3)),
                (32, _MatmulMConfig(16, 64, 8, 3)),
                (64, _MatmulMConfig(16, 64, 4, 5)),
                (256, _MatmulMConfig(64, 64, 8, 3)),
                (512, _MatmulMConfig(64, 128, 4, 3)),
                (1024, _MatmulMConfig(64, 64, 4, 3)),
                (2048, _MatmulMConfig(64, 128, 4, 3)),
            ),
        ),
        (2048, 1024): _MatmulShapeConfig(  # Qwen3-1.7B tp2_o_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 256, 8, 2)),
                (4, _MatmulMConfig(16, 32, 8, 4)),
                (8, _MatmulMConfig(32, 128, 8, 5)),
                (16, _MatmulMConfig(32, 128, 4, 4)),
                (32, _MatmulMConfig(64, 128, 4, 3)),
                (64, _MatmulMConfig(16, 256, 4, 3)),
                (256, _MatmulMConfig(32, 128, 4, 3)),
                (512, _MatmulMConfig(64, 128, 4, 4)),
                (1024, _MatmulMConfig(128, 128, 8, 3)),
                (2048, _MatmulMConfig(64, 128, 4, 5)),
            ),
        ),
        (75968, 2048): _MatmulShapeConfig(  # Qwen3-1.7B tp2_lm_head
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 256, 8, 3)),
                (4, _MatmulMConfig(16, 256, 8, 3)),
                (8, _MatmulMConfig(16, 256, 8, 3)),
                (16, _MatmulMConfig(16, 256, 4, 3)),
                (32, _MatmulMConfig(32, 256, 8, 3)),
                (64, _MatmulMConfig(64, 256, 8, 3)),
                (256, _MatmulMConfig(64, 128, 4, 5)),
                (512, _MatmulMConfig(128, 128, 4, 3)),
                (1024, _MatmulMConfig(128, 128, 4, 3)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (3072, 2048): _MatmulShapeConfig(  # Qwen3-1.7B tp4_gate_up_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 256, 4, 3)),
                (4, _MatmulMConfig(32, 128, 4, 5)),
                (8, _MatmulMConfig(16, 128, 8, 3)),
                (16, _MatmulMConfig(32, 128, 8, 4)),
                (32, _MatmulMConfig(32, 128, 8, 5)),
                (64, _MatmulMConfig(64, 32, 8, 3)),
                (256, _MatmulMConfig(128, 64, 8, 3)),
                (512, _MatmulMConfig(128, 128, 4, 3)),
                (1024, _MatmulMConfig(64, 64, 4, 5)),
                (2048, _MatmulMConfig(64, 128, 8, 4)),
            ),
        ),
        (2048, 1536): _MatmulShapeConfig(  # Qwen3-1.7B tp4_down_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 128, 8, 2)),
                (4, _MatmulMConfig(16, 256, 4, 3)),
                (8, _MatmulMConfig(32, 128, 4, 2)),
                (16, _MatmulMConfig(32, 32, 8, 5)),
                (32, _MatmulMConfig(32, 32, 8, 4)),
                (64, _MatmulMConfig(32, 128, 4, 4)),
                (256, _MatmulMConfig(64, 64, 4, 3)),
                (512, _MatmulMConfig(128, 64, 8, 4)),
                (1024, _MatmulMConfig(128, 128, 4, 3)),
                (2048, _MatmulMConfig(128, 64, 8, 4)),
            ),
        ),
        (1024, 2048): _MatmulShapeConfig(  # Qwen3-1.7B tp4_qkv_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(32, 128, 8, 4)),
                (4, _MatmulMConfig(16, 32, 8, 4)),
                (8, _MatmulMConfig(16, 32, 4, 4)),
                (16, _MatmulMConfig(32, 128, 8, 3)),
                (32, _MatmulMConfig(32, 128, 8, 4)),
                (64, _MatmulMConfig(32, 64, 4, 5)),
                (256, _MatmulMConfig(64, 32, 4, 3)),
                (512, _MatmulMConfig(64, 64, 4, 3)),
                (1024, _MatmulMConfig(64, 128, 4, 5)),
                (2048, _MatmulMConfig(128, 128, 8, 3)),
            ),
        ),
        (2048, 512): _MatmulShapeConfig(  # Qwen3-1.7B tp4_o_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(32, 32, 4, 5)),
                (4, _MatmulMConfig(64, 128, 4, 2)),
                (8, _MatmulMConfig(16, 128, 4, 2)),
                (16, _MatmulMConfig(16, 256, 8, 3)),
                (32, _MatmulMConfig(64, 32, 8, 4)),
                (64, _MatmulMConfig(16, 128, 8, 4)),
                (256, _MatmulMConfig(64, 64, 8, 5)),
                (512, _MatmulMConfig(32, 128, 8, 4)),
                (1024, _MatmulMConfig(64, 64, 8, 3)),
                (2048, _MatmulMConfig(64, 128, 8, 3)),
            ),
        ),
        (37984, 2048): _MatmulShapeConfig(  # Qwen3-1.7B tp4_lm_head
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 128, 8, 4)),
                (4, _MatmulMConfig(16, 128, 8, 5)),
                (8, _MatmulMConfig(16, 128, 4, 5)),
                (16, _MatmulMConfig(16, 128, 8, 4)),
                (32, _MatmulMConfig(32, 128, 4, 5)),
                (64, _MatmulMConfig(64, 128, 4, 5)),
                (256, _MatmulMConfig(64, 128, 4, 5)),
                (512, _MatmulMConfig(128, 128, 4, 3)),
                (1024, _MatmulMConfig(128, 128, 4, 3)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (1536, 2048): _MatmulShapeConfig(  # Qwen3-1.7B tp8_gate_up_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(32, 128, 8, 3)),
                (4, _MatmulMConfig(16, 256, 4, 3)),
                (8, _MatmulMConfig(16, 32, 8, 5)),
                (16, _MatmulMConfig(32, 32, 8, 3)),
                (32, _MatmulMConfig(32, 128, 8, 4)),
                (64, _MatmulMConfig(16, 128, 8, 4)),
                (256, _MatmulMConfig(32, 128, 8, 3)),
                (512, _MatmulMConfig(64, 128, 8, 4)),
                (1024, _MatmulMConfig(128, 128, 4, 3)),
                (2048, _MatmulMConfig(64, 64, 4, 4)),
            ),
        ),
        (2048, 768): _MatmulShapeConfig(  # Qwen3-1.7B tp8_down_proj
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(32, 64, 4, 3)),
                (4, _MatmulMConfig(32, 64, 8, 3)),
                (8, _MatmulMConfig(16, 32, 4, 5)),
                (16, _MatmulMConfig(32, 64, 4, 2)),
                (32, _MatmulMConfig(16, 64, 8, 3)),
                (64, _MatmulMConfig(16, 32, 8, 2)),
                (256, _MatmulMConfig(32, 32, 4, 4)),
                (512, _MatmulMConfig(32, 64, 4, 5)),
                (1024, _MatmulMConfig(64, 128, 4, 3)),
                (2048, _MatmulMConfig(64, 128, 4, 3)),
            ),
        ),
        (512, 2048): _MatmulShapeConfig(  # Qwen3-1.7B tp8_qkv_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(64, 64, 8, 5)),
                (4, _MatmulMConfig(16, 32, 4, 4)),
                (8, _MatmulMConfig(16, 32, 4, 5)),
                (16, _MatmulMConfig(32, 128, 8, 3)),
                (32, _MatmulMConfig(64, 64, 8, 5)),
                (64, _MatmulMConfig(32, 128, 8, 4)),
                (256, _MatmulMConfig(16, 128, 4, 4)),
                (512, _MatmulMConfig(32, 64, 8, 3)),
                (1024, _MatmulMConfig(64, 64, 4, 4)),
                (2048, _MatmulMConfig(64, 128, 4, 4)),
            ),
        ),
        (2048, 256): _MatmulShapeConfig(  # Qwen3-1.7B tp8_o_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(32, 128, 8, 2)),
                (4, _MatmulMConfig(16, 128, 4, 2)),
                (8, _MatmulMConfig(32, 256, 4, 2)),
                (16, _MatmulMConfig(32, 32, 8, 4)),
                (32, _MatmulMConfig(16, 256, 4, 3)),
                (64, _MatmulMConfig(128, 32, 8, 3)),
                (256, _MatmulMConfig(16, 32, 8, 2)),
                (512, _MatmulMConfig(16, 32, 4, 3)),
                (1024, _MatmulMConfig(64, 256, 4, 3)),
                (2048, _MatmulMConfig(64, 128, 8, 4)),
            ),
        ),
        (18992, 2048): _MatmulShapeConfig(  # Qwen3-1.7B tp8_lm_head
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 64, 8, 5)),
                (4, _MatmulMConfig(16, 128, 8, 5)),
                (8, _MatmulMConfig(16, 128, 8, 3)),
                (16, _MatmulMConfig(16, 128, 4, 5)),
                (32, _MatmulMConfig(32, 128, 8, 5)),
                (64, _MatmulMConfig(64, 128, 8, 4)),
                (256, _MatmulMConfig(128, 128, 4, 3)),
                (512, _MatmulMConfig(128, 128, 4, 3)),
                (1024, _MatmulMConfig(128, 128, 8, 3)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (9728, 2560): _MatmulShapeConfig(  # Qwen3-4B tp2_gate_up_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 32, 8, 4)),
                (4, _MatmulMConfig(16, 64, 8, 3)),
                (8, _MatmulMConfig(16, 128, 4, 5)),
                (16, _MatmulMConfig(16, 32, 4, 5)),
                (32, _MatmulMConfig(32, 64, 8, 4)),
                (64, _MatmulMConfig(32, 128, 4, 3)),
                (256, _MatmulMConfig(128, 128, 4, 3)),
                (512, _MatmulMConfig(128, 128, 4, 3)),
                (1024, _MatmulMConfig(64, 128, 4, 5)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (2560, 4864): _MatmulShapeConfig(  # Qwen3-4B tp2_down_proj
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 32, 4, 3)),
                (4, _MatmulMConfig(16, 128, 8, 3)),
                (8, _MatmulMConfig(16, 64, 8, 4)),
                (16, _MatmulMConfig(32, 64, 8, 4)),
                (32, _MatmulMConfig(16, 64, 4, 3)),
                (64, _MatmulMConfig(32, 32, 8, 4)),
                (256, _MatmulMConfig(64, 64, 4, 3)),
                (512, _MatmulMConfig(64, 128, 4, 3)),
                (1024, _MatmulMConfig(64, 128, 4, 3)),
                (2048, _MatmulMConfig(64, 128, 4, 3)),
            ),
        ),
        (3072, 2560): _MatmulShapeConfig(  # Qwen3-4B tp2_qkv_proj
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 128, 8, 3)),
                (4, _MatmulMConfig(16, 32, 8, 2)),
                (8, _MatmulMConfig(16, 32, 8, 4)),
                (16, _MatmulMConfig(16, 64, 4, 3)),
                (32, _MatmulMConfig(32, 64, 4, 4)),
                (64, _MatmulMConfig(16, 128, 4, 3)),
                (256, _MatmulMConfig(64, 128, 4, 3)),
                (512, _MatmulMConfig(64, 64, 4, 3)),
                (1024, _MatmulMConfig(64, 128, 4, 3)),
                (2048, _MatmulMConfig(64, 128, 4, 3)),
            ),
        ),
        (2560, 2048): _MatmulShapeConfig(  # Qwen3-4B tp2_o_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(32, 128, 8, 4)),
                (4, _MatmulMConfig(32, 128, 4, 4)),
                (8, _MatmulMConfig(16, 128, 4, 3)),
                (16, _MatmulMConfig(32, 128, 4, 5)),
                (32, _MatmulMConfig(16, 128, 8, 4)),
                (64, _MatmulMConfig(16, 128, 8, 4)),
                (256, _MatmulMConfig(128, 32, 8, 5)),
                (512, _MatmulMConfig(128, 64, 8, 3)),
                (1024, _MatmulMConfig(128, 128, 4, 3)),
                (2048, _MatmulMConfig(128, 128, 8, 3)),
            ),
        ),
        (75968, 2560): _MatmulShapeConfig(  # Qwen3-4B tp2_lm_head
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 64, 8, 5)),
                (4, _MatmulMConfig(16, 64, 8, 5)),
                (8, _MatmulMConfig(16, 64, 4, 5)),
                (16, _MatmulMConfig(16, 64, 4, 5)),
                (32, _MatmulMConfig(32, 64, 4, 5)),
                (64, _MatmulMConfig(64, 256, 4, 3)),
                (256, _MatmulMConfig(64, 128, 4, 5)),
                (512, _MatmulMConfig(128, 128, 4, 3)),
                (1024, _MatmulMConfig(128, 128, 4, 3)),
                (2048, _MatmulMConfig(128, 128, 8, 3)),
            ),
        ),
        (4864, 2560): _MatmulShapeConfig(  # Qwen3-4B tp4_gate_up_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 32, 8, 5)),
                (4, _MatmulMConfig(16, 128, 4, 4)),
                (8, _MatmulMConfig(16, 32, 8, 4)),
                (16, _MatmulMConfig(16, 128, 8, 4)),
                (32, _MatmulMConfig(16, 128, 4, 5)),
                (64, _MatmulMConfig(16, 128, 4, 3)),
                (256, _MatmulMConfig(64, 128, 4, 5)),
                (512, _MatmulMConfig(128, 128, 4, 3)),
                (1024, _MatmulMConfig(128, 128, 4, 3)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (2560, 2432): _MatmulShapeConfig(  # Qwen3-4B tp4_down_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 32, 8, 4)),
                (4, _MatmulMConfig(16, 32, 4, 5)),
                (8, _MatmulMConfig(32, 64, 8, 4)),
                (16, _MatmulMConfig(32, 128, 8, 5)),
                (32, _MatmulMConfig(16, 128, 4, 4)),
                (64, _MatmulMConfig(64, 32, 4, 3)),
                (256, _MatmulMConfig(32, 128, 8, 4)),
                (512, _MatmulMConfig(64, 128, 4, 5)),
                (1024, _MatmulMConfig(128, 128, 4, 3)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (1536, 2560): _MatmulShapeConfig(  # Qwen3-4B tp4_qkv_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 32, 8, 4)),
                (4, _MatmulMConfig(16, 32, 8, 3)),
                (8, _MatmulMConfig(16, 32, 4, 4)),
                (16, _MatmulMConfig(32, 128, 8, 3)),
                (32, _MatmulMConfig(16, 128, 4, 3)),
                (64, _MatmulMConfig(16, 64, 8, 4)),
                (256, _MatmulMConfig(128, 32, 4, 5)),
                (512, _MatmulMConfig(64, 128, 4, 5)),
                (1024, _MatmulMConfig(64, 256, 4, 3)),
                (2048, _MatmulMConfig(64, 128, 4, 4)),
            ),
        ),
        (2560, 1024): _MatmulShapeConfig(  # Qwen3-4B tp4_o_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(32, 128, 8, 5)),
                (4, _MatmulMConfig(16, 32, 8, 3)),
                (8, _MatmulMConfig(16, 128, 4, 3)),
                (16, _MatmulMConfig(32, 256, 4, 3)),
                (32, _MatmulMConfig(16, 128, 4, 2)),
                (64, _MatmulMConfig(32, 256, 8, 3)),
                (256, _MatmulMConfig(32, 128, 4, 5)),
                (512, _MatmulMConfig(64, 128, 8, 3)),
                (1024, _MatmulMConfig(128, 128, 4, 3)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (37984, 2560): _MatmulShapeConfig(  # Qwen3-4B tp4_lm_head
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 128, 8, 3)),
                (4, _MatmulMConfig(16, 128, 8, 3)),
                (8, _MatmulMConfig(16, 128, 4, 3)),
                (16, _MatmulMConfig(16, 32, 4, 5)),
                (32, _MatmulMConfig(16, 32, 4, 5)),
                (64, _MatmulMConfig(32, 32, 4, 5)),
                (256, _MatmulMConfig(64, 128, 4, 3)),
                (512, _MatmulMConfig(64, 128, 4, 3)),
                (1024, _MatmulMConfig(64, 128, 4, 3)),
                (2048, _MatmulMConfig(64, 128, 4, 3)),
            ),
        ),
        (2432, 2560): _MatmulShapeConfig(  # Qwen3-4B tp8_gate_up_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 32, 4, 5)),
                (4, _MatmulMConfig(32, 128, 4, 3)),
                (8, _MatmulMConfig(16, 128, 8, 3)),
                (16, _MatmulMConfig(16, 128, 4, 4)),
                (32, _MatmulMConfig(16, 128, 8, 4)),
                (64, _MatmulMConfig(16, 128, 4, 3)),
                (256, _MatmulMConfig(64, 64, 4, 5)),
                (512, _MatmulMConfig(64, 128, 4, 4)),
                (1024, _MatmulMConfig(128, 128, 8, 3)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (2560, 1216): _MatmulShapeConfig(  # Qwen3-4B tp8_down_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 256, 8, 2)),
                (4, _MatmulMConfig(16, 128, 8, 4)),
                (8, _MatmulMConfig(16, 32, 4, 3)),
                (16, _MatmulMConfig(32, 32, 4, 3)),
                (32, _MatmulMConfig(16, 128, 8, 4)),
                (64, _MatmulMConfig(32, 128, 8, 4)),
                (256, _MatmulMConfig(32, 128, 4, 5)),
                (512, _MatmulMConfig(64, 128, 8, 4)),
                (1024, _MatmulMConfig(64, 256, 4, 3)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (768, 2560): _MatmulShapeConfig(  # Qwen3-4B tp8_qkv_proj
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(32, 32, 4, 2)),
                (4, _MatmulMConfig(16, 64, 8, 3)),
                (8, _MatmulMConfig(16, 64, 8, 4)),
                (16, _MatmulMConfig(16, 64, 4, 5)),
                (32, _MatmulMConfig(32, 64, 8, 4)),
                (64, _MatmulMConfig(16, 64, 8, 4)),
                (256, _MatmulMConfig(64, 32, 4, 5)),
                (512, _MatmulMConfig(64, 64, 8, 3)),
                (1024, _MatmulMConfig(64, 128, 4, 3)),
                (2048, _MatmulMConfig(64, 64, 4, 3)),
            ),
        ),
        (2560, 512): _MatmulShapeConfig(  # Qwen3-4B tp8_o_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 128, 8, 4)),
                (4, _MatmulMConfig(16, 32, 4, 2)),
                (8, _MatmulMConfig(16, 128, 4, 5)),
                (16, _MatmulMConfig(32, 32, 4, 5)),
                (32, _MatmulMConfig(32, 256, 4, 3)),
                (64, _MatmulMConfig(32, 128, 8, 5)),
                (256, _MatmulMConfig(16, 256, 4, 2)),
                (512, _MatmulMConfig(32, 128, 8, 4)),
                (1024, _MatmulMConfig(128, 128, 8, 3)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (18992, 2560): _MatmulShapeConfig(  # Qwen3-4B tp8_lm_head
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 64, 8, 4)),
                (4, _MatmulMConfig(16, 64, 8, 5)),
                (8, _MatmulMConfig(16, 128, 8, 3)),
                (16, _MatmulMConfig(16, 128, 8, 5)),
                (32, _MatmulMConfig(32, 128, 8, 5)),
                (64, _MatmulMConfig(64, 128, 8, 4)),
                (256, _MatmulMConfig(128, 128, 4, 3)),
                (512, _MatmulMConfig(128, 128, 4, 3)),
                (1024, _MatmulMConfig(128, 128, 4, 3)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (12288, 4096): _MatmulShapeConfig(  # Qwen3-8B tp2_gate_up_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 128, 8, 3)),
                (4, _MatmulMConfig(16, 128, 8, 5)),
                (8, _MatmulMConfig(16, 128, 8, 4)),
                (16, _MatmulMConfig(16, 128, 4, 4)),
                (32, _MatmulMConfig(32, 128, 4, 4)),
                (64, _MatmulMConfig(64, 128, 4, 5)),
                (256, _MatmulMConfig(64, 64, 4, 5)),
                (512, _MatmulMConfig(128, 64, 4, 3)),
                (1024, _MatmulMConfig(128, 128, 4, 3)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (4096, 6144): _MatmulShapeConfig(  # Qwen3-8B tp2_down_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 32, 8, 5)),
                (4, _MatmulMConfig(16, 32, 8, 5)),
                (8, _MatmulMConfig(16, 32, 8, 3)),
                (16, _MatmulMConfig(16, 32, 4, 5)),
                (32, _MatmulMConfig(16, 64, 8, 4)),
                (64, _MatmulMConfig(32, 64, 4, 5)),
                (256, _MatmulMConfig(64, 128, 4, 5)),
                (512, _MatmulMConfig(128, 128, 4, 3)),
                (1024, _MatmulMConfig(128, 64, 8, 4)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (3072, 4096): _MatmulShapeConfig(  # Qwen3-8B tp2_qkv_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 32, 8, 4)),
                (4, _MatmulMConfig(16, 32, 8, 3)),
                (8, _MatmulMConfig(16, 128, 4, 3)),
                (16, _MatmulMConfig(16, 32, 4, 4)),
                (32, _MatmulMConfig(32, 32, 8, 5)),
                (64, _MatmulMConfig(16, 128, 8, 4)),
                (256, _MatmulMConfig(64, 128, 4, 5)),
                (512, _MatmulMConfig(128, 128, 8, 3)),
                (1024, _MatmulMConfig(64, 64, 4, 5)),
                (2048, _MatmulMConfig(128, 64, 4, 3)),
            ),
        ),
        (75968, 4096): _MatmulShapeConfig(  # Qwen3-8B tp2_lm_head
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 256, 8, 3)),
                (4, _MatmulMConfig(16, 256, 8, 3)),
                (8, _MatmulMConfig(16, 256, 8, 3)),
                (16, _MatmulMConfig(16, 256, 8, 3)),
                (32, _MatmulMConfig(32, 256, 8, 3)),
                (64, _MatmulMConfig(64, 256, 8, 3)),
                (256, _MatmulMConfig(128, 128, 4, 3)),
                (512, _MatmulMConfig(128, 128, 8, 3)),
                (1024, _MatmulMConfig(128, 128, 4, 3)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (4096, 3072): _MatmulShapeConfig(  # Qwen3-8B tp4_down_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 256, 8, 3)),
                (4, _MatmulMConfig(16, 128, 4, 5)),
                (8, _MatmulMConfig(16, 128, 4, 4)),
                (16, _MatmulMConfig(16, 64, 4, 5)),
                (32, _MatmulMConfig(32, 128, 4, 3)),
                (64, _MatmulMConfig(16, 128, 4, 4)),
                (256, _MatmulMConfig(64, 128, 4, 5)),
                (512, _MatmulMConfig(128, 128, 4, 3)),
                (1024, _MatmulMConfig(128, 64, 8, 4)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (1536, 4096): _MatmulShapeConfig(  # Qwen3-8B tp4_qkv_proj
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 64, 4, 5)),
                (4, _MatmulMConfig(16, 64, 8, 3)),
                (8, _MatmulMConfig(32, 64, 4, 5)),
                (16, _MatmulMConfig(16, 64, 8, 4)),
                (32, _MatmulMConfig(16, 32, 8, 3)),
                (64, _MatmulMConfig(16, 64, 8, 5)),
                (256, _MatmulMConfig(64, 64, 4, 3)),
                (512, _MatmulMConfig(64, 128, 4, 3)),
                (1024, _MatmulMConfig(64, 64, 4, 3)),
                (2048, _MatmulMConfig(64, 128, 4, 3)),
            ),
        ),
        (4096, 1024): _MatmulShapeConfig(  # Qwen3-8B tp4_o_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 128, 8, 4)),
                (4, _MatmulMConfig(32, 128, 8, 4)),
                (8, _MatmulMConfig(16, 256, 8, 3)),
                (16, _MatmulMConfig(16, 128, 8, 3)),
                (32, _MatmulMConfig(16, 128, 4, 4)),
                (64, _MatmulMConfig(64, 64, 4, 3)),
                (256, _MatmulMConfig(128, 64, 4, 3)),
                (512, _MatmulMConfig(64, 64, 4, 5)),
                (1024, _MatmulMConfig(128, 64, 4, 4)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (37984, 4096): _MatmulShapeConfig(  # Qwen3-8B tp4_lm_head
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 128, 8, 3)),
                (4, _MatmulMConfig(16, 128, 4, 5)),
                (8, _MatmulMConfig(16, 128, 4, 5)),
                (16, _MatmulMConfig(16, 128, 8, 3)),
                (32, _MatmulMConfig(32, 128, 8, 3)),
                (64, _MatmulMConfig(64, 128, 8, 3)),
                (256, _MatmulMConfig(64, 128, 4, 5)),
                (512, _MatmulMConfig(128, 128, 4, 3)),
                (1024, _MatmulMConfig(128, 128, 4, 3)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
        (4096, 1536): _MatmulShapeConfig(  # Qwen3-8B tp8_down_proj
            block_k=32,
            m_buckets=(
                (1, _MatmulMConfig(32, 128, 8, 5)),
                (4, _MatmulMConfig(16, 256, 4, 5)),
                (8, _MatmulMConfig(16, 64, 4, 4)),
                (16, _MatmulMConfig(16, 128, 8, 3)),
                (32, _MatmulMConfig(16, 64, 4, 3)),
                (64, _MatmulMConfig(16, 128, 4, 4)),
                (256, _MatmulMConfig(64, 128, 4, 4)),
                (512, _MatmulMConfig(128, 128, 8, 5)),
                (1024, _MatmulMConfig(128, 256, 8, 4)),
                (2048, _MatmulMConfig(128, 128, 8, 4)),
            ),
        ),
        (768, 4096): _MatmulShapeConfig(  # Qwen3-8B tp8_qkv_proj
            block_k=128,
            m_buckets=(
                (1, _MatmulMConfig(16, 32, 8, 5)),
                (4, _MatmulMConfig(16, 32, 8, 3)),
                (8, _MatmulMConfig(32, 64, 4, 5)),
                (16, _MatmulMConfig(32, 64, 8, 3)),
                (32, _MatmulMConfig(32, 32, 8, 5)),
                (64, _MatmulMConfig(64, 32, 8, 3)),
                (256, _MatmulMConfig(32, 64, 4, 4)),
                (512, _MatmulMConfig(64, 64, 8, 3)),
                (1024, _MatmulMConfig(64, 128, 4, 3)),
                (2048, _MatmulMConfig(64, 64, 4, 3)),
            ),
        ),
        (4096, 512): _MatmulShapeConfig(  # Qwen3-8B tp8_o_proj
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 128, 8, 2)),
                (4, _MatmulMConfig(32, 128, 4, 2)),
                (8, _MatmulMConfig(16, 256, 4, 2)),
                (16, _MatmulMConfig(32, 256, 4, 3)),
                (32, _MatmulMConfig(32, 256, 8, 3)),
                (64, _MatmulMConfig(32, 128, 4, 5)),
                (256, _MatmulMConfig(64, 32, 8, 4)),
                (512, _MatmulMConfig(128, 128, 8, 3)),
                (1024, _MatmulMConfig(128, 64, 4, 4)),
                (2048, _MatmulMConfig(128, 128, 8, 3)),
            ),
        ),
        (18992, 4096): _MatmulShapeConfig(  # Qwen3-8B tp8_lm_head
            block_k=64,
            m_buckets=(
                (1, _MatmulMConfig(16, 64, 8, 5)),
                (4, _MatmulMConfig(16, 64, 8, 4)),
                (8, _MatmulMConfig(16, 64, 8, 5)),
                (16, _MatmulMConfig(16, 128, 4, 3)),
                (32, _MatmulMConfig(32, 64, 8, 5)),
                (64, _MatmulMConfig(64, 128, 8, 3)),
                (256, _MatmulMConfig(128, 128, 4, 3)),
                (512, _MatmulMConfig(128, 128, 4, 3)),
                (1024, _MatmulMConfig(128, 128, 4, 3)),
                (2048, _MatmulMConfig(128, 128, 4, 3)),
            ),
        ),
    },
}


def _matmul_launch_metadata(
    grid: Callable[..., Any], kernel: Any, args: dict[str, Any]
) -> dict[str, Any]:
    ret = {}
    m, n, k = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={m}, N={n}, K={k}]"

    bytes_per_elem = args["c_ptr"].element_size()
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * m * n * k
    ret["bytes"] = bytes_per_elem * (m * k + n * k + m * n)
    return ret


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    bias_ptr,
    M,
    N,
    K,  #
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bias,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        if A_LARGE:
            offs_am = offs_am.to(tl.int64)
        if B_LARGE:
            offs_bn = offs_bn.to(tl.int64)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            if A_LARGE or B_LARGE:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
            else:
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (
                offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
            )
            b_ptrs = b_ptr + (
                offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
            )

            a = tl.load(
                a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            b = tl.load(
                b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            accumulator = tl.dot(a, b, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M
        )
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        if C_LARGE:
            offs_cm = offs_cm.to(tl.int64)
            offs_cn = offs_cn.to(tl.int64)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if HAS_BIAS:
            bias_ptrs = bias_ptr + offs_cn * stride_bias
            bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0).to(tl.float32)
            accumulator += bias
        c = accumulator.to(c_ptr.dtype.element_ty)
        tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def bmm_kernel(
    a_ptr,  # (*, ) pointer to A, (B, M, K)
    b_ptr,  # (*, ) pointer to B, (B, K, N)
    c_ptr,  # (*, ) pointer to C, (B, M, N)
    B,  # int, batch size
    M,  # int, output rows
    N,  # int, output cols
    K,  # int, reduction dim
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
):
    """Batched GEMM: (B, M, K) x (B, K, N) -> (B, M, N).

    Each program computes one (batch_idx, tile_m, tile_n) tile, accumulating
    along K in a fixed order to preserve batch invariance.
    """
    pid_b = tl.program_id(0)
    pid = tl.program_id(1)

    if pid_b >= B:
        return

    # number of tiles along M / N
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    if pid_m >= num_pid_m or pid_n >= num_pid_n:
        return

    # offs_m / offs_n: raw global row/col indices for this tile
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # masks for valid logical rows/cols within (M, N)
    mask_m = offs_m < M  # [BLOCK_SIZE_M]
    mask_n = offs_n < N  # [BLOCK_SIZE_N]

    if A_LARGE or B_LARGE or C_LARGE:
        offs_m = offs_m.to(tl.int64)
        offs_n = offs_n.to(tl.int64)

    offs_m = tl.where(mask_m, offs_m, 0)
    offs_n = tl.where(mask_n, offs_n, 0)

    # hint for triton contiguous memory
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)

    # base pointers for current batch, shape-wise:
    #   a_batch_ptr points to A[pid_b, 0, 0]
    #   b_batch_ptr points to B[pid_b, 0, 0]
    #   c_batch_ptr points to C[pid_b, 0, 0]
    a_batch_ptr = a_ptr + pid_b * stride_ab
    b_batch_ptr = b_ptr + pid_b * stride_bb
    c_batch_ptr = c_ptr + pid_b * stride_cb

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # number of K-blocks this tile iterates over
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    offs_k_mask = tl.arange(0, BLOCK_SIZE_K)

    for ki in range(k_tiles):
        if A_LARGE or B_LARGE:
            # offs_k: [BLOCK_SIZE_K], global K indices
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
        else:
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        # a_ptrs: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        #   element (i, j) points to A[pid_b, offs_m[i], offs_k[j]]
        a_ptrs = a_batch_ptr + (
            offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        )
        # b_ptrs: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        #   element (i, j) points to B[pid_b, offs_k[i], offs_n[j]]
        b_ptrs = b_batch_ptr + (
            offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        )

        # valid K lanes for this block
        k_valid = offs_k_mask < (K - ki * BLOCK_SIZE_K)
        # A mask within (M, K): [BLOCK_SIZE_M, BLOCK_SIZE_K]
        a_mask = mask_m[:, None] & k_valid[None, :]
        # B mask within (K, N): [BLOCK_SIZE_K, BLOCK_SIZE_N]
        b_mask = k_valid[:, None] & mask_n[None, :]

        # a: [BLOCK_SIZE_M, BLOCK_SIZE_K] from A[offs_m, offs_k]
        a = tl.load(
            a_ptrs,
            mask=a_mask,
            other=0.0,
        )
        # b: [BLOCK_SIZE_K, BLOCK_SIZE_N] from B[offs_k, offs_n]
        b = tl.load(
            b_ptrs,
            mask=b_mask,
            other=0.0,
        )
        accumulator = tl.dot(a, b, accumulator)

    # c_m / c_n: [BLOCK_SIZE_M] / [BLOCK_SIZE_N], row/col indices for C
    c_m = offs_m
    c_n = offs_n
    if C_LARGE:
        c_m = c_m.to(tl.int64)
        c_n = c_n.to(tl.int64)

    # c_ptrs: [BLOCK_SIZE_M, BLOCK_SIZE_N]
    #   element (i, j) points to C[pid_b, c_m[i], c_n[j]]
    c_ptrs = c_batch_ptr + stride_cm * c_m[:, None] + stride_cn * c_n[None, :]
    # mask out elements that fall outside logical (M, N) range
    c_mask = mask_m[:, None] & mask_n[None, :]
    # cast FP32 accumulator back to original dtype of C
    c = accumulator.to(c_ptr.dtype.element_ty)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def bmm_kernel_broadcast(
    a_ptr,  # (*, ) pointer to A, (B, M, K)
    b_ptr,  # (*, ) pointer to B, (B, K, N)
    c_ptr,  # (*, ) pointer to C, (B, M, N)
    B,  # int, batch size
    M,  # int, output rows
    N,  # int, output cols
    K,  # int, reduction dim
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BATCH_SHAPE: tl.constexpr,
    A_BATCH_STRIDES: tl.constexpr,
    B_BATCH_STRIDES: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
):
    """Batched GEMM: (B, M, K) x (B, K, N) -> (B, M, N).

    Each program computes one (batch_idx, tile_m, tile_n) tile, accumulating
    along K in a fixed order to preserve batch invariance.
    """
    pid_b = tl.program_id(0)
    pid = tl.program_id(1)

    if pid_b >= B:
        return

    # number of tiles along M / N
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    if pid_m >= num_pid_m or pid_n >= num_pid_n:
        return

    # offs_m / offs_n: raw global row/col indices for this tile
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # masks for valid logical rows/cols within (M, N)
    mask_m = offs_m < M  # [BLOCK_SIZE_M]
    mask_n = offs_n < N  # [BLOCK_SIZE_N]

    if A_LARGE or B_LARGE or C_LARGE:
        offs_m = offs_m.to(tl.int64)
        offs_n = offs_n.to(tl.int64)

    offs_m = tl.where(mask_m, offs_m, 0)
    offs_n = tl.where(mask_n, offs_n, 0)

    # hint for triton contiguous memory
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_SIZE_N), BLOCK_SIZE_N)

    # base pointers for current batch, shape-wise:
    #   a_batch_ptr points to A[pid_b, 0, 0]
    #   b_batch_ptr points to B[pid_b, 0, 0]
    #   c_batch_ptr points to C[pid_b, 0, 0]
    remaining = pid_b.to(tl.int64)
    offset_a = tl.full((), 0, tl.int64)
    offset_b = tl.full((), 0, tl.int64)
    for dim in tl.static_range(len(BATCH_SHAPE) - 1, -1, -1):
        coordinate = remaining % BATCH_SHAPE[dim]
        remaining = remaining // BATCH_SHAPE[dim]
        offset_a += coordinate * A_BATCH_STRIDES[dim]
        offset_b += coordinate * B_BATCH_STRIDES[dim]
    a_batch_ptr = a_ptr + offset_a
    b_batch_ptr = b_ptr + offset_b
    c_batch_ptr = c_ptr + pid_b * stride_cb

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # number of K-blocks this tile iterates over
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    offs_k_mask = tl.arange(0, BLOCK_SIZE_K)

    for ki in range(k_tiles):
        if A_LARGE or B_LARGE:
            # offs_k: [BLOCK_SIZE_K], global K indices
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
        else:
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        # a_ptrs: [BLOCK_SIZE_M, BLOCK_SIZE_K]
        #   element (i, j) points to A[pid_b, offs_m[i], offs_k[j]]
        a_ptrs = a_batch_ptr + (
            offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        )
        # b_ptrs: [BLOCK_SIZE_K, BLOCK_SIZE_N]
        #   element (i, j) points to B[pid_b, offs_k[i], offs_n[j]]
        b_ptrs = b_batch_ptr + (
            offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        )

        # valid K lanes for this block
        k_valid = offs_k_mask < (K - ki * BLOCK_SIZE_K)
        # A mask within (M, K): [BLOCK_SIZE_M, BLOCK_SIZE_K]
        a_mask = mask_m[:, None] & k_valid[None, :]
        # B mask within (K, N): [BLOCK_SIZE_K, BLOCK_SIZE_N]
        b_mask = k_valid[:, None] & mask_n[None, :]

        # a: [BLOCK_SIZE_M, BLOCK_SIZE_K] from A[offs_m, offs_k]
        a = tl.load(
            a_ptrs,
            mask=a_mask,
            other=0.0,
        )
        # b: [BLOCK_SIZE_K, BLOCK_SIZE_N] from B[offs_k, offs_n]
        b = tl.load(
            b_ptrs,
            mask=b_mask,
            other=0.0,
        )
        accumulator = tl.dot(a, b, accumulator)

    # c_m / c_n: [BLOCK_SIZE_M] / [BLOCK_SIZE_N], row/col indices for C
    c_m = offs_m
    c_n = offs_n
    if C_LARGE:
        c_m = c_m.to(tl.int64)
        c_n = c_n.to(tl.int64)

    # c_ptrs: [BLOCK_SIZE_M, BLOCK_SIZE_N]
    #   element (i, j) points to C[pid_b, c_m[i], c_n[j]]
    c_ptrs = c_batch_ptr + stride_cm * c_m[:, None] + stride_cn * c_n[None, :]
    # mask out elements that fall outside logical (M, N) range
    c_mask = mask_m[:, None] & mask_n[None, :]
    # cast FP32 accumulator back to original dtype of C
    c = accumulator.to(c_ptr.dtype.element_ty)
    tl.store(c_ptrs, c, mask=c_mask)


def _broadcast_shape(left, right):
    """Compute only the leading broadcast shape using Python integers."""
    rank = max(len(left), len(right))
    left = (1,) * (rank - len(left)) + tuple(left)
    right = (1,) * (rank - len(right)) + tuple(right)
    result = []
    for a, b in zip(left, right):
        if a != b and a != 1 and b != 1:
            raise RuntimeError(f"Batch shapes do not broadcast: {left}, {right}")
        result.append(b if a == 1 else a)
    return tuple(result)


def _batch_flatten_requires_copy(value):
    """Test batch-stride affine layout without constructing per-axis lists."""
    expected_stride = None
    for size, stride in zip(reversed(value.shape[:-2]), reversed(value.stride()[:-2])):
        if size <= 1:
            continue
        if expected_stride is not None and stride != expected_stride:
            return True
        expected_stride = stride * size
    return False


def _prepare_batch_operand(
    value, source_shape, batch_shape, batch, source_batch, expanded
):
    m, k = source_shape[-2:]
    if source_batch == batch:
        # Only singleton axes differ: flatten the existing batch directly.
        return value if value.ndim == 3 else value.reshape(batch, m, k)
    if source_batch == 1:
        # Preserve storage offset and matrix strides; no expand/stride scan needed.
        return value.as_strided(
            (batch, m, k),
            (0, value.stride(-2), value.stride(-1)),
            storage_offset=value.storage_offset(),
        )
    if expanded is None:
        expanded = value.expand(tuple(batch_shape) + source_shape[-2:])
    return expanded.reshape(batch, m, k)


_SMALL_MM = dict(
    BLOCK_SIZE_M=32,
    BLOCK_SIZE_N=32,
    BLOCK_SIZE_K=64,
    GROUP_SIZE_M=8,
    num_warps=4,
    num_stages=3,
)
_SMALL_BMM = dict(
    BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=64, num_warps=4, num_stages=3
)


@cache
def _device_policy(index):
    """Resolve once per input device; never query CUDA during module import."""
    props = torch.cuda.get_device_properties(index)
    shared = triton.runtime.driver.active.utils.get_device_properties(index)[
        "max_shared_mem"
    ]
    family = {9: "hopper", 10: "blackwell", 12: "sm120"}.get(props.major)
    if (props.major, props.minor) == (8, 9):
        family = "ada"
    return dict(
        h20=(props.major, props.minor) == (9, 0) and "H20" in props.name.split(),
        sms=props.multi_processor_count,
        supported=props.major >= 8,
        fp16_n=256 if shared > 106496 else 128,
        fp32_n=128 if shared > 106496 else 32,
        fp32_stages=3 if shared > 106496 else 2,
        tuned=_BATCH_INVARIANT_MATMUL_TUNED_CONFIGS.get(family, {}),
    )


def _small_scope(dtype, M, N, K, policy):
    return (
        policy["h20"]
        and dtype in (torch.float16, torch.bfloat16)
        and 0 < M <= 128
        and N in (16, 32)
        and K in (16, 32, 64)
    )


def _config(dtype, M, N, K, policy, *, batched):
    if _small_scope(dtype, M, N, K, policy):
        return _SMALL_BMM if batched else _SMALL_MM
    cfg = dict(
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, num_warps=8, num_stages=3
    )
    if dtype == torch.float16:
        cfg["BLOCK_SIZE_N"] = policy["fp16_n"]
    elif dtype == torch.float32:
        cfg["BLOCK_SIZE_K"] = 32
        if N == 1:
            cfg.update(BLOCK_SIZE_N=policy["fp32_n"], num_stages=policy["fp32_stages"])
    if not batched:
        cfg["GROUP_SIZE_M"] = 8
        tuned = policy["tuned"].get((N, K)) if dtype == torch.bfloat16 else None
        if tuned is not None:
            m_config = tuned.m_buckets[-1][1]
            for max_m, bucket_config in tuned.m_buckets:
                if max_m >= M:
                    m_config = bucket_config
                    break
            cfg.update(
                BLOCK_SIZE_M=m_config.block_m,
                BLOCK_SIZE_N=m_config.block_n,
                BLOCK_SIZE_K=tuned.block_k,
                num_warps=m_config.num_warps,
                num_stages=m_config.num_stages,
            )
    return cfg


def _validate(a, b, out):
    if a.ndim < 2 or b.ndim < 2:
        raise ValueError("Both inputs must have rank at least two")
    if a.device != b.device:
        raise ValueError("Inputs must be on the same device")
    if a.dtype != b.dtype:
        raise ValueError("Inputs must have identical dtypes")
    if a.layout != torch.strided or b.layout != torch.strided:
        raise NotImplementedError("Only strided tensors are supported")
    if a.device.type != "cuda" or torch.version.hip is not None:
        raise NotImplementedError("This port supports NVIDIA CUDA only")
    if a.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError("Supported dtypes are FP16, BF16 and FP32")
    if a.requires_grad or b.requires_grad:
        raise NotImplementedError("This operator supports forward inference only")
    if a.shape[-1] != b.shape[-2]:
        raise ValueError("Incompatible inner dimensions")
    shape = _broadcast_shape(a.shape[:-2], b.shape[:-2]) + (a.shape[-2], b.shape[-1])
    if out is not None:
        if tuple(out.shape) != shape or out.dtype != a.dtype or out.device != a.device:
            raise ValueError("out must match output shape, dtype and device")
        if out.layout != torch.strided or out.requires_grad:
            raise ValueError("out must be a strided inference tensor")
        span = 1
        for stride, size in sorted(zip(out.stride(), out.shape)):
            if size <= 1:
                continue
            if stride < span:
                raise ValueError("out must have a provably non-overlapping layout")
            span += (size - 1) * stride
        if out.numel() and any(
            out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
            for x in (a, b)
        ):
            raise ValueError("out must not share input storage")
    return shape


def _mm(a, b, policy):
    M, K = a.shape
    N = b.shape[1]
    cfg = _config(a.dtype, M, N, K, policy, batched=False)
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (
        min(
            policy["sms"],
            triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]),
        ),
    )
    matmul_kernel_persistent[grid](
        a,
        b,
        c,
        None,
        M,
        N,
        K,
        *a.stride(),
        *b.stride(),
        *c.stride(),
        1,
        NUM_SMS=policy["sms"],
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        HAS_BIAS=False,
        **cfg,
    )
    return c


def _bmm(a, b, policy):
    B, M, K = a.shape
    N = b.shape[-1]
    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    cfg = _config(a.dtype, M, N, K, policy, batched=True)
    grid = (
        B,
        triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]),
    )
    bmm_kernel[grid](
        a,
        b,
        c,
        B,
        M,
        N,
        K,
        *a.stride(),
        *b.stride(),
        *c.stride(),
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        **cfg,
    )
    return c


def _broadcast_bmm(a, b, shape, policy):
    B = math.prod(shape)
    M, K = a.shape[-2:]
    N = b.shape[-1]
    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    cfg = _config(a.dtype, M, N, K, policy, batched=True)
    grid = (
        B,
        triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]),
    )
    bmm_kernel_broadcast[grid](
        a,
        b,
        c,
        B,
        M,
        N,
        K,
        a.stride(-2),
        a.stride(-1),
        b.stride(-2),
        b.stride(-1),
        *c.stride(),
        BATCH_SHAPE=tuple(shape),
        A_BATCH_STRIDES=tuple(a.stride()[:-2]),
        B_BATCH_STRIDES=tuple(b.stride()[:-2]),
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        **cfg,
    )
    return c.reshape(tuple(shape) + (M, N))


def _optimized_bmm(a, b, shape, policy):
    batch = math.prod(shape)
    a_shape, b_shape = tuple(a.shape), tuple(b.shape)
    a_batch, b_batch = math.prod(a_shape[:-2]), math.prod(b_shape[:-2])
    a_expanded = b_expanded = None
    a_copies = b_copies = False
    if a_batch > 1:
        if a_batch != batch:
            a_expanded = a.expand(shape + a_shape[-2:])
            a_copies = not a_expanded.is_contiguous() and _batch_flatten_requires_copy(
                a_expanded
            )
        elif a.ndim > 3 and not a.is_contiguous():
            a_copies = _batch_flatten_requires_copy(a)
    if b_batch > 1:
        if b_batch != batch:
            b_expanded = b.expand(shape + b_shape[-2:])
            b_copies = not b_expanded.is_contiguous() and _batch_flatten_requires_copy(
                b_expanded
            )
        elif b.ndim > 3 and not b.is_contiguous():
            b_copies = _batch_flatten_requires_copy(b)
    if a_copies or b_copies:
        if a_expanded is None:
            a_expanded = (
                a if a_shape == shape + a_shape[-2:] else a.expand(shape + a_shape[-2:])
            )
        if b_expanded is None:
            b_expanded = (
                b if b_shape == shape + b_shape[-2:] else b.expand(shape + b_shape[-2:])
            )
        return _broadcast_bmm(a_expanded, b_expanded, shape, policy)
    a3 = _prepare_batch_operand(a, a_shape, shape, batch, a_batch, a_expanded)
    b3 = _prepare_batch_operand(b, b_shape, shape, batch, b_batch, b_expanded)
    return _bmm(a3, b3, policy).reshape(shape + (a.shape[-2], b.shape[-1]))


def _matmul(a, b, shape):
    if math.prod(shape) == 0:
        return torch.empty(shape, device=a.device, dtype=a.dtype)
    if a.shape[-1] == 0:
        return torch.zeros(shape, device=a.device, dtype=a.dtype)
    policy = _device_policy(a.device.index)
    if not policy["supported"]:
        raise NotImplementedError("NVIDIA compute capability SM80 or newer is required")
    if b.ndim == 2:
        flat = a.reshape(-1, a.shape[-1])
        return _mm(flat, b, policy).reshape(shape)
    batch_shape = shape[:-2]
    M, K, N = a.shape[-2], a.shape[-1], b.shape[-1]
    if _small_scope(a.dtype, M, N, K, policy):
        return _optimized_bmm(a, b, batch_shape, policy)
    # Original expand/reshape + original BMM kernel/config outside the fast domain.
    batch = math.prod(batch_shape)
    a3 = a.expand(batch_shape + tuple(a.shape[-2:])).reshape(batch, M, K)
    b3 = b.expand(batch_shape + tuple(b.shape[-2:])).reshape(batch, K, N)
    return _bmm(a3, b3, policy).reshape(shape)


def matmul_persistent(a, b, *, out=None):
    """vLLM matmul_batch_invariant(a, b, *, out=None), forward NVIDIA port.

    Only the bounded H20 FP16/BF16 small-matrix domain enables the new batch
    preparation and broadcast kernel. No ATen dispatch registration is performed.
    """
    shape = _validate(a, b, out)
    if torch.cuda.current_device() != a.device.index:
        with torch.cuda.device(a.device):
            result = _matmul(a, b, shape)
    else:
        result = _matmul(a, b, shape)
    if out is not None:
        out.copy_(result)
        return out
    return result
