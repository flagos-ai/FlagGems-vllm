# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone snapshot of vLLM persistent matmul and its test dependencies.

Kernels, wrappers, tuned configurations and mode initialization are copied
unchanged. Small platform helpers replace vLLM imports. The auxiliary BMM,
descriptor, softmax and mean definitions preserve the original test/setup paths.
No installed vLLM or FlagGems package is required.
"""

import importlib.metadata
import math
import os
from collections.abc import Callable
from functools import cache
from typing import Any, NamedTuple

import torch
import triton
import triton.language as tl
from packaging import version
from packaging.version import Version


class DeviceCapability(NamedTuple):
    major: int
    minor: int


class _StandalonePlatform:
    """Only the vLLM platform queries used by the copied definitions/tests."""

    @staticmethod
    def is_cuda():
        return torch.version.hip is None and torch.cuda.is_available()

    @staticmethod
    def is_rocm():
        return torch.version.hip is not None and torch.cuda.is_available()

    @staticmethod
    def is_xpu():
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    def is_cuda_alike(self):
        return self.is_cuda() or self.is_rocm()

    @property
    def device_type(self):
        if self.is_cuda_alike():
            return "cuda"
        return "xpu" if self.is_xpu() else "cpu"

    @property
    def dispatch_key(self):
        return "XPU" if self.is_xpu() and not self.is_cuda_alike() else "CUDA"

    def get_device_capability(self, device_id=0):
        if not self.is_cuda():
            return None
        return DeviceCapability(*torch.cuda.get_device_capability(device_id))

    def is_device_capability_family(self, capability, device_id=0):
        actual = self.get_device_capability(device_id)
        return actual is not None and actual.major == capability // 10

    def has_device_capability(self, capability, device_id=0):
        actual = self.get_device_capability(device_id)
        return actual is not None and actual.major * 10 + actual.minor >= capability


current_platform = _StandalonePlatform()


@cache
def num_compute_units(device_id: int = 0) -> int:
    if current_platform.is_xpu() and not current_platform.is_cuda_alike():
        return torch.xpu.get_device_properties(device_id).max_compute_units
    return torch.cuda.get_device_properties(device_id).multi_processor_count


@cache
def get_max_shared_memory_bytes(gpu: int = 0) -> int:
    # Triton's driver queries the same CUDA OPTIN / HIP device attribute as vLLM.
    max_shared_mem = triton.runtime.driver.active.utils.get_device_properties(gpu)[
        "max_shared_mem"
    ]
    assert max_shared_mem > 0, "max_shared_mem cannot be zero"
    return int(max_shared_mem)


# Copied from vllm/utils/torch_version.py.


def _is_torch_equal_or_newer(torch_version: str, target: str) -> bool:
    return version.parse(torch_version) >= version.parse(target)


def is_torch_equal_or_newer(target: str) -> bool:
    """Check if the installed torch version is >= the target version
    (a string like "2.6.0").
    """
    try:
        return _is_torch_equal_or_newer(str(torch.__version__), target)
    except Exception:
        # Fallback to PKG-INFO to load the package info, needed by the doc gen.
        return Version(importlib.metadata.version("torch")) >= Version(target)


# Copied from vllm/model_executor/determinism/batch_invariant_configs.py.


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


_TUNED_MATMUL_CONFIGS_FOR_DEVICE: dict[tuple[int, int], _MatmulShapeConfig] | None = (
    None
)


_TUNED_MATMUL_CONFIGS_RESOLVED = False


def _get_tuned_matmul_arch_family(capability: DeviceCapability | None) -> str | None:
    if capability is None:
        return None
    if capability.major == 12:
        return "sm120"
    if capability.major == 10:
        return "blackwell"
    if capability.major == 9:
        return "hopper"
    if capability.major == 8 and capability.minor == 9:
        return "ada"
    return None


def resolve_tuned_matmul_configs() -> None:
    global _TUNED_MATMUL_CONFIGS_FOR_DEVICE
    global _TUNED_MATMUL_CONFIGS_RESOLVED

    if _TUNED_MATMUL_CONFIGS_RESOLVED:
        return

    # Normal vLLM initialization resolves this before torch.compile tracing.
    capability = (
        current_platform.get_device_capability() if current_platform.is_cuda() else None
    )
    arch_family = _get_tuned_matmul_arch_family(capability)
    if arch_family is None:
        _TUNED_MATMUL_CONFIGS_FOR_DEVICE = None
    else:
        _TUNED_MATMUL_CONFIGS_FOR_DEVICE = _BATCH_INVARIANT_MATMUL_TUNED_CONFIGS.get(
            arch_family
        )
    _TUNED_MATMUL_CONFIGS_RESOLVED = True


def _get_matmul_config(
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    default: dict[str, int],
) -> dict[str, int]:
    if not _TUNED_MATMUL_CONFIGS_RESOLVED:
        resolve_tuned_matmul_configs()
    if dtype != torch.bfloat16:
        return default
    device_configs = _TUNED_MATMUL_CONFIGS_FOR_DEVICE
    if device_configs is None:
        return default
    shape_config = device_configs.get((N, K))
    if shape_config is None:
        return default
    # Values above the tuned range reuse the largest bucket;
    # shape-wide BLOCK_K keeps this batch-invariant.
    m_config = shape_config.m_buckets[-1][1]
    for max_m, bucket_config in shape_config.m_buckets:
        if max_m >= M:
            m_config = bucket_config
            break
    return {
        "BLOCK_SIZE_M": m_config.block_m,
        "BLOCK_SIZE_N": m_config.block_n,
        "BLOCK_SIZE_K": shape_config.block_k,
        "GROUP_SIZE_M": 8,
        "num_warps": m_config.num_warps,
        "num_stages": m_config.num_stages,
    }


def _get_descriptor_matmul_config(
    M: int, N: int, K: int, dtype: torch.dtype
) -> dict[str, int]:
    """Select a config for the XPU descriptor matmul from shape and dtype.

    The default 128×128×64 config is optimized for large square-ish GEMMs but
    wastes resources on skinny shapes (small M during decode, or very tall/thin
    matrices during prefill).  Shape-dependent tuning closes the gap with oneDNN.

    Only BLOCK_SIZE_M/N and the launch parameters vary with M; dtype alone
    decides BLOCK_SIZE_K, so the K-reduction order never depends on batch size.
    """
    # fp32 uses smaller BLOCK_SIZE_K due to register pressure
    block_k = 32 if dtype == torch.float32 else 64

    if M <= 16:
        # Decode: M=1-16. Tiny M means most of a 128-row tile is wasted.
        # Use small M-block, wide N-block to maximize useful work per tile.
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": 1,
            "num_stages": 4,
            "num_warps": 8,
        }
    elif M <= 64:
        # Small batch decode or very short prefill.
        return {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": 4,
            "num_stages": 4,
            "num_warps": 8,
        }
    else:
        # Medium and large prefill (M > 64). 64×128 tiles provide the best
        # balance of register pressure vs parallelism on Intel XPU.
        # M=2048, N=4096 → 32×32 = 1024 tiles, well above ~160 compute units.
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        }


# Copied from vllm/model_executor/determinism/batch_invariant.py.


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


@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_descriptor_persistent(
    a_ptr,
    b_ptr,
    c_ptr,  #
    bias_ptr,
    M,
    N,
    K,  #
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_SMS: tl.constexpr,  #
    HAS_BIAS: tl.constexpr,
):
    """Persistent matmul using tensor descriptors for 2D block I/O.

    Expects b_ptr to point to a transposed B matrix of shape [N, K] with
    row-major (K-contiguous) layout.  The dot product transposes each loaded
    B-tile back: dot(A_tile, B_tile.T).

    ~3x faster than the pointer-based persistent kernel on Intel XPU because
    tensor descriptors leverage hardware 2D block load/store with automatic
    bounds checking (no explicit masks needed).
    """
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M
        )
        offs_cm = pid_m * BLOCK_SIZE_M
        offs_cn = pid_n * BLOCK_SIZE_N

        if HAS_BIAS:
            bias_offs = offs_cn + tl.arange(0, BLOCK_SIZE_N)
            bias_vals = tl.load(bias_ptr + bias_offs, mask=bias_offs < N, other=0.0).to(
                tl.float32
            )
            accumulator += bias_vals[None, :]

        c = accumulator.to(dtype)
        c_desc.store([offs_cm, offs_cn], c)


def matmul_descriptor_persistent(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Persistent matmul using tensor descriptors (Intel XPU fast path).

    Args:
        a: Input matrix [M, K], must be contiguous.
        b: Weight matrix [K, N] (standard layout — transposed internally).
        bias: Optional 1D bias vector [N].

    Returns:
        Output matrix [M, N] with dtype matching the inputs.

    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert bias is None or bias.dim() == 1, "bias must be 1D"

    M, K = a.shape
    _, N = b.shape
    dtype = a.dtype

    # Tensor descriptors require contiguous row-major layout.
    a = a.contiguous()
    # Descriptor kernel expects B in [N, K] layout (K-contiguous).
    b_t = b.t().contiguous()

    c = torch.empty((M, N), device=a.device, dtype=dtype)
    # This path is XPU-only. _NUM_SMS is set by enable_batch_invariant_mode()
    # before torch.compile traces this function, so Dynamo can lift it
    # as a compile-time constant.
    NUM_SMS = _NUM_SMS if _NUM_SMS > 0 else num_compute_units(a.device.index)

    def grid(META):
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            ),
        )

    cfg = _get_descriptor_matmul_config(M, N, K, dtype)

    matmul_kernel_descriptor_persistent[grid](
        a,
        b_t,
        c,
        bias,
        M,
        N,
        K,
        NUM_SMS=NUM_SMS,
        HAS_BIAS=bias is not None,
        **cfg,
    )
    return c


def matmul_persistent(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert (
        bias is None or bias.dim() == 1
    ), "Currently assuming bias is 1D, let Horace know if you run into this"
    NUM_SMS = num_compute_units(a.device.index)
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype)

    # 1D launch kernel where each block gets its own program.
    def grid(META):
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            ),
        )

    if not _BLOCK_SIZES_RESOLVED:
        _resolve_block_sizes()
    configs = {
        torch.bfloat16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": _fp16_block_size_n,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float32: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": _fp32_block_size_n if N == 1 else 128,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_stages": _fp32_num_stages if N == 1 else 3,
            "num_warps": 8,
        },
    }
    matmul_kernel_persistent[grid](
        a,
        b,
        c,  #
        bias,
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        bias.stride(0) if bias is not None else 1,
        NUM_SMS=NUM_SMS,  #
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        HAS_BIAS=bias is not None,
        **_get_matmul_config(M, N, K, dtype, configs[dtype]),
    )
    return c


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
def _softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    input_col_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    LOG: tl.constexpr = False,
):
    """Compute softmax, or log_softmax when ``LOG``, along the last dimension of a
    2D tensor. Each block handles one row of the input tensor, in a fixed block
    order, so the result never depends on the row count.
    """
    # Get the row index for this block
    row_idx = tl.program_id(0).to(tl.int64)

    # Compute base pointers for input and output rows
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Step 1: Find maximum value in the row for numerical stability
    max_val = -float("inf")
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(
            row_start_ptr + col_idx * input_col_stride, mask=mask, other=-float("inf")
        )

        # Update maximum
        max_val = tl.max(tl.maximum(vals, max_val))

    # Step 2: Compute sum of exp(x - max_val)
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx * input_col_stride, mask=mask, other=0.0)

        # Compute exp(x - max_val) and accumulate
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0))

    log_sum_exp = tl.log(sum_exp) if LOG else 0.0

    # Step 3: normalise
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx * input_col_stride, mask=mask)

        if LOG:
            output = vals - max_val - log_sum_exp
        else:
            output = tl.exp(vals - max_val) / sum_exp

        # Store results
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)


def log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute log_softmax using Triton kernel.

    Args:
        input: Input tensor
        dim: Dimension along which to compute log_softmax
             (only -1 or last dim supported)

    Returns:
        Tensor with log_softmax applied along the specified dimension

    """
    if dim != -1 and dim != input.ndim - 1:
        raise ValueError(
            "This implementation only supports log_softmax along the last dimension"
        )

    # Flatten all dimensions except the last one
    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])

    n_rows, n_cols = input_2d.shape

    # Allocate output tensor
    output = torch.empty((n_rows, n_cols), dtype=input_2d.dtype, device=input_2d.device)

    # Choose block size based on the number of columns
    BLOCK_SIZE = 1024

    # Launch kernel with one block per row
    grid = (n_rows,)
    _softmax_kernel[grid](
        input_2d,
        output,
        input_2d.stride(0),
        input_2d.stride(1),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        LOG=True,
    )
    # Reshape output back to original shape
    return output.reshape(original_shape)


def softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute softmax along the last dimension using a Triton kernel."""
    if dim != -1 and dim != input.ndim - 1:
        raise ValueError(
            "This implementation only supports softmax along the last dimension"
        )

    original_shape = input.shape
    input_2d = input.reshape(-1, input.shape[-1])
    n_rows, n_cols = input_2d.shape
    output = torch.empty((n_rows, n_cols), dtype=input_2d.dtype, device=input_2d.device)

    _softmax_kernel[(n_rows,)](
        input_2d,
        output,
        input_2d.stride(0),
        input_2d.stride(1),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=1024,
    )
    return output.reshape(original_shape)


@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    input_stride0,
    input_stride1,
    input_stride2,
    output_stride0,
    output_stride1,
    M,  # size before reduction dim
    N,  # size of reduction dim
    K,  # size after reduction dim
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel for computing mean along a single dimension.
    Input is viewed as (M, N, K) where N is the dimension being reduced.
    """
    # Program ID gives us which output element we're computing
    pid = tl.program_id(0)

    # Compute output indices
    m_idx = pid // K
    k_idx = pid % K

    # Bounds check
    if m_idx >= M or k_idx >= K:
        return

    # Accumulate sum across reduction dimension
    acc = 0.0
    for n_start in range(0, N, BLOCK_SIZE):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N

        # Calculate input indices
        input_idx = (
            m_idx * input_stride0 + n_offsets * input_stride1 + k_idx * input_stride2
        )

        # Load and accumulate
        vals = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        acc += tl.sum(vals)

    # Compute mean and store
    mean_val = acc / N
    output_idx = m_idx * output_stride0 + k_idx * output_stride1
    tl.store(output_ptr + output_idx, mean_val)


def mean_dim(
    input: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Triton implementation of torch.mean with single dimension reduction.

    Args:
        input: Input tensor
        dim: Single dimension along which to compute mean
        keepdim: Whether to keep the reduced dimension
        dtype: Output dtype. If None, uses input dtype
               (or float32 for integer inputs)

    Returns:
        Tensor with mean values along specified dimension

    """
    # Validate inputs
    assert (
        -input.ndim <= dim < input.ndim
    ), f"Invalid dimension {dim} for tensor with {input.ndim} dimensions"

    # Handle negative dim
    if dim < 0:
        dim = dim + input.ndim

    # Handle dtype
    if dtype is None:
        if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            dtype = torch.float32
        else:
            dtype = input.dtype

    # Convert input to appropriate dtype if needed
    if input.dtype != dtype:
        input = input.to(dtype)

    # Get input shape and strides
    shape = list(input.shape)

    # Calculate dimensions for kernel
    M = 1
    for i in range(dim):
        M *= shape[i]

    N = shape[dim]

    K = 1
    for i in range(dim + 1, len(shape)):
        K *= shape[i]

    # Reshape input to 3D view (M, N, K)
    input_3d = input.reshape(M, N, K)

    # Create output shape
    if keepdim:
        output_shape = shape.copy()
        output_shape[dim] = 1
    else:
        output_shape = shape[:dim] + shape[dim + 1 :]

    # Create output tensor
    output = torch.empty(output_shape, dtype=dtype, device=input.device)

    # Reshape output for kernel
    output_2d = output.reshape(M, 1, K).squeeze(1) if keepdim else output.reshape(M, K)

    # Launch kernel
    grid = (M * K,)
    BLOCK_SIZE = 1024

    mean_kernel[grid](
        input_3d,
        output_2d,
        input_3d.stride(0),
        input_3d.stride(1),
        input_3d.stride(2),
        output_2d.stride(0),
        output_2d.stride(1) if output_2d.ndim > 1 else 0,
        M,
        N,
        K,
        BLOCK_SIZE,
    )

    return output


def mm_batch_invariant(a, b, bias=None):
    if a.device.type == "xpu":
        # The descriptor kernel reads the bias at unit stride.
        if bias is not None:
            bias = bias.contiguous()
        return matmul_descriptor_persistent(a, b, bias=bias)
    return matmul_persistent(a, b, bias=bias)


def matmul_batch_invariant(a, b, *, out=None):
    # torch.matmul can handle various dimensions
    # For 2D x 2D, it's the same as mm
    if a.ndim == 2 and b.ndim == 2:
        result = mm_batch_invariant(a, b)
        if out is not None:
            out.copy_(result)
            return out
        return result
    elif b.ndim == 2:
        # Handle ND x 2D: Common for linear layers
        # (..., batch, seq, hidden) @ (hidden, out) -> (..., batch, seq, out)
        batch_dims = a.shape[:-1]
        hidden = a.shape[-1]
        out_dim = b.shape[-1]
        a_2d = a.reshape(-1, hidden)
        result_2d = mm_batch_invariant(a_2d, b)
        result = result_2d.reshape(batch_dims + (out_dim,))
        if out is not None:
            out.copy_(result)
            return out
        return result
    elif a.ndim >= 2 and b.ndim >= 3:
        # Generic handler for 2D x ND and ND x ND (except 1D)
        # Broadcast dims to ensure both matrices have the same shape
        # If 2D x ND, then unsqueeze to add a dim to a
        if a.ndim == 2:
            a = a.unsqueeze(0)
        broadcast_shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
        a = a.expand(broadcast_shape + a.shape[-2:])
        b = b.expand(broadcast_shape + b.shape[-2:])
        batch_dim = math.prod(broadcast_shape)
        # Reuse broadcast shape to get all dims except mm dims
        a_3d = a.reshape(batch_dim, a.shape[-2], a.shape[-1])
        b_3d = b.reshape(batch_dim, b.shape[-2], b.shape[-1])
        # Do batched matmul
        result_3d = bmm_batch_invariant(a_3d, b_3d)
        # Reshape back to [broadcast_shape, seq_a, seq_b]
        result = result_3d.reshape(broadcast_shape + (a.shape[-2], b.shape[-1]))
        if out is not None:
            out.copy_(result)
            return out
        return result
    else:
        raise ValueError(
            f"matmul_batch_invariant requires both inputs be at least 2D "
            f"got shapes {a.shape} and {b.shape}"
        )


def bmm_batch_invariant(a, b, *, out=None):
    # Batched matrix multiply: (B, M, K) x (B, K, N) -> (B, M, N)
    if not (a.ndim == 3 and b.ndim == 3):
        raise ValueError(
            f"bmm_batch_invariant expects 3D tensors, "
            f"got shapes {a.shape} and {b.shape}"
        )

    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"Batch dimensions of tensors must match, "
            f"but got {a.shape[0]} and {b.shape[0]}."
        )
    if a.shape[2] != b.shape[1]:
        raise ValueError(
            f"Incompatible inner dimensions for matmul: got {a.shape} and {b.shape}."
        )
    if a.dtype != b.dtype:
        raise ValueError(f"Incompatible dtypes: got {a.dtype} and {b.dtype}.")

    B, M, K = a.shape
    _, _, N = b.shape
    dtype = a.dtype

    if out is None:
        c = torch.empty((B, M, N), device=a.device, dtype=dtype)
    else:
        assert out.shape == (B, M, N), "out tensor has incorrect shape"
        assert out.dtype == dtype and out.device == a.device, "out tensor mismatch"
        c = out

    if not _BLOCK_SIZES_RESOLVED:
        _resolve_block_sizes()
    configs = {
        torch.bfloat16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float16: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": _fp16_block_size_n,
            "BLOCK_SIZE_K": 64,
            "num_stages": 3,
            "num_warps": 8,
        },
        torch.float32: {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": _fp32_block_size_n if N == 1 else 128,
            "BLOCK_SIZE_K": 32,
            "num_stages": _fp32_num_stages if N == 1 else 3,
            "num_warps": 8,
        },
    }

    cfg = configs[dtype]
    # grid = (B, num_tiles_per_matrix)
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
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        A_LARGE=a.numel() > 2**31,
        B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31,
        **cfg,
    )

    return c


def _addmm_fused_bias(bias, beta, alpha, n):
    # The kernel adds bias into its fp32 accumulator indexed by output column,
    # so it can only absorb a row vector of width n, unscaled by alpha.
    if beta == 0 or alpha != 1:
        return None
    if bias.dim() != 1 or bias.shape[0] != n:
        return None
    # fp32 keeps the scaling exact; the kernel casts the bias to fp32 anyway.
    return bias if beta == 1 else bias.float() * beta


def _addmm_impl(bias, a, b, beta, alpha, out):
    fused_bias = _addmm_fused_bias(bias, beta, alpha, b.shape[1])
    result = mm_batch_invariant(a, b, bias=fused_bias)
    if fused_bias is None:
        if alpha != 1:
            result = result * alpha
        if beta != 0:
            result = result + (bias if beta == 1 else beta * bias)
    if out is None:
        return result
    return out.copy_(result)


def addmm_batch_invariant(bias, a, b, *, beta=1, alpha=1):
    return _addmm_impl(bias, a, b, beta, alpha, None)


def mm_out_batch_invariant(a, b, *, out):
    return matmul_batch_invariant(a, b, out=out)


def addmm_out_batch_invariant(bias, a, b, *, beta=1, alpha=1, out):
    return _addmm_impl(bias, a, b, beta, alpha, out)


def bmm_out_batch_invariant(a, b, *, out):
    return bmm_batch_invariant(a, b, out=out)


def _log_softmax_batch_invariant(input, dim, _half_to_float):
    if _half_to_float:
        return log_softmax(input.float(), dim=dim)
    return log_softmax(input, dim=dim)


def softmax_batch_invariant(input, dim, half_to_float=False):
    if half_to_float:
        input = input.float()
    if dim == -1 or dim == input.ndim - 1:
        return softmax(input, dim=-1)

    # Off the last dimension this falls back to torch ops, whose reduction split
    # depends on the tensor shape.
    input_max = torch.amax(input, dim=dim, keepdim=True)
    # First subtract max for numerical stability (standard practice)
    input = input - input_max
    exp_x = torch.exp(input)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x


def mean_batch_invariant(input, dim, keepdim=False, dtype: torch.dtype | None = None):
    assert dtype is None or dtype == torch.float32, f"unsupported dtype: {dtype}"

    result = input.to(torch.float32)

    if len(dim) == 0:
        dim = [i for i in range(len(input.shape))]

    # Sort dimensions to reduce from largest to smallest to handle shifting dims
    # during iterative reduction.
    sorted_dims = sorted([d % input.ndim for d in dim], reverse=True)

    # Iteratively apply a deterministic mean.
    for d in sorted_dims:
        result = mean_dim(result, dim=d, keepdim=True)

    if not keepdim:
        # Squeeze the reduced dimensions.
        for d in sorted_dims:
            result = result.squeeze(d)

    return result if dtype is not None else result.to(input.dtype)


def linear_batch_invariant(input, weight, bias=None):
    if bias is not None:
        bias = bias.expand(weight.shape[0])
    output = mm_batch_invariant(
        input.reshape(-1, input.shape[-1]), weight.t(), bias=bias
    )
    return output.view(*input.shape[:-1], output.shape[-1])


_batch_invariant_MODE = False


_batch_invariant_LIB = None


_fp16_block_size_n = 256


_fp32_block_size_n = 128


_fp32_num_stages = 3


_BLOCK_SIZES_RESOLVED = False


_NUM_SMS: int = 0


def _resolve_block_sizes():
    """Size the tiles to the shared memory, to avoid triton OutOfResources.

    Runs when the mode is enabled, or on first use by a caller that never
    enables it, such as a kernel unit test.
    """
    global _fp16_block_size_n, _fp32_block_size_n, _fp32_num_stages
    global _BLOCK_SIZES_RESOLVED
    _BLOCK_SIZES_RESOLVED = True
    if not current_platform.is_cuda_alike():
        return
    if get_max_shared_memory_bytes() > 106496:
        _fp16_block_size_n = 256
        _fp32_block_size_n = 128
        _fp32_num_stages = 3
    else:
        _fp16_block_size_n = 128
        # SM89 N=1 fp32 128/stages=3 compiles to 131072B (>101376).
        # Only N=1 OORs; keep 128/stages=3 for every other N.
        _fp32_block_size_n = 32
        _fp32_num_stages = 2


def enable_batch_invariant_mode():
    global _batch_invariant_MODE, _batch_invariant_LIB
    global _fp16_block_size_n, _fp32_block_size_n, _fp32_num_stages
    global _NUM_SMS

    if _batch_invariant_MODE:
        return

    _batch_invariant_MODE = True
    _batch_invariant_LIB = torch.library.Library("aten", "IMPL")

    key = current_platform.dispatch_key
    if current_platform.is_cuda_alike():
        if current_platform.is_rocm() or current_platform.is_device_capability_family(
            80
        ):
            # Neither cuBLASLt on SM80 (Ampere) nor hipBLASLt can be made batch
            # invariant; install the triton persistent matmul overrides for
            # mm/addmm/matmul/linear.
            _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, key)
            _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, key)
            _batch_invariant_LIB.impl("aten::mm.out", mm_out_batch_invariant, key)
            _batch_invariant_LIB.impl("aten::addmm.out", addmm_out_batch_invariant, key)
            _batch_invariant_LIB.impl("aten::matmul", matmul_batch_invariant, key)
            _batch_invariant_LIB.impl("aten::linear", linear_batch_invariant, key)
        else:
            # Hopper (SM90) and Blackwell (SM100): the only source of batch
            # variance is split-k, which we disable via the cuBLAS workspace
            # config.
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
            os.environ["CUBLASLT_WORKSPACE_SIZE"] = "1"

        _resolve_block_sizes()
    elif current_platform.is_xpu():
        # Tensor descriptors need a global-memory allocator; must be set outside
        # torch.compile regions (triton.set_allocator modifies global state).
        def _triton_alloc_fn(size: int, alignment: int, stream: int | None):
            return torch.empty(size, device="xpu", dtype=torch.int8)

        triton.set_allocator(_triton_alloc_fn)

        _NUM_SMS = num_compute_units(0)
        _batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, key)
        _batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, key)
        _batch_invariant_LIB.impl("aten::mm.out", mm_out_batch_invariant, key)
        _batch_invariant_LIB.impl("aten::addmm.out", addmm_out_batch_invariant, key)
        _batch_invariant_LIB.impl("aten::matmul", matmul_batch_invariant, key)
        _batch_invariant_LIB.impl("aten::linear", linear_batch_invariant, key)

        _fp16_block_size_n = 128

    # Native mean can change reduction order with the row count on XPU too.
    _batch_invariant_LIB.impl("aten::mean.dim", mean_batch_invariant, key)

    # Softmax, log_softmax, and bmm are already batch-invariant on XPU
    # (oneDNN backend produces bitwise-identical results regardless of batch
    # context). Only register these overrides on CUDA where they are needed.
    if not current_platform.is_xpu():
        _batch_invariant_LIB.impl(
            "aten::_log_softmax", _log_softmax_batch_invariant, key
        )
        _batch_invariant_LIB.impl("aten::softmax", softmax_batch_invariant, key)
        _batch_invariant_LIB.impl("aten::_softmax", softmax_batch_invariant, key)
        # torch 2.12+ registers a built-in Triton bmm kernel for CUDA
        # (torch._native.ops.bmm_outer_product), so we need allow_override
        # to replace it at the dispatcher level.
        _batch_invariant_LIB.impl(
            "aten::bmm", bmm_batch_invariant, key, allow_override=True
        )
        _batch_invariant_LIB.impl(
            "aten::bmm.out", bmm_out_batch_invariant, key, allow_override=True
        )

    reduced_precision_val = (
        (False, False) if is_torch_equal_or_newer("2.10.0") else False
    )
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
        reduced_precision_val
    )
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
        reduced_precision_val
    )
    if current_platform.is_cuda():
        torch.backends.cuda.preferred_blas_library(backend="cublaslt")


# Benchmark/collector contract. The candidate never imports this module.
OP_NAME = "matmul_persistent"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False
MUTATED_INPUTS = ()
NATIVE_API = "vllm.model_executor.determinism.batch_invariant.matmul_batch_invariant"
NATIVE_KIND = "ported_vllm_triton_snapshot"
NATIVE_UNAVAILABLE_REASON = None
NATIVE_VERSION = None
NATIVE_SOURCE_SHA256 = (
    "b5edb99d0a1612e2a3245a3024cea885aa82c51215319dba85aedd5dd3f255be"
)
SHAPES = (
    ((32, 64), (64, 16)),
    ((64, 16), (4, 16, 32)),
    ((4, 32, 64), (64, 16)),
    ((1, 4, 32, 64), (64, 16)),
    ((4, 32, 64), (4, 64, 16)),
    ((2, 32, 64), (1, 2, 64, 16)),
    ((1, 2, 32, 64), (2, 64, 16)),
    ((1, 2, 32, 64), (4, 2, 64, 16)),
    ((32, 64), (1, 2, 64, 16)),
    ((32, 64), (1, 2, 2, 64, 16)),
    ((1, 2, 2, 32, 64), (64, 16)),
    ((1, 2, 4, 32, 64), (1, 2, 4, 64, 16)),
)


def native():
    return matmul_batch_invariant


def grid():
    return [dict(a_shape=a, b_shape=b) for a, b in SHAPES]


def key_shape(binding):
    a, b = binding["a_shape"], binding["b_shape"]
    return [len(a), *a, len(b), *b]


def binding_from_shape(shape):
    values = list(shape)
    if not values or any(
        not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in values
    ):
        raise ValueError("Shape descriptor requires positive integers")
    rank = values[0]
    if rank < 2 or len(values) <= rank + 1:
        raise ValueError("Invalid A rank")
    a = tuple(values[1 : rank + 1])
    b_rank = values[rank + 1]
    b = tuple(values[rank + 2 :])
    if b_rank < 2 or len(b) != b_rank or a[-1] != b[-2]:
        raise ValueError("Invalid B rank or inner dimensions")
    torch.broadcast_shapes(a[:-2], b[:-2])
    return dict(a_shape=a, b_shape=b)


def build_inputs(binding, dtype, device):
    if dtype not in DTYPES:
        raise ValueError("Frozen workload supports FP16/BF16")
    generator = torch.Generator(device=device).manual_seed(42)
    options = dict(dtype=dtype, device=device, generator=generator)
    return (
        torch.rand(tuple(binding["a_shape"]), **options),
        torch.rand(tuple(binding["b_shape"]), **options),
    ), {}


def config(binding, dtype):
    if dtype not in DTYPES:
        raise ValueError("Frozen workload supports FP16/BF16")
    a, b = binding["a_shape"], binding["b_shape"]
    shape = tuple(torch.broadcast_shapes(a[:-2], b[:-2])) + (a[-2], b[-1])

    def tensor(s):
        return dict(shape=list(s), dtype=str(dtype))

    return dict(
        inputs=dict(a=dict(tensor(a), fill="rand"), b=dict(tensor(b), fill="rand")),
        outputs=dict(c=tensor(shape)),
        dims=dict(
            descriptor=key_shape(binding),
            seed=42,
            native_api=NATIVE_API,
            shape_key_format="[a.ndim, *a.shape, b.ndim, *b.shape]",
            theoretical_flops=2 * math.prod(shape) * a[-1],
        ),
    )
