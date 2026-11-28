# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""MThreads inference FlashAttention: validation, scheduling and Triton kernels."""

from flaggems_vllm.runtime.backend._mthreads.ops.flash_attention.launcher import (
    launch_attention,
)

__all__ = ["launch_attention"]
