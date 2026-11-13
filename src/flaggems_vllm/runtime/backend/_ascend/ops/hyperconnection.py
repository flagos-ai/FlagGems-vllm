# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Compatibility adapter for the legacy Qwen4 inject-combine API."""

from __future__ import annotations

import torch

from flaggems_vllm.ops.qwen4.hc_combine import _combine


def qwen4_hc_inject_combine(
    injection_logits: torch.Tensor,
    block_output: torch.Tensor,
    residual: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    return _combine(residual, block_output, injection_logits, hc_count, legacy=True)
