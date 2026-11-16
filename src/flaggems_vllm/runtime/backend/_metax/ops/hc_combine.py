# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Canonical metax HC combine entry point."""

from __future__ import annotations

import logging

import torch

from flaggems_vllm.ops.qwen4.hc_combine import _combine

logger = logging.getLogger(__name__)


def hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor | None,
    hc_count: int,
) -> torch.Tensor:
    logger.debug("GEMS_METAX HC_COMBINE")
    return _combine(residual, block_output, injection_logits, hc_count)
