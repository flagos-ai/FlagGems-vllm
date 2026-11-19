# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch


def v2_apply_grammar_bitmask(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    logits_indices: torch.Tensor,
    cu_num_logits: torch.Tensor,
    mask_stride: int,
) -> None:
    """Mask V2 structured-output logits in place using packed int32 words."""
    raise NotImplementedError("v2_apply_grammar_bitmask requires Kunlunxin")
