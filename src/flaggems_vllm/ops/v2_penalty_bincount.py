# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import torch


def v2_penalty_bincount(
    expanded_idx_mapping: torch.Tensor,
    all_token_ids: torch.Tensor,
    prompt_len: torch.Tensor,
    prefill_len: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
    max_prefill_len: int,
) -> None:
    """Rebuild the selected V2 request slots' packed prompt and output counts."""
    raise NotImplementedError("v2_penalty_bincount requires Kunlunxin")
