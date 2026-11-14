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

import torch


def v2_apply_penalties(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    token_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    repetition_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
) -> None:
    """Apply vLLM V2 repetition, frequency and presence penalties in place.

    Logits are FP16, BF16, or FP32 with shape [tokens, vocabulary], contiguous
    columns and optional row padding. Request vectors contain FP32 penalties.
    Per-request prompt bits (32 tokens per word) and output counts are int32
    matrices with contiguous columns. Integer token metadata uses the same
    flattened request/position convention as vLLM's V2 batch. For draft rows,
    prior draft tokens contribute to output counts only for that row. Counts,
    masks and metadata are read-only. All operands are on the same Ascend
    device. Empty logits need no launch.
    """
    raise NotImplementedError("v2_apply_penalties is supported on Ascend")
