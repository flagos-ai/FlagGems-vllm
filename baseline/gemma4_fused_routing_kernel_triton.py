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


def gemma4_fused_routing_kernel_triton(
    gating_output, topk, per_expert_scale, num_warps=1, *, is_padding=None
):
    """Torch reference for Gemma 4 routing with FP32 weights and int32 IDs."""
    logits = gating_output.float()
    # Stable sorting keeps lower expert IDs first when scores tie at the cutoff.
    scores, ids = torch.sort(logits, dim=-1, descending=True, stable=True)
    scores, ids = scores[..., :topk], ids[..., :topk]
    scales = per_expert_scale.float()[ids]
    weights = torch.softmax(scores, dim=-1) * scales
    ids = ids.int()
    if is_padding is not None:
        weights = weights.masked_fill(is_padding[:, None], 0)
        ids = ids.masked_fill(is_padding[:, None], -1)
    return weights, ids
