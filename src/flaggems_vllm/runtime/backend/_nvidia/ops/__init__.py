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

from .fused_mm_input_norm import fused_mm_input_norm
from .grouped_topk import grouped_topk
from .ihc_pair import can_ihc_pair, ihc_post_pre_pair, ihc_pre_pair
from .indexer_prefix_topk import indexer_prefix_topk, supports_indexer_prefix_topk
from .int8_indexer_empty_tile import int8_indexer_paged_logits_empty_tile
from .pad_mla_q_sink_slot_bound import pad_mla_q_sink_slot_bound
from .silu_clamp_int8_quant import (
    int8_quantize_per_token,
    silu_clamp,
    silu_clamp_int8_quant,
    supports_int8_quantize_per_token,
    supports_silu_clamp_int8_quant,
)

__all__ = [
    "can_ihc_pair",
    "fused_mm_input_norm",
    "grouped_topk",
    "ihc_post_pre_pair",
    "ihc_pre_pair",
    "indexer_prefix_topk",
    "int8_indexer_paged_logits_empty_tile",
    "pad_mla_q_sink_slot_bound",
    "silu_clamp",
    "silu_clamp_int8_quant",
    "supports_indexer_prefix_topk",
    "supports_silu_clamp_int8_quant",
]


__all__ += ["int8_quantize_per_token", "supports_int8_quantize_per_token"]
