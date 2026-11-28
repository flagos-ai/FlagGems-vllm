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

"""Sparse MLA decode attention over an FP8 paged KV cache, for Ascend 950DT.

Replaces ``cann_ops_transformer.mixed_quant_sparse_flash_mla``
(``aclnnMixedQuantSparseFlashMla``) in the ORI_SPARSE decode form that
vllm-ascend uses for DeepSeek-style MLA inference.

Numerical definition, per query row::

    seed:  row_max = sinks, row_sum = 1, acc = 0
    for each tile of sparse_indices:
        kv     = dequant(gather(ori_kv, block_table))        # bf16, K == V
        scores = q @ kv.T * softmax_scale                    # fp32
        tile_max = max(scores).clamp_min(0)
        online_softmax_update(row_max, row_sum, acc, scores, kv)
    out = (acc / row_sum).to(bf16)

Layout and dequantization (``quant_mode=1``)::

    ori_kv: [block_num, block_size, 1, 544] uint8, PA_BBND
            544B/token = 512B fp8_e4m3 (nope[448] || rope[64])
                       + 32B bf16 scale (16 groups x 32 channels)
    kv[g*32 + i] = fp8[g*32 + i].to(bf16) * scale[g]

Under MLA absorb the dequantized row is both the key and the value, so a tile
is fetched once and feeds ``QK^T`` and ``PV``.  ``sinks`` seeds the running max
with ``row_sum = 1``, which is the sink's own ``exp(sink - sink)`` term.
Probabilities are cast to bf16 before the ``PV`` matmul, matching the vendor's
numerical flow.

This lives under ``a5/`` rather than ``_ascend/ops/`` because the whole design
is specific to the 950DT: the 544B row layout, the 32-Cube grid, and the
216 KB UB budget that fixes the tile sizes.

Coverage: decode (one query token per sequence), ``quant_mode=1``,
``layout_q=TND``, ``layout_kv=PA_BBND``, D=512, rope_head_dim=64,
fp8_e4m3 ``ori_kv``.  The FP4 compressed side (``cmp_kv``, 320B rows) and
``quant_mode != 1`` raise ``NotImplementedError``.

Accuracy: checked against a torch reference over the dequantized cache, which
is the exact arithmetic above.  A value-exact comparison against the vendor is
not possible in this call form: its ``quant_mode=1`` cache is 608B/token with
group size 64 and a bf16 rope segment, so the two ops dequantize different
bytes.  The vendor comparison in ``benchmark/`` is therefore latency only.
"""

from flaggems_vllm.runtime.backend._ascend.a5.ops.mixed_quant_sparse_flash_mla.launcher import (  # noqa: E501
    PATHS,
    MixedQuantSparseFlashMlaPlan,
    mixed_quant_sparse_flash_mla,
)

__all__ = [
    "PATHS",
    "MixedQuantSparseFlashMlaPlan",
    "mixed_quant_sparse_flash_mla",
]
