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

"""BNSD FlashAttention forward (Ascend 950DT), dense/causal router.

A plain ``tl.dot`` + ``tl.max/exp/sum`` forward kernel that lets the compiler
split the Cube/Vector pipeline (``--set-cv-pipeline-mode``) rather than a
hand-written ``cann.scope``/``sync_block_*`` schedule. On an idle 950DT
(CANN 9.2.1 / triton_ascend 3.6) it passes 240/240 compatible shapes against
``torch_npu.npu_fusion_attention`` (``aclnnFlashAttentionScoreV4``) at a mean
0.786x and up to 413 TFLOPS (95% of cube peak), max error 1.95e-03.

Distinct from the varlen/paged ``flash_attn_varlen_func`` in
``_ascend/fused/attention.py``: this is a dense BNSD interface,
``q[B, Hq, S, D]``, ``k/v[B, Hkv, S, D]``, fp16, ``D == 64``, ``Hq % Hkv == 0``,
``Sq == Skv``.

    from flaggems_vllm.runtime.backend._ascend.ops import flash_attention_fwd
    out = flash_attention_fwd(q, k, v, is_causal=True)

The router dispatches to the dense or causal kernel. They were one kernel with
a ``CAUSAL: tl.constexpr`` until the causal side grew its own structure
(de-peeled mask, its own PEEL gate) and the dense side its own tile autotune.
They share ``build_tile_order``, ``_max_nan`` and ``_COMPILE_OPTS`` from
``fa_naive``; everything else is separate so each can be tuned without
regressing the other.
"""

from .fa_causal import flash_attention_fwd_causal
from .fa_dense import flash_attention_fwd_dense

__all__ = [
    "flash_attention_fwd",
    "flash_attention_fwd_causal",
    "flash_attention_fwd_dense",
]


def flash_attention_fwd(q, k, v, is_causal=False, **kw):
    """Dense/causal router. Extra kwargs go to the selected kernel.

    ``autotune`` is dense-only (the causal kernel has nothing to tune -- see
    ``fa_causal._pick``), so it is dropped on the causal path rather than
    raising, which keeps the two call signatures interchangeable.

    ``combine_batch`` was a knob on the retired hand-written CV kernel; catching
    it by name gives a readable message instead of the backend's launch-time
    ``KeyError`` that names neither the flag's origin nor the replacement.
    """
    if "combine_batch" in kw:
        raise TypeError(
            "flash_attention_fwd() has no `combine_batch`: it was a knob on "
            "the retired hand-written CV kernel. The router selects its own "
            "tile/pipeline configuration (fa_causal._pick / fa_dense._CONFIGS)."
            " Drop the argument.")
    if is_causal:
        kw.pop("autotune", None)
        return flash_attention_fwd_causal(q, k, v, **kw)
    return flash_attention_fwd_dense(q, k, v, **kw)
