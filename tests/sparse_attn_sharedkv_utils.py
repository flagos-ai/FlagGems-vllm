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

"""The six validated DSV4 SparseAttnSharedKV cases and their input construction.

These are the fixed shapes the operator's six schedules are validated on: three
decode cases at KV 8193 and three prefill cases at Q=KV=8192, constructed with a
fixed seed so a measurement can be reproduced.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from functools import lru_cache

import torch

SEED = 43
Q_HEADS = 64
KV_HEADS = 1
HEAD_DIM = 512
PAGE_SIZE = 128
SOFTMAX_SCALE = 0.04419417
ATOL = RTOL = 2e-2


@dataclasses.dataclass(frozen=True)
class SparseAttnSharedkvCase:
    name: str
    mode: str
    q_len: int
    kv_len: int
    cmp_ratio: int = 0
    topk: int = 0

    @property
    def has_cmp(self) -> bool:
        return self.mode != "SWA"

    @property
    def has_sparse(self) -> bool:
        return self.mode == "SCFA"

    @property
    def is_decode(self) -> bool:
        return self.q_len == 1


CASES = (
    SparseAttnSharedkvCase("scfa_decode", "SCFA", 1, 8193, 4, 512),
    SparseAttnSharedkvCase("swa_decode", "SWA", 1, 8193),
    SparseAttnSharedkvCase("cfa_decode", "CFA", 1, 8193, 128),
    SparseAttnSharedkvCase("scfa_prefill", "SCFA", 8192, 8192, 4, 512),
    SparseAttnSharedkvCase("swa_prefill", "SWA", 8192, 8192),
    SparseAttnSharedkvCase("cfa_prefill", "CFA", 8192, 8192, 128),
)


def case_by_name(name: str) -> SparseAttnSharedkvCase:
    return next(case for case in CASES if case.name == name)


@lru_cache(maxsize=1)
def official_operator():
    """The vLLM-Ascend AscendC operator used as the reference."""
    try:
        import vllm_ascend.utils

        if not vllm_ascend.utils.enable_custom_op():
            return None
        return torch.ops._C_ascend.npu_sparse_attn_sharedkv
    except (ImportError, AttributeError):
        return None


@lru_cache(maxsize=1)
def metadata_operator():
    try:
        import vllm_ascend.utils

        if not vllm_ascend.utils.enable_custom_op():
            return None
        return torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata
    except (ImportError, AttributeError):
        return None


def _paged_tensor(
    token_count: int,
    generator: torch.Generator,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    page_count = math.ceil(token_count / PAGE_SIZE)
    block_table = torch.randperm(
        page_count, generator=generator, dtype=torch.int32, device="cpu"
    ).view(1, page_count)
    kv = torch.rand(
        (page_count, PAGE_SIZE, KV_HEADS, HEAD_DIM),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    return (kv * 15 - 5).to(torch.bfloat16).to(device), block_table.to(device)


def make_inputs(
    case: SparseAttnSharedkvCase,
    device: str = "npu",
    metadata_op: Callable[..., torch.Tensor] | None = None,
) -> dict[str, object]:
    if metadata_op is None:
        metadata_op = metadata_operator()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    q = torch.rand(
        (case.q_len, Q_HEADS, HEAD_DIM),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    q = (q * 20 - 10).to(torch.bfloat16).to(device)
    ori_kv, ori_block_table = _paged_tensor(case.kv_len, generator, device)
    cu_seqlens_q = torch.tensor([0, case.q_len], dtype=torch.int32, device=device)
    seqused_kv = torch.tensor([case.kv_len], dtype=torch.int32, device=device)

    cmp_kv = cmp_block_table = cmp_sparse_indices = None
    if case.has_cmp:
        cmp_tokens = case.kv_len // case.cmp_ratio
        cmp_kv, cmp_block_table = _paged_tensor(cmp_tokens, generator, device)
        if case.has_sparse:
            indices = torch.full(
                (case.q_len, KV_HEADS, case.topk), -1, dtype=torch.int32, device="cpu"
            )
            for q_pos in range(case.q_len):
                visible = (
                    case.kv_len // case.cmp_ratio
                    if case.q_len == 1
                    else (q_pos + 1) // case.cmp_ratio
                )
                count = min(visible, case.topk)
                if count:
                    indices[q_pos, 0, :count] = torch.randperm(
                        visible, generator=generator, dtype=torch.int32, device="cpu"
                    )[:count]
            cmp_sparse_indices = indices.to(device)

    sinks = torch.rand(
        (Q_HEADS,), generator=generator, dtype=torch.float32, device="cpu"
    ).to(device)
    inputs = {
        "q": q,
        "ori_kv": ori_kv,
        "cmp_kv": cmp_kv,
        "cmp_sparse_indices": cmp_sparse_indices,
        "ori_block_table": ori_block_table,
        "cmp_block_table": cmp_block_table,
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_kv": seqused_kv,
        "sinks": sinks,
        "softmax_scale": SOFTMAX_SCALE,
        "cmp_ratio": case.cmp_ratio,
        "ori_mask_mode": 4,
        "cmp_mask_mode": 3,
        "ori_win_left": 127,
        "ori_win_right": 0,
        "layout_q": "TND",
        "layout_kv": "PA_ND",
    }
    inputs["metadata"] = metadata_op(
        num_heads_q=Q_HEADS,
        num_heads_kv=KV_HEADS,
        head_dim=HEAD_DIM,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        batch_size=1,
        max_seqlen_q=case.q_len,
        max_seqlen_kv=case.kv_len,
        cmp_topk=case.topk,
        cmp_ratio=case.cmp_ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
        has_ori_kv=True,
        has_cmp_kv=case.has_cmp,
        device=device,
    )
    return inputs


def output_tensor(value):
    return value[0] if isinstance(value, tuple) else value


def shape_label(case: SparseAttnSharedkvCase) -> str:
    return (
        f"{case.mode}/TND/BF16; Q=[{case.q_len},64,512], KV={case.kv_len}, "
        f"ratio={case.cmp_ratio}, topk={case.topk}"
    )
