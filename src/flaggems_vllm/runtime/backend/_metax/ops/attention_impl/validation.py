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

"""Validate the executable fields of an internal attention plan."""
from .scheduling import (
    MetaXAttentionPlan,
    MetaXGridOrder,
    MetaXKernelFamily,
    MetaXTaskMapper,
)


def validate_metax_attention_plan(plan: MetaXAttentionPlan) -> None:
    if not isinstance(plan, MetaXAttentionPlan):
        raise TypeError("MetaX launcher requires a MetaXAttentionPlan")
    expected_mappers = {
        MetaXKernelFamily.LEGACY: (MetaXTaskMapper.RECTANGULAR,),
        MetaXKernelFamily.COMPACT_WORKLIST: (MetaXTaskMapper.COMPACT_WORKLIST,),
        MetaXKernelFamily.SPLIT_KV: (MetaXTaskMapper.SPLIT_BATCH_HEAD,),
        MetaXKernelFamily.D256_TN_DIRECT: (
            MetaXTaskMapper.RECTANGULAR,
            MetaXTaskMapper.COMPACT_WORKLIST,
        ),
    }
    if plan.task_mapper not in expected_mappers.get(plan.family, ()):
        raise RuntimeError("MetaX family and task mapper disagree")
    if plan.grid_order not in (
        MetaXGridOrder.ROW_BATCH_HEAD,
        MetaXGridOrder.HEAD_BATCH_ROW,
    ):
        raise RuntimeError("MetaX plan has an unsupported grid order")
    split = plan.family is MetaXKernelFamily.SPLIT_KV
    compact = plan.task_mapper is MetaXTaskMapper.COMPACT_WORKLIST
    if split:
        if not 2 <= plan.max_splits <= 32:
            raise RuntimeError("Split-KV split count must be in [2, 32]")
        if (plan.split_kv_block_m, plan.split_kv_block_n) not in ((4, 16), (16, 16)):
            raise RuntimeError("Split-KV tile configuration is invalid")
        if (
            plan.split_kv_q_tiles <= 0
            or not 0 < plan.workspace_bytes <= 32 * 1024 * 1024
        ):
            raise RuntimeError(
                "Split-KV requires positive tasks and at most 32 MiB of workspace"
            )
    elif (
        plan.max_splits,
        plan.split_kv_block_m,
        plan.split_kv_block_n,
        plan.split_kv_q_tiles,
    ) != (1, 0, 0, 0):
        raise RuntimeError("Non-split routes must disable Split-KV metadata")
    if compact:
        if (
            plan.worklist_task_upper <= 0
            or plan.worklist_block_m not in (16, 32, 64, 128)
            or plan.workspace_bytes != plan.worklist_task_upper * 8
            or (plan.workspace_bytes > 1024 * 1024)
        ):
            raise RuntimeError("Compact worklist metadata is invalid")
        if plan.grid_order is not MetaXGridOrder.HEAD_BATCH_ROW:
            raise RuntimeError("Compact worklist requires head-first grid order")
    elif plan.worklist_task_upper != 0 or plan.worklist_block_m != 0:
        raise RuntimeError("Non-worklist routes must disable worklist metadata")
    if not split and (not compact) and (plan.workspace_bytes != 0):
        raise RuntimeError("Direct routes must not allocate planning workspace")


__all__ = ["validate_metax_attention_plan"]
