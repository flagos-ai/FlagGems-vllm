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

"""D256 production routing and generic varlen fallback for MetaX C550."""
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache


class MetaXKernelFamily(str, Enum):
    LEGACY = "legacy"
    COMPACT_WORKLIST = "compact_worklist"
    SPLIT_KV = "split_kv"
    D256_TN_DIRECT = "d256_tn_direct"


class MetaXTaskMapper(str, Enum):
    RECTANGULAR = "rectangular"
    COMPACT_WORKLIST = "compact_worklist"
    SPLIT_BATCH_HEAD = "split_batch_head"


class MetaXGridOrder(str, Enum):
    ROW_BATCH_HEAD = "row_batch_head"
    HEAD_BATCH_ROW = "head_batch_row"


@dataclass(frozen=True, slots=True)
class MetaXAttentionPlan:
    """Only values consumed by dispatch or workspace validation are retained."""

    family: MetaXKernelFamily
    task_mapper: MetaXTaskMapper
    grid_order: MetaXGridOrder
    max_splits: int = 1
    split_kv_block_m: int = 0
    split_kv_block_n: int = 0
    split_kv_q_tiles: int = 0
    worklist_task_upper: int = 0
    worklist_block_m: int = 0
    workspace_bytes: int = 0


D256_TN_BLOCK_M = 32
D256_TN_BLOCK_N = 128


class MetaXAttentionScheduler:
    """Select fixed production routes without reading GPU metadata on the host."""

    SPLIT_KV_BLOCK_M = 4
    SPLIT_KV_BLOCK_N = 16
    SPLIT_KV_SHORT_Q_BLOCK_M = 16
    SPLIT_KV_SHORT_Q_BLOCK_N = 16
    SPLIT_KV_SHORT_Q_MAX = 256
    SPLIT_KV_LONG_K_MIN = 32768
    D256_SHORT_Q_TARGET_K_BLOCKS_PER_SPLIT = 172
    SPLIT_KV_MAX_SPLITS = 32
    SPLIT_KV_MAX_WORKSPACE_BYTES = 32 * 1024 * 1024
    SPLIT_KV_AUTO_MAX_SPLITS = 16
    SPLIT_KV_AUTO_MIN_K_BLOCKS = 64
    SPLIT_KV_AUTO_MIN_SAVING_PERCENT = 10
    C550_NUM_SMS = 104
    COMPACT_WORKLIST_DESCRIPTOR_BYTES = 8
    COMPACT_WORKLIST_MAX_WORKSPACE_BYTES = 1 * 1024 * 1024
    COMPACT_WORKLIST_MIN_SAVING_NUMERATOR = 3
    COMPACT_WORKLIST_MIN_SAVING_DENOMINATOR = 2

    @staticmethod
    def _ceildiv(value: int, divisor: int) -> int:
        return (value + divisor - 1) // divisor

    @staticmethod
    def _direct_block_m(
        *, total_q: int, batch_size: int, num_heads: int, head_size: int
    ) -> int:
        """Mirror Direct's C550 BLOCK_M bucket without touching device state."""
        if total_q <= 0 or batch_size <= 0 or num_heads <= 0:
            return 0
        avg_rows_per_sm = total_q * num_heads / MetaXAttentionScheduler.C550_NUM_SMS
        avg_rows_per_batch = total_q / batch_size
        avg_rows_per_cta = min(avg_rows_per_batch, avg_rows_per_sm)
        if avg_rows_per_cta > 64:
            block_m = 128
        elif avg_rows_per_cta > 32:
            block_m = 64
        elif avg_rows_per_cta > 16:
            block_m = 32
        else:
            block_m = 16
        if head_size == 256 and block_m >= 64:
            return 64
        return block_m

    @staticmethod
    def _select_auto_splits(*, base_tasks: int, k_blocks: int, split_cap: int) -> int:
        """Choose D256 decode splits using the existing CTA-wave cost model."""
        resident_ctas = 2
        capacity = MetaXAttentionScheduler.C550_NUM_SMS * resident_ctas
        if base_tasks <= 0 or k_blocks <= 0 or split_cap < 2:
            return 0
        wave_fill_splits = MetaXAttentionScheduler._ceildiv(capacity, base_tasks)
        target_blocks = 5
        kv_parallel_splits = MetaXAttentionScheduler._ceildiv(k_blocks, target_blocks)
        selected = min(
            max(2, wave_fill_splits, kv_parallel_splits), split_cap, k_blocks
        )
        while selected > 2 and MetaXAttentionScheduler._ceildiv(
            k_blocks, selected
        ) == MetaXAttentionScheduler._ceildiv(k_blocks, selected - 1):
            selected -= 1
        one_pass_waves = MetaXAttentionScheduler._ceildiv(base_tasks, capacity)
        split_waves = MetaXAttentionScheduler._ceildiv(base_tasks * selected, capacity)
        one_pass_cost = one_pass_waves * k_blocks
        selected_cost = (
            split_waves * MetaXAttentionScheduler._ceildiv(k_blocks, selected)
            + selected
        )
        if selected < 2 or selected_cost * 100 > one_pass_cost * (
            100 - MetaXAttentionScheduler.SPLIT_KV_AUTO_MIN_SAVING_PERCENT
        ):
            return 0
        return selected

    @staticmethod
    @lru_cache(maxsize=128)
    def build(
        *,
        is_bfloat16: bool = False,
        is_paged: bool = False,
        block_size: int = 0,
        is_cu_seqlens_q: bool = False,
        max_seqlen_q: int = 0,
        max_seqlen_k: int = 0,
        total_q: int = 0,
        batch_size: int = 0,
        num_heads: int = 0,
        num_heads_k: int = 0,
        head_size: int = 0,
        is_causal: bool = False,
        is_local: bool = False,
        is_dropout: bool = False,
        is_alibi: bool = False,
        is_softcap: bool = False,
        seqlenq_ngroups_swapped: bool = False,
        num_splits: int = 0,
    ) -> MetaXAttentionPlan:
        if isinstance(num_splits, bool) or not isinstance(num_splits, int):
            raise TypeError(
                f"num_splits must be an integer in [0, 32], got {num_splits!r}"
            )
        if num_splits < 0 or num_splits > MetaXAttentionScheduler.SPLIT_KV_MAX_SPLITS:
            raise ValueError(f"num_splits must be in [0, 32], got {num_splits}")
        head_first_eligible = (
            is_bfloat16
            and is_paged
            and is_cu_seqlens_q
            and (max_seqlen_q >= 128)
            and (total_q > 0)
            and (batch_size >= 1)
            and (head_size == 256)
            and is_causal
            and (not is_local)
            and (not is_dropout)
            and (not is_alibi)
            and (not is_softcap)
        )
        direct_block_m = MetaXAttentionScheduler._direct_block_m(
            total_q=total_q,
            batch_size=batch_size,
            num_heads=num_heads,
            head_size=head_size,
        )
        worklist_block_m = direct_block_m
        worklist_task_upper = (
            MetaXAttentionScheduler._ceildiv(total_q, worklist_block_m) + batch_size - 1
            if worklist_block_m > 0
            else 0
        )
        legacy_worklist_tasks = (
            batch_size
            * MetaXAttentionScheduler._ceildiv(max_seqlen_q, worklist_block_m)
            if worklist_block_m > 0
            else 0
        )
        legacy_worklist_workspace_bytes = (
            worklist_task_upper
            * MetaXAttentionScheduler.COMPACT_WORKLIST_DESCRIPTOR_BYTES
        )
        compact_worklist_eligible = (
            head_first_eligible
            and batch_size >= 2
            and (num_splits <= 1)
            and (num_heads > 0)
            and (num_heads_k > 0)
            and (num_heads % num_heads_k == 0)
            and (worklist_task_upper > 0)
            and (
                legacy_worklist_tasks
                * MetaXAttentionScheduler.COMPACT_WORKLIST_MIN_SAVING_DENOMINATOR
                >= worklist_task_upper
                * MetaXAttentionScheduler.COMPACT_WORKLIST_MIN_SAVING_NUMERATOR
            )
            and (
                legacy_worklist_workspace_bytes
                <= MetaXAttentionScheduler.COMPACT_WORKLIST_MAX_WORKSPACE_BYTES
            )
        )
        d256_tn_eligible = (
            is_bfloat16
            and is_paged
            and (
                block_size == 32
                or (block_size == 16 and num_heads == 4 and (num_heads_k == 1))
            )
            and is_cu_seqlens_q
            and (total_q > 0)
            and (batch_size > 0)
            and (num_heads > 0)
            and (num_heads_k > 0)
            and (num_heads % num_heads_k == 0)
            and (D256_TN_BLOCK_M % (num_heads // num_heads_k) == 0)
            and (head_size == 256)
            and (direct_block_m == 64 or (block_size == 16 and max_seqlen_q >= 1024))
            and is_causal
            and (not is_local)
            and (not is_dropout)
            and (not is_alibi)
            and (not is_softcap)
            and (not seqlenq_ngroups_swapped)
            and (num_splits <= 1)
        )
        split_workspace_per_split = total_q * num_heads * (head_size + 1) * 4
        workspace_split_cap = (
            MetaXAttentionScheduler.SPLIT_KV_MAX_WORKSPACE_BYTES
            // split_workspace_per_split
            if split_workspace_per_split > 0
            else 0
        )
        short_q_long_kv = (
            is_cu_seqlens_q
            and MetaXAttentionScheduler.SPLIT_KV_BLOCK_M
            < max_seqlen_q
            <= MetaXAttentionScheduler.SPLIT_KV_SHORT_Q_MAX
            and (max_seqlen_k >= MetaXAttentionScheduler.SPLIT_KV_LONG_K_MIN)
        )
        split_kv_capable = (
            is_bfloat16
            and is_paged
            and (is_cu_seqlens_q or seqlenq_ngroups_swapped)
            and (
                0 < max_seqlen_q <= MetaXAttentionScheduler.SPLIT_KV_BLOCK_M
                or short_q_long_kv
            )
            and (max_seqlen_k >= 0)
            and (total_q > 0)
            and (batch_size > 0)
            and (num_heads > 0)
            and (num_heads_k > 0)
            and (num_heads % num_heads_k == 0)
            and (head_size in (64, 128, 192, 256))
            and (is_causal or seqlenq_ngroups_swapped)
            and (not is_local)
            and (not is_dropout)
            and (not is_alibi)
            and (not is_softcap)
        )
        split_kv_block_m = (
            MetaXAttentionScheduler.SPLIT_KV_SHORT_Q_BLOCK_M
            if short_q_long_kv
            else MetaXAttentionScheduler.SPLIT_KV_BLOCK_M
        )
        split_kv_block_n = (
            MetaXAttentionScheduler.SPLIT_KV_SHORT_Q_BLOCK_N
            if short_q_long_kv
            else MetaXAttentionScheduler.SPLIT_KV_BLOCK_N
        )
        split_kv_q_tiles = MetaXAttentionScheduler._ceildiv(
            max_seqlen_q, split_kv_block_m
        )
        split_kv_base_tasks = batch_size * num_heads * split_kv_q_tiles
        split_kv_k_blocks = MetaXAttentionScheduler._ceildiv(
            max_seqlen_k, split_kv_block_n
        )
        explicit_split_kv = split_kv_capable and num_splits >= 2
        auto_split_kv = (
            split_kv_capable
            and num_splits == 0
            and seqlenq_ngroups_swapped
            and (head_size == 256)
            and (
                split_kv_k_blocks >= MetaXAttentionScheduler.SPLIT_KV_AUTO_MIN_K_BLOCKS
            )
        )
        auto_split_cap = min(
            MetaXAttentionScheduler.SPLIT_KV_AUTO_MAX_SPLITS, workspace_split_cap
        )
        auto_selected_splits = MetaXAttentionScheduler._select_auto_splits(
            base_tasks=split_kv_base_tasks,
            k_blocks=split_kv_k_blocks,
            split_cap=auto_split_cap if auto_split_kv else 0,
        )
        selected_splits = (
            min(num_splits, workspace_split_cap)
            if explicit_split_kv
            else auto_selected_splits
        )
        if explicit_split_kv and short_q_long_kv and (head_size == 256):
            d256_short_q_splits = MetaXAttentionScheduler._ceildiv(
                split_kv_k_blocks,
                MetaXAttentionScheduler.D256_SHORT_Q_TARGET_K_BLOCKS_PER_SPLIT,
            )
            selected_splits = min(selected_splits, d256_short_q_splits)
        if selected_splits < 2:
            selected_splits = 0
        split_kv_enabled = selected_splits >= 2
        d256_tn_enabled = d256_tn_eligible and (not split_kv_enabled)
        if d256_tn_enabled:
            worklist_block_m = D256_TN_BLOCK_M
            worklist_task_upper = (
                MetaXAttentionScheduler._ceildiv(total_q, worklist_block_m)
                + batch_size
                - 1
            )
        worklist_workspace_bytes = (
            worklist_task_upper
            * MetaXAttentionScheduler.COMPACT_WORKLIST_DESCRIPTOR_BYTES
        )
        compact_worklist_enabled = (
            compact_worklist_eligible
            and (not split_kv_enabled)
            and (
                worklist_workspace_bytes
                <= MetaXAttentionScheduler.COMPACT_WORKLIST_MAX_WORKSPACE_BYTES
            )
        )
        family = (
            MetaXKernelFamily.SPLIT_KV
            if split_kv_enabled
            else (
                MetaXKernelFamily.D256_TN_DIRECT
                if d256_tn_enabled
                else (
                    MetaXKernelFamily.COMPACT_WORKLIST
                    if compact_worklist_enabled
                    else MetaXKernelFamily.LEGACY
                )
            )
        )
        return MetaXAttentionPlan(
            family=family,
            task_mapper=(
                MetaXTaskMapper.SPLIT_BATCH_HEAD
                if split_kv_enabled
                else (
                    MetaXTaskMapper.COMPACT_WORKLIST
                    if compact_worklist_enabled
                    else MetaXTaskMapper.RECTANGULAR
                )
            ),
            grid_order=(
                MetaXGridOrder.HEAD_BATCH_ROW
                if head_first_eligible
                else MetaXGridOrder.ROW_BATCH_HEAD
            ),
            max_splits=selected_splits if split_kv_enabled else 1,
            split_kv_block_m=split_kv_block_m if split_kv_enabled else 0,
            split_kv_block_n=split_kv_block_n if split_kv_enabled else 0,
            split_kv_q_tiles=split_kv_q_tiles if split_kv_enabled else 0,
            worklist_task_upper=worklist_task_upper if compact_worklist_enabled else 0,
            worklist_block_m=worklist_block_m if compact_worklist_enabled else 0,
            workspace_bytes=(
                selected_splits * split_workspace_per_split
                if split_kv_enabled
                else worklist_workspace_bytes if compact_worklist_enabled else 0
            ),
        )


__all__ = [
    "MetaXAttentionPlan",
    "MetaXAttentionScheduler",
    "MetaXGridOrder",
    "MetaXKernelFamily",
    "MetaXTaskMapper",
    "D256_TN_BLOCK_M",
    "D256_TN_BLOCK_N",
]
